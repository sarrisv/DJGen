import json
import logging
import os
from collections import defaultdict
from math import prod
from typing import Dict, Any, List, Set, Tuple

import dask.dataframe as dd
from dask.dataframe import DataFrame

logger = logging.getLogger("djp")


def _get_used_attrs_from_plan(
    execution_plan: Dict[str, Any],
) -> Dict[str, Set[str]]:
    join_attrs_per_rel: Dict[str, Set[str]] = defaultdict(set)

    for stage in execution_plan.get("stages", []):
        for key_list in stage.get("on_attributes", []):
            for full_key in key_list:
                rel_name, attr_name = full_key.split("_", 1)
                join_attrs_per_rel[rel_name].add(attr_name)

    return join_attrs_per_rel


def _load_datafiles(
    rel_paths: Dict[str, str], join_attrs_per_rel: Dict[str, Set[str]]
) -> Dict[str, dd.DataFrame]:
    dfs: Dict[str, dd.DataFrame] = {}
    for name, path in rel_paths.items():
        df = dd.read_parquet(path)

        drop_set = set(col for col in df.columns if col not in join_attrs_per_rel[name])
        df = df.drop(columns=drop_set)

        rename_map = {attr: f"{name}_{attr}" for attr in df.columns}
        dfs[name] = df.rename(columns=rename_map)

    return dfs


def _perform_binary_join_stage(
    dfs: Dict[str, dd.DataFrame], stage: Dict[str, Any]
) -> dd.DataFrame:
    rel0 = stage["input_relations"][0]
    rel1 = stage["input_relations"][1]

    rel0_df = dfs[rel0]
    rel1_df = dfs[rel1]

    rel0_keys = stage["on_attributes"][0]
    rel1_keys = stage["on_attributes"][1]

    # cartesian product for join branch merging
    if not rel0_keys:
        rel0_df["key"] = 1
        rel1_df["key"] = 1
        return dd.merge(rel0_df, rel1_df, on="key", how="inner").drop(columns=["key"])

    return dd.merge(
        rel0_df, rel1_df, left_on=rel0_keys, right_on=rel1_keys, how="inner"
    )


def _perform_nary_join_stage(
    dfs: Dict[str, dd.DataFrame], stage: Dict[str, Any]
) -> dd.DataFrame:
    rels = stage["input_relations"].copy()
    join_keys = stage["on_attributes"].copy()

    rel0 = rels.pop(0)
    rel0_df = dfs[rel0]
    rel0_keys = join_keys.pop(0)

    for i, rel1 in enumerate(rels):
        rel1_df = dfs[rel1]
        rel1_keys = join_keys[i]

        if not rel0_keys:
            rel0_df["key"] = 1
            rel1_df["key"] = 1
            rel0_df = dd.merge(rel0_df, rel1_df, on="key", how="inner").drop(
                columns=["key"]
            )
        else:
            if rel0 == rel1:
                select_filter = rel0_df[rel0_keys[0]] == rel1_df[rel1_keys[0]]
                rel0_df = rel0_df[select_filter]
            else:
                rel0_df = dd.merge(
                    rel0_df,
                    rel1_df,
                    left_on=rel0_keys,
                    right_on=rel1_keys,
                    how="inner",
                )

    return rel0_df


def _analyze_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    execution_plan = plan["execution_plan"]
    catalog = plan["catalog"]

    rel_paths = {name: details["location"] for name, details in catalog.items()}
    join_attrs_per_rel = _get_used_attrs_from_plan(execution_plan)
    dfs = _load_datafiles(rel_paths, join_attrs_per_rel)

    rel_composition: Dict[str, Set[str]] = defaultdict(set)
    for rel_name in catalog:
        rel_composition[rel_name].add(rel_name)

    analysis_results = {"stages": []}
    total_intermediates = 0
    join_type = execution_plan["type"]
    for i, stage in enumerate(execution_plan["stages"]):
        stage_result = {"stage": i}
        try:
            logger.debug(
                f"\t\t\t\tAnalyzing stage {i}: Joining {stage['input_relations']} on {stage['on_attributes']}"
            )

            if join_type == "binary":
                result_df = _perform_binary_join_stage(dfs, stage)
            else:
                result_df = _perform_nary_join_stage(dfs, stage)

            dfs[stage["name"]] = result_df

            new_composition = set()
            for input_relation in stage["input_relations"]:
                new_composition.update(rel_composition[input_relation])
            rel_composition[stage["name"]] = new_composition
            contains = sorted(list(new_composition))

            output_size = len(result_df)
            if join_type == "binary":
                total_intermediates += output_size
            else:
                total_intermediates = output_size
            selectivity = output_size / prod(
                (len(dfs[rel]) for rel in stage["input_relations"])
            )

            stage_result = {
                "stage_id": stage["stage_id"],
                "contains": list(contains),
                "output_size": output_size,
                "total_intermediates": total_intermediates,
                "selectivity": max(round(selectivity, 4), 0.0001),
            }
            logger.debug(f"\t\t\t\t\tStage {i} completed: {output_size} rows")

        except Exception as e:
            error_message = f"{type(e).__name__} - {e}"
            logger.debug(f"\t\t\t\t\tERROR analyzing stage {i}: {error_message}")
            stage_result["error"] = error_message

        analysis_results["stages"].append(stage_result)

    return analysis_results


def generate_analysis_for_iteration(output_dir: str) -> None:
    logger.debug(f"\t\tStarting analysis for {output_dir}...")
    plans_dir = os.path.join(output_dir, "plans")

    if not os.path.exists(plans_dir):
        logger.debug("\t\tSKIPPING, no plans directory found to analyze")
        return

    for plan_file in os.listdir(plans_dir):
        if not plan_file.endswith(".json"):
            continue

        plan_path = os.path.join(plans_dir, plan_file)
        logger.debug(f"\t\t\tAnalyzing plan: {plan_file}")

        plan = {}
        with open(plan_path, "r") as f:
            plan = json.load(f)

        analysis_results = _analyze_plan(plan)
        plan["analysis"] = analysis_results

        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=4)
        logger.debug(f"\t\t\t\t...analysis appended to {plan_path}")
    #
