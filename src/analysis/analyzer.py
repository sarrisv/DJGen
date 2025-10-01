import json
import logging
import os
from collections import defaultdict
from math import prod
from typing import Dict, Any, List, Set, Tuple

import dask.dataframe as dd
from dask.dataframe import DataFrame

logger = logging.getLogger("djp")


def _create_binary_execution_plan(plan_stages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Set[str]]]:
    execution_stages: List[Dict[str, Any]] = []
    earliest_stage: Dict[frozenset[str], int] = {}
    latest_stage: Dict[str, int] = {}
    result_counter = 1

    # used for removing unused columns when loading data
    join_attrs_per_table: Dict[str, Set[str]] = defaultdict(set)
    # used to track which tables contain which tables
    table_composition: Dict[str, Set[str]] = defaultdict(set)
    for stage in plan_stages:
        for table in stage["tables"]:
            table_composition[table].add(table)
            join_attrs_per_table[table].add(stage["on_attribute"])

    # used to track join branches so we can join them together at the end if needed
    join_branches = []

    for stage in plan_stages:
        attr = stage["on_attribute"]
        base_table0 = stage["tables"][0]
        base_table1 = stage["tables"][1]

        table_pair = frozenset({base_table0, base_table1})
        earliest_stage_idx = earliest_stage.get(table_pair, None)

        # Case 1: this pair of tables have been joined before, add the attr to the earliest stage they both exist
        if earliest_stage_idx is not None:
            earliest_execution_stage = execution_stages[earliest_stage_idx]

            table0_key = f"{base_table0}_{attr}"
            table1_key = f"{base_table1}_{attr}"

            # table0 is / is in tables[0] => table1 is / is in tables[1]
            if base_table0 in table_composition[earliest_execution_stage["tables"][0]]:
                earliest_execution_stage["on_attributes"][0].append(table0_key)
                earliest_execution_stage["on_attributes"][1].append(table1_key)

            # table1 is / is in tables[0] => table0 is / is in tables[1]
            else:
                earliest_execution_stage["on_attributes"][0].append(table1_key)
                earliest_execution_stage["on_attributes"][1].append(table0_key)

            continue

        # Case 2: this pair of tables has not been joined before, create a new execution stage
        table0_stage = latest_stage.get(base_table0, None)
        table1_stage = latest_stage.get(base_table1, None)

        current_table0 = (
            base_table0
            if table0_stage is None
            else execution_stages[table0_stage]["name"]
        )
        current_table1 = (
            base_table1
            if table1_stage is None
            else execution_stages[table1_stage]["name"]
        )

        result_name = f"result_{result_counter}"

        table0_contains = table_composition[current_table0]
        table1_contains = table_composition[current_table1]
        result_contains = table0_contains.union(table1_contains)
        table_composition[result_name].update(result_contains)

        stage_idx = len(execution_stages)
        execution_stages.append(
            {
                "type": stage["type"],
                "name": result_name,
                "tables": [current_table0, current_table1],
                "on_attributes": [
                    [f"{base_table0}_{attr}"],
                    [f"{base_table1}_{attr}"],
                ],
                "contains": table_composition[result_name],
            }
        )
        result_counter += 1

        # Update latest stage for all the tables in the new result
        for table in result_contains:
            latest_stage[table] = stage_idx

        # Create an earliest stage entry for all new pairs of tables
        # set ensures order does not matter
        for left_table in table0_contains:
            for right_table in table1_contains:
                new_pair = frozenset({left_table, right_table})
                if new_pair not in earliest_stage:
                    earliest_stage[new_pair] = stage_idx

        # Clean up join branches
        if current_table0 in join_branches:
            join_branches.remove(current_table0)
        if current_table1 in join_branches:
            join_branches.remove(current_table1)
        join_branches.append(result_name)

    # If there are dangling join branches join them via a cartesian product so we end with a single final result
    # Possible if analyzing a bushy join plan that produces disconnected results
    # Example with two stage bushy join plan -- stage 1: rel0 join rel1 -> result_1, stage 2: rel2 join rel3 -> result_2, need a final cartesian product to produce final output result_3
    while len(join_branches) > 1:
        table0 = join_branches.pop(0)
        table1 = join_branches.pop(0)

        result_name = f"result_{result_counter}"
        result_contains = table_composition[table0].union(table_composition[table1])
        table_composition[result_name].update(result_contains)

        execution_stages.append(
            {
                "type": "binary",
                "name": result_name,
                "tables": [table0, table1],
                "on_attributes": [],
                "contains": table_composition[result_name],
            }
        )
        result_counter += 1

        join_branches.append(result_name)

    return execution_stages, join_attrs_per_table


def _create_nary_execution_plan(plan_stages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Set[str]]]:
    execution_stages: List[Dict[str, Any]] = []
    latest_stage: Dict[str, int] = {}
    result_counter = 1

    # used for removing unused columns when loading data
    join_attrs_per_table: Dict[str, Set[str]] = defaultdict(set)

    # used to track which tables contain which tables
    table_composition: Dict[str, Set[str]] = defaultdict(set)
    for stage in plan_stages:
        attr = stage["on_attribute"]
        for table in stage["tables"]:
            table_composition[table].add(table)
            join_attrs_per_table[table].add(attr)

    # used to track join branches so we can join them together at the end if needed
    join_branches = []

    for stage in plan_stages:
        attr = stage["on_attribute"]
        base_tables = stage["tables"]

        result_name = f"result_{result_counter}"
        current_tables = []
        for table in base_tables:
            table = (
                execution_stages[latest_stage[table]]["name"]
                if table in latest_stage
                else table
            )
            current_tables.append(table)
            table_composition[result_name].update(table_composition[table])
        on_attributes = [[f"{table}_{attr}"] for table in base_tables]

        stage_idx = len(execution_stages)
        execution_stages.append(
            {
                "type": stage["type"],
                "name": result_name,
                "tables": current_tables,
                "on_attributes": on_attributes,
                "contains": table_composition[result_name],
            }
        )

        join_branches.append(result_name)

        for table in current_tables:
            if table in join_branches:
                join_branches.remove(table)

        for table in table_composition[result_name]:
            latest_stage[table] = stage_idx

        result_counter += 1

    if len(join_branches) > 1:
        result_name = f"result_{result_counter}"
        for table in join_branches:
            table_composition[result_name].update(table_composition[table])

        execution_stages.append(
            {
                "type": "n-ary",
                "name": f"result_{result_counter}",
                "tables": join_branches,
                "on_attributes": [],
                "contains": table_composition[result_name],
            }
        )

    return execution_stages, join_attrs_per_table


def _load_datafiles(table_paths: Dict[str, str], join_attrs_per_table: Dict[str, Set[str]]) -> Dict[str, dd.DataFrame]:
    dfs: Dict[str, dd.DataFrame] = {}
    for name, path in table_paths.items():
        df = dd.read_parquet(path)

        drop_set = set(
            col for col in df.columns if col not in join_attrs_per_table[name]
        )
        df = df.drop(columns=drop_set)

        rename_map = {attr: f"{name}_{attr}" for attr in df.columns}
        dfs[name] = df.rename(columns=rename_map)

    return dfs


def _perform_binary_join_stage(dfs: Dict[str, dd.DataFrame], stage: Dict[str, Any]) -> dd.DataFrame:
    table0 = stage["tables"][0]
    table1 = stage["tables"][1]

    table0_df = dfs[table0]
    table1_df = dfs[table1]

    table0_keys = stage["on_attributes"][0]
    table1_keys = stage["on_attributes"][1]

    # cartesian product for join branch merging
    if not table0_keys:
        table0_df["key"] = 1
        table1_df["key"] = 1
        return dd.merge(table0_df, table1_df, on="key", how="inner").drop(
            columns=["key"]
        )

    return dd.merge(
        table0_df, table1_df, left_on=table0_keys, right_on=table1_keys, how="inner"
    )


def _perform_nary_join_stage(dfs: Dict[str, dd.DataFrame], stage: Dict[str, Any]) -> dd.DataFrame:
    tables = stage["tables"].copy()
    join_keys = stage["on_attributes"].copy()

    table0 = tables.pop(0)
    table0_df = dfs[table0]
    table0_keys = join_keys.pop(0)

    for i, table1 in enumerate(tables):
        table1_df = dfs[table1]
        table1_keys = join_keys[i]

        if not table0_keys:
            table0_df["key"] = 1
            table1_df["key"] = 1
            table0_df = dd.merge(table0_df, table1_df, on="key", how="inner").drop(
                columns=["key"]
            )
        else:
            if table0 == table1:
                select_filter = table0_df[table0_keys[0]] == table1_df[table1_keys[0]]
                table0_df = table0_df[select_filter]
            else:
                table0_df = dd.merge(
                    table0_df,
                    table1_df,
                    left_on=table0_keys,
                    right_on=table1_keys,
                    how="inner",
                )

    return table0_df


def _analyze_plan(plan_path: str, table_paths: Dict[str, str]) -> Dict[str, Any]:
    with open(plan_path) as f:
        plan_stages = json.load(f)

    if plan_stages[0]["type"] == "binary":
        execution_plan, join_attrs_per_table = _create_binary_execution_plan(
            plan_stages
        )
    else:
        execution_plan, join_attrs_per_table = _create_nary_execution_plan(plan_stages)

    dfs = _load_datafiles(table_paths, join_attrs_per_table)

    analysis_results = {
        "base_relations": {name: len(df) for name, df in dfs.items()},
        "stages": [],
    }

    total_intermediates = 0
    for i, stage in enumerate(execution_plan):
        stage_result = {"stage": i, "type": stage["type"]}
        try:
            logger.debug(
                f"\t\t\t\tAnalyzing stage {i}: Joining {stage['tables']} on {stage['on_attributes']}"
            )

            if stage["type"] == "binary":
                result_df = _perform_binary_join_stage(dfs, stage)
            else:
                result_df = _perform_nary_join_stage(dfs, stage)

            dfs[stage["name"]] = result_df
            output_size = len(result_df)
            if stage["type"] == "binary":
                total_intermediates += output_size
            else:
                total_intermediates = output_size
            selectivity = output_size / prod(
                (len(dfs[table]) for table in stage["tables"])
            )

            stage_result.update(
                {
                    "name": stage["name"],
                    "tables": stage["tables"],
                    "on_attributes": stage["on_attributes"],
                    "contains": list(stage["contains"]),
                    "output_size": output_size,
                    "total_intermediates": total_intermediates,
                    "selectivity": max(round(selectivity, 4), 0.0001),
                }
            )

            logger.debug(f"\t\t\t\t\tStage {i} completed: {output_size} rows")

        except Exception as e:
            error_message = f"{type(e).__name__} - {e}"
            logger.debug(f"\t\t\t\t\tERROR analyzing stage {i}: {error_message}")
            stage_result["error"] = error_message

        analysis_results["stages"].append(stage_result)

    return analysis_results


def generate_analysis_for_iteration(output_dir: str) -> None:
    logger.debug(f"\t\tStarting analysis for {output_dir}...")
    data_dir = os.path.join(output_dir, "data")
    plans_dir = os.path.join(output_dir, "plans")
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    table_paths = {
        os.path.basename(p): os.path.join(data_dir, p)
        for p in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, p))
    }
    if not table_paths:
        logger.debug("\t\tSKIPPING, no data found to analyze")
        return

    for plan_file in os.listdir(plans_dir):
        if not plan_file.endswith(".json") or "base" in plan_file:
            continue

        plan_name = plan_file.replace(".json", "")
        logger.debug(f"\t\t\tAnalyzing plan: {plan_name}")
        plan_path = os.path.join(plans_dir, plan_file)

        plan_analysis = _analyze_plan(plan_path, table_paths)

        output_path = os.path.join(analysis_dir, f"{plan_name}_analysis.json")
        with open(output_path, "w") as f:
            json.dump(plan_analysis, f, indent=4)
        logger.debug(f"\t\t\t\t...written to {output_path}")
