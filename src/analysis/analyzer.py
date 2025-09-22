import os
import json
import logging
from math import prod
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import dask.dataframe as dd
from dask.dataframe import DataFrame

logger = logging.getLogger("djp")

def _create_binary_execution_plan(plan_stages):
    execution_stages: List[Dict[str, any]] = []
    latest_stage: Dict[str, int] = {}
    stage_counter = 0
    result_counter = 1

    # used for removing unused columns when loading data
    join_attrs_per_table: Dict[str, set] = defaultdict(set)

    # used to track which tables contain which tables
    table_composition: Dict[str, set] = defaultdict(set)
    for stage in plan_stages:
        for table in stage["tables"]:
            table_composition[table].add(table)

    # used to track join branches so we can join them together at the end if needed
    join_branches = []

    for stage in plan_stages:
        attr = stage["on_attribute"]
        table0 = stage["tables"][0]
        table1 = stage["tables"][1]

        join_attrs_per_table[table0].add(attr)
        join_attrs_per_table[table1].add(attr)

        table0_stage = latest_stage.get(table0, None)
        table1_stage = latest_stage.get(table1, None)

        if table0_stage is not None and table1_stage is not None:
            join_stage = max(table0_stage, table1_stage)
            intermediate_name = execution_stages[join_stage]["name"]

            if (
                table0 in table_composition[intermediate_name]
                and table1 in table_composition[intermediate_name]
            ):
                # Both tables already joined in same join plan branch -- no need for a new join stage
                execution_stages[join_stage]["on_attributes"].add(attr)
                continue

        if table0_stage is not None:
            table0 = execution_stages[table0_stage]["name"]
        if table1_stage is not None:
            table1 = execution_stages[table1_stage]["name"]

        intermediate_name = f"result_{result_counter}"
        table_composition[intermediate_name].update(table_composition[table0])
        table_composition[intermediate_name].update(table_composition[table1])

        execution_stages.append(
            {
                "type": stage["type"],
                "name": intermediate_name,
                "tables": {table0, table1},
                "on_attributes": {stage["on_attribute"]},
                "contains": table_composition[intermediate_name],
            }
        )

        join_branches.append(intermediate_name)
        if table0 in join_branches:
            join_branches.remove(table0)
        if table1 in join_branches:
            join_branches.remove(table1)

        for table in table_composition[intermediate_name]:
            latest_stage[table] = stage_counter

        result_counter += 1
        stage_counter += 1

    while len(join_branches) > 1:
        table0 = join_branches[0]
        table1 = join_branches[1]

        intermediate_name = f"result_{result_counter}"
        table_composition[intermediate_name].update(table_composition[table0])
        table_composition[intermediate_name].update(table_composition[table1])

        execution_stages.append(
            {
                "type": "binary",
                "name": f"result_{stage_counter}",
                "tables": {table0, table1},
                "on_attributes": {},
                "contains": table_composition[intermediate_name],
            }
        )

        join_branches.pop(0)

    return execution_stages, join_attrs_per_table


def _create_nary_execution_plan(plan_stages):
    execution_stages: List[Dict[str, any]] = []
    attr_stage: Dict[str, int] = {}
    latest_stage: Dict[str, int] = {}
    stage_counter = 0
    result_counter = 1

    # used for removing unused columns when loading data
    join_attrs_per_table: Dict[str, set] = defaultdict(set)

    # used to track which tables contain which tables
    table_composition: Dict[str, set] = defaultdict(set)
    for stage in plan_stages:
        for table in stage["tables"]:
            table_composition[table].add(table)

    # used to track join branches so we can join them together at the end if needed
    join_branches = []

    for stage in plan_stages:
        attr = stage["on_attribute"]
        tables = stage["tables"]

        table_stages = {}
        intermediate_name = f"result_{result_counter}"
        for i, table in enumerate(tables):
            join_attrs_per_table[table].add(attr)
            table_stages[table] = latest_stage.get(table, None)
            if table_stages[table] is not None:
                tables[i] = execution_stages[table_stages[table]]["name"]
            table_composition[intermediate_name].update(table_composition[tables[i]])

        execution_stages.append(
            {
                "type": stage["type"],
                "name": intermediate_name,
                "tables": list(set(tables)),
                "on_attributes": {stage["on_attribute"]},
                "contains": table_composition[intermediate_name],
            }
        )

        join_branches.append(intermediate_name)

        for table in tables:
            if table in join_branches:
                join_branches.remove(table)

        attr_stage[attr] = stage_counter
        for table in table_composition[intermediate_name]:
            latest_stage[table] = stage_counter

        result_counter += 1
        stage_counter += 1

    if len(join_branches) > 1:
        intermediate_name = f"result_{result_counter}"
        for table in join_branches:
            table_composition[intermediate_name].update(table_composition[table])

        execution_stages.append(
            {
                "type": "nary",
                "name": f"result_{stage_counter}",
                "tables": join_branches,
                "on_attributes": {},
                "contains": table_composition[intermediate_name],
            }
        )

    return execution_stages, join_attrs_per_table


def _load_datafiles(table_paths, join_attrs_per_table) -> Dict[str, dd.DataFrame]:
    dfs: Dict[str, dd.DataFrame] = {}
    for name, path in table_paths.items():
        df = dd.read_parquet(path)

        # drop unused columns from each table
        drop_set = set(
            col for col in df.columns if col not in join_attrs_per_table[name]
        )
        dfs[name] = df.drop(columns=drop_set)

    return dfs


def _perform_stage_join(
    dfs: Dict[str, dd.DataFrame],
    tables: List[str],
    on_attributes: List[str],
) -> dd.DataFrame:
    left_table = tables.pop(0)
    left_df = dfs[left_table]

    for right_table in tables:
        right_df = dfs[right_table]

        if on_attributes:
            left_df = dd.merge(
                left_df,
                right_df,
                on=on_attributes,
                how="inner",
                suffixes=(None, "_right"),
            )
        else:
            # Makeshift cartesian product since dask doesn't support it
            left_df["key"] = 1
            right_df["key"] = 1
            left_df = dd.merge(
                left_df, right_df, on="key", how="inner", suffixes=(None, "_right")
            )
            left_df = left_df.drop(columns=["key"])

        duplicate_columns = [col for col in left_df.columns if "_right" in col]
        if duplicate_columns:
            left_df = left_df.drop(columns=duplicate_columns)

    return left_df


def _analyze_plan(plan_path: str, table_paths: Dict[str, str]) -> Dict[str, Any]:
    with open(plan_path) as f:
        plan_stages = json.load(f)

    analysis_results = {"stages": []}

    if plan_stages[0]["type"] == "binary":
        execution_plan, join_attrs_per_table = _create_binary_execution_plan(
            plan_stages
        )
    else:
        execution_plan, join_attrs_per_table = _create_nary_execution_plan(plan_stages)

    dfs = _load_datafiles(table_paths, join_attrs_per_table)
    analysis_results["base_relations"] = {name: len(df) for name, df in dfs.items()}

    total_intermediates = 0
    for i, stage in enumerate(execution_plan):
        stage_result = {"stage": i, "type": stage["type"]}
        try:
            tables = stage["tables"]
            on_attributes = stage["on_attributes"]

            logger.debug(f"\t\t\t\tAnalyzing stage {i}: Joining {tables} on {on_attributes}")

            # Perform the actual join
            result_df = _perform_stage_join(dfs, list(tables), list(on_attributes))

            dfs[stage["name"]] = result_df

            total_intermediates += len(result_df)
            selectivity = len(result_df) / prod((len(dfs[table]) for table in tables))

            stage_result["name"] = stage["name"]
            stage_result["output_size"] = len(result_df)
            stage_result["tables"] = list(tables)
            stage_result["on_attributes"] = list(on_attributes)
            stage_result["contains"] = list(stage["contains"])
            stage_result["total_intermediates"] = total_intermediates
            stage_result["selectivity"] = max(round(selectivity, 4), 0.0001)

            logger.debug(
                f"\t\t\t\t\tStage {i} completed: {len(result_df)} rows, tables: {sorted(tables)}"
            )

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

    if not os.path.exists(data_dir) or not os.path.exists(plans_dir):
        logger.debug("\t\tData or plans directory missing. Skipping analysis.")
        return

    table_paths = {
        os.path.basename(p): os.path.join(data_dir, p) for p in os.listdir(data_dir)
    }
    if not table_paths:
        logger.debug("\t\tNo data found to analyze. Skipping.")
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
