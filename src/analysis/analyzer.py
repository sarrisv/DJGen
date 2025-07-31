import os
import json
from typing import Dict, Any, List, Tuple, Set
import dask.dataframe as dd


def _consolidate_stage_inputs(
    input_logical_names: List[str], available_dfs: Dict[str, dd.DataFrame]
) -> List[Dict[str, Any]]:
    """Groups logical table names by their underlying physical DataFrame"""

    stage_input_map = {}
    for name in input_logical_names:
        df = available_dfs[name]

        # Use the DataFrame's memory ID to group tables that are the same physical object
        df_id = id(df)
        if df_id not in stage_input_map:
            stage_input_map[df_id] = {"df": df, "tables": set()}
        stage_input_map[df_id]["tables"].add(name)
    return list(stage_input_map.values())


def _find_physical_key(
    df: dd.DataFrame, logical_key: str, contained_tables: List[str]
) -> str:
    """Finds the physical column name for a logical join key within a DataFrame"""

    # Case 1: The key is from the most recent join and has its logical name
    if logical_key in df.columns:
        return logical_key

    # Case 2: The key is from a prior join and has been renamed with a prefix
    for table_name in contained_tables:
        physical_key = f"{table_name}_{logical_key}"
        if physical_key in df.columns:
            return physical_key

    raise KeyError(
        f"\t\t\t\tCould not find logical key '{logical_key}' in the given DataFrame.\n"
        f"\t\t\t\tAvailable columns: {df.columns.tolist()}"
    )


def _perform_stage_join(
    df_metas: List[Dict[str, Any]], logical_key: str
) -> Tuple[dd.DataFrame, Set[str]]:
    """
    Sequentially joins a list of DataFrames based on a logical key

    Handles physical key resolution, column renaming to avoid conflicts,
    and key name reconciliation

    Returns the final merged DataFrame and the set of all tables it contains
    """
    if len(df_metas) < 2:
        meta = df_metas[0]
        return meta["df"], meta["tables"]

    # Start with the first DataFrame as the left side of the join
    left_meta = df_metas.pop(0)
    left_df = left_meta["df"]

    for right_meta in df_metas:
        right_df = right_meta["df"]

        # Find the correct physical key name in BOTH DataFrames
        left_on_key = _find_physical_key(
            left_df, logical_key, list(left_meta["tables"])
        )
        right_on_key = _find_physical_key(
            right_df, logical_key, list(right_meta["tables"])
        )

        # Proactively rename non-key columns of the right-hand side
        # to prevent collisions, but ONLY if it's a base table
        if right_on_key == logical_key:
            # This indicates the right dataframe is a base table not yet joined
            table_name = list(right_meta["tables"])[0]
            rename_map = {
                c: f"{table_name}_{c}" for c in right_df.columns if c != logical_key
            }
            right_df = right_df.rename(columns=rename_map)

        # Perform the merge
        left_df = dd.merge(
            left_df,
            right_df,
            left_on=left_on_key,
            right_on=right_on_key,
            how="inner",
        )

        # Reconcile the key name for the next stage
        if left_on_key != logical_key:
            left_df = left_df.rename(columns={left_on_key: logical_key})

        # Update the metadata for the evolving left-hand side
        left_meta["tables"].update(right_meta["tables"])

    # Persist the final result of the stage to avoid recomputation
    result_df = left_df.persist()
    all_consumed_tables = left_meta["tables"]

    return result_df, all_consumed_tables


def _analyze_plan(plan_path: str, table_paths: Dict[str, str]) -> Dict[str, Any]:
    with open(plan_path) as f:
        plan_stages = json.load(f)

    analysis_results = {"stages": []}
    base_dfs = {name: dd.read_parquet(path) for name, path in table_paths.items()}
    available_dfs = base_dfs.copy()

    analysis_results["base_relations"] = {
        name: len(df) for name, df in base_dfs.items()
    }

    for i, stage in enumerate(plan_stages):
        stage_result = {"stage": i, "type": stage["type"]}

        try:
            logical_key = stage["on_attribute"]
            input_logical_names = stage["tables"]
            print(
                f"\t\t\t\tAnalyzing stage {i}: Joining {input_logical_names} on '{logical_key}'"
            )

            # 1. Consolidate metadata for all unique physical DFs in this stage
            df_metas = _consolidate_stage_inputs(input_logical_names, available_dfs)

            # 2. Perform the join logic for the stage
            result_df, all_consumed_tables = _perform_stage_join(df_metas, logical_key)

            # 3. Update the global state map for the next stage
            for name in all_consumed_tables:
                available_dfs[name] = result_df

            # 4. Record results for this stage
            stage_result["output_size"] = len(result_df)
            stage_result["cumulative_tables_joined"] = sorted(list(all_consumed_tables))

        except Exception as e:
            error_message = f"{type(e).__name__} - {e}"
            print(f"            ERROR analyzing stage {i}: {error_message}")
            stage_result["error"] = error_message

        analysis_results["stages"].append(stage_result)

    return analysis_results


def generate_analysis_for_iteration(output_dir: str) -> None:
    print(f"\t\tStarting analysis for {output_dir}...")
    data_dir = os.path.join(output_dir, "data")
    plans_dir = os.path.join(output_dir, "plans")
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    if not os.path.exists(data_dir) or not os.path.exists(plans_dir):
        print("\t\tData or plans directory missing. Skipping analysis.")
        return

    table_paths = {
        os.path.basename(p): os.path.join(data_dir, p) for p in os.listdir(data_dir)
    }

    if not table_paths:
        print("\t\tNo data found to analyze. Skipping.")
        return

    for plan_file in os.listdir(plans_dir):
        if not plan_file.endswith(".json"):
            continue

        plan_name = plan_file.replace(".json", "")
        print(f"\t\t\tAnalyzing plan: {plan_name}")
        plan_path = os.path.join(plans_dir, plan_file)

        plan_analysis = _analyze_plan(plan_path, table_paths)

        output_path = os.path.join(analysis_dir, f"{plan_name}_analysis.json")
        with open(output_path, "w") as f:
            json.dump(plan_analysis, f, indent=4)
        print(f"\t\t\t...written to {output_path}")
