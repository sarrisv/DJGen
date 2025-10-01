import logging
import os
from typing import Dict, Any

import dask.array as da
import dask.dataframe as dd
import numpy as np

from src.datagen import distributions

logger = logging.getLogger("djp")

DISTRIBUTION_FUNCTIONS = {
    "sequential": distributions.generate_sequential_column,
    "uniform": distributions.generate_uniform_column,
    "gaussian": distributions.generate_gaussian_column,
    "zipf": distributions.generate_zipf_column,
}

def _calculate_rows_per_partition(
    num_rows: int, total_bytes_per_row: int, target_mb: int = 128
) -> int:
    """Calculates the number of rows per Dask partition for a target memory size"""

    if total_bytes_per_row == 0:
        return num_rows
    target_bytes = target_mb * 1024 * 1024
    rows_per_partition = target_bytes // total_bytes_per_row
    return max(1, min(rows_per_partition, num_rows))


def _generate_column(column_config: Dict[str, Any], num_rows: int, chunk_size: int) -> dd.Series:
    """Generates a single Dask DataFrame column based on the provided configuration"""

    dist_config = column_config.get("distribution", {})
    dist_type = dist_config.get("type", "uniform")

    params = dist_config.copy()
    params.pop("type", None)

    # Handle defaults for distributions.
    if dist_type == "sequential":
        params["start"] = params.get("start", 0)
    elif dist_type == "uniform":
        params["low"] = column_config.get("low", 0)
        params["high"] = column_config.get("high", num_rows)
    elif dist_type == "gaussian":
        params["mean"] = params.get("mean", num_rows / 2)
        params["std"] = params.get("std", num_rows / 6)
        params["low"] = column_config.get("low", params["mean"] - params["std"] * 6)
        params["high"] = column_config.get("high", params["mean"] + params["std"] * 6)
    elif dist_type == "zipf":
        params["skew"] = params.get("skew", num_rows / 6)
        params["low"] = column_config.get("low", 0)
        params["high"] = column_config.get("high", num_rows)

    generator_func = DISTRIBUTION_FUNCTIONS[dist_type]

    col_df = generator_func(
        num_rows=num_rows,
        chunk_size=chunk_size,
        dtype=np.dtype(column_config.get("dtype", "int64")),
        **params,
    )
    col_df.name = column_config["name"]

    null_ratio = column_config.get("null_ratio", 0)
    if 0 < null_ratio < 1:
        # Create a boolean mask to introduce nulls into the column
        mask = da.random.random(size=num_rows, chunks=chunk_size) < null_ratio
        col_df = col_df.mask(dd.from_dask_array(mask))

    return col_df


def _generate_table(table_config: Dict[str, Any]) -> dd.DataFrame:
    """Generates a Dask DataFrame for a table based on its configuration"""

    num_rows = table_config.get("num_rows", 1000)

    # Account for the auto-generated 'uid' column's size
    total_bytes_per_row = np.dtype("int64").itemsize
    for col_conf in table_config.get("columns", []):
        dtype_str = col_conf.get("dtype", "int64")
        total_bytes_per_row += np.dtype(dtype_str).itemsize
    chunk_size = _calculate_rows_per_partition(num_rows, total_bytes_per_row)

    uid_config = {
        "name": "uid",
        "dtype": "int64",
        "distribution": {
            "type": "sequential",
            "start": 0,
        },
    }

    all_column_configs = [uid_config] + table_config.get("columns", [])

    columns = [
        _generate_column(col_conf, num_rows, chunk_size)
        for col_conf in all_column_configs
    ]

    table_df = dd.concat(columns, axis=1)

    return table_df.set_index("uid")


def generate_data_for_iteration(data_gen_config: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """Orchestrates the data generation for all tables within a single iteration"""

    table_paths = {}
    data_output_path = os.path.join(output_dir, "data")
    os.makedirs(data_output_path, exist_ok=True)

    for table_config in data_gen_config.get("tables", []):
        table_name = table_config["name"]
        logger.debug(f"\t\tGenerating {table_name}...")

        table_df = _generate_table(table_config)

        output_path = os.path.join(data_output_path, f"{table_name}")
        table_df.to_parquet(output_path, overwrite=True)

        table_paths[table_name] = output_path
        logger.debug(f"\t\t\t...written to {output_path}")

    return table_paths
