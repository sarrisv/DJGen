import logging
import os
from typing import Dict, Any

import dask.array as da
import dask.dataframe as dd
import numpy as np

from src.datagen import distributions

logger = logging.getLogger("djp")

DISTRIBUTION_FUNCTIONS = {
    "sequential": distributions.generate_sequential_attribute,
    "uniform": distributions.generate_uniform_attribute,
    "gaussian": distributions.generate_gaussian_attribute,
    "zipf": distributions.generate_zipf_attribute,
}


def _calculate_rows_per_partition(
    num_rows: int, total_bytes_per_row: int, target_mb: int = 128
) -> int:
    """Calculate rows per partition for target memory size"""

    # Avoid division by zero
    if total_bytes_per_row == 0:
        return num_rows

    # Convert target MB to bytes
    target_bytes = target_mb * 1024 * 1024

    # Calculate how many rows fit in target memory
    rows_per_partition = target_bytes // total_bytes_per_row

    # Ensure at least 1 row per partition, but not more than total rows
    return max(1, min(rows_per_partition, num_rows))


def _generate_attribute(
    attr_config: Dict[str, Any], num_rows: int, chunk_size: int
) -> dd.Series:
    """Generate single attribute from config"""

    dist_config = attr_config.get("distribution", {})
    dist_type = dist_config.get("type", "uniform")

    params = dist_config.copy()
    params.pop("type", None)

    # Handle defaults for distributions.
    if dist_type == "sequential":
        params["start"] = params.get("start", 0)
    elif dist_type == "uniform":
        params["low"] = attr_config.get("low", 0)
        params["high"] = attr_config.get("high", num_rows)
    elif dist_type == "gaussian":
        params["mean"] = params.get("mean", num_rows / 2)
        params["std"] = params.get("std", num_rows / 6)
        params["low"] = attr_config.get("low", params["mean"] - params["std"] * 6)
        params["high"] = attr_config.get("high", params["mean"] + params["std"] * 6)
    elif dist_type == "zipf":
        params["skew"] = params.get("skew", num_rows / 6)
        params["low"] = attr_config.get("low", 0)
        params["high"] = attr_config.get("high", num_rows)

    generator_func = DISTRIBUTION_FUNCTIONS[dist_type]

    attr_df = generator_func(
        num_rows=num_rows,
        chunk_size=chunk_size,
        dtype=np.dtype(attr_config.get("dtype", "int64")),
        **params,
    )
    attr_df.name = attr_config["name"]

    null_ratio = attr_config.get("null_ratio", 0)
    if 0 < null_ratio < 1:
        # Create a boolean mask to introduce nulls into the attribute
        mask = da.random.random(size=num_rows, chunks=chunk_size) < null_ratio
        attr_df = attr_df.mask(dd.from_dask_array(mask))

    return attr_df


def _generate_relation(rel_config: Dict[str, Any]) -> dd.DataFrame:
    """Generate relation from config"""

    num_rows = rel_config.get("num_rows", 1000)

    # Account for the auto-generated 'uid' attribute's size
    total_bytes_per_row = np.dtype("int64").itemsize
    for attr_conf in rel_config.get("attributes", []):
        dtype_str = attr_conf.get("dtype", "int64")
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

    all_attr_configs = [uid_config] + rel_config.get("attributes", [])

    attrs = [
        _generate_attribute(attr_conf, num_rows, chunk_size)
        for attr_conf in all_attr_configs
    ]

    rel_df = dd.concat(attrs, axis=1)

    return rel_df.set_index("uid")


def generate_data_for_iteration(
    datagen_config: Dict[str, Any], output_dir: str
) -> Dict[str, str]:
    """Generate all relations for iteration"""

    rel_paths = {}
    data_output_path = os.path.join(output_dir, "data")
    os.makedirs(data_output_path, exist_ok=True)

    for rel_config in datagen_config.get("relations", []):
        rel_name = rel_config["name"]
        logger.debug(f"\t\tGenerating {rel_name}...")

        rel_df = _generate_relation(rel_config)

        output_path = os.path.join(data_output_path, f"{rel_name}")
        rel_df.to_parquet(output_path, overwrite=True)

        rel_paths[rel_name] = output_path
        logger.debug(f"\t\t\t...written to {output_path}")

    return rel_paths
