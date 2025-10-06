import dask.array as da
import dask.dataframe as dd
from dask.dataframe import DataFrame
from numpy import dtype
from typing import Union


def generate_sequential_column(
    num_rows: int,
    chunk_size: int,
    dtype: dtype,
    start: int,
) -> DataFrame:
    """Generate sequential integers column"""

    dask_array = da.arange(start, start + num_rows, chunks=chunk_size, dtype=dtype)
    return dd.from_dask_array(dask_array)


def generate_uniform_column(
    num_rows: int,
    chunk_size: int,
    dtype: dtype,
    low: Union[int, float],
    high: Union[int, float],
) -> DataFrame:
    """Generate uniform random integers column"""

    dask_array = da.random.randint(
        low, high, size=num_rows, chunks=chunk_size, dtype=dtype
    )
    return dd.from_dask_array(dask_array)


def generate_gaussian_column(
    num_rows: int,
    chunk_size: int,
    dtype: dtype,
    mean: float,
    std: float,
    low: Union[int, float],
    high: Union[int, float],
) -> DataFrame:
    """Generate gaussian random numbers column"""

    dask_array = da.random.normal(mean, std, size=num_rows, chunks=chunk_size).clip(
        low, high
    )
    return dd.from_dask_array(dask_array).astype(dtype)


def generate_zipf_column(
    num_rows: int,
    chunk_size: int,
    dtype: dtype,
    low: Union[int, float],
    high: Union[int, float],
    skew: float,
) -> DataFrame:
    """Generate zipf-distributed random numbers column"""

    dask_array = da.random.zipf(a=skew, size=num_rows, chunks=chunk_size).clip(
        low, high
    )
    return dd.from_dask_array(dask_array).astype(dtype)
