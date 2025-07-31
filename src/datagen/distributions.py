import dask.array as da
import dask.dataframe as dd
from dask.dataframe import DataFrame
from numpy import dtype


def generate_sequential_column(
    num_rows: int,
    chunk_size: int,
    dtype: dtype,
    start: int,
) -> DataFrame:
    """Generates a Dask DataFrame with a single column of sequential integers"""

    dask_array = da.arange(start, start + num_rows, chunks=chunk_size, dtype=dtype)
    return dd.from_dask_array(dask_array)


def generate_uniform_column(
    num_rows: int,
    chunk_size: int,
    dtype: dtype,
    low: int,
    high: int,
) -> DataFrame:
    """Generates a Dask DataFrame with a single column of uniformly distributed random integers"""

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
    low: int,
    high: int,
) -> DataFrame:
    """Generates a Dask DataFrame with a single column of normally distributed random numbers"""

    dask_array = da.random.normal(mean, std, size=num_rows, chunks=chunk_size).clip(
        low, high
    )
    return dd.from_dask_array(dask_array).astype(dtype)


def generate_zipf_column(
    num_rows: int,
    chunk_size: int,
    dtype: dtype,
    low: int,
    high: int,
    skew: float,
) -> DataFrame:
    """Generates a Dask DataFrame with a single column of Zipf-distributed random numbers"""

    dask_array = da.random.zipf(a=skew, size=num_rows, chunks=chunk_size).clip(
        low, high
    )
    return dd.from_dask_array(dask_array).astype(dtype)
