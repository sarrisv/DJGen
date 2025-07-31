# Data & Join Plan Generator (djp-generator)

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dask](https://img.shields.io/badge/powered%20by-Dask-orange.svg)](https://dask.org/)

A command-line tool for generating configurable synthetic datasets and configurable fully conjunctive natural join plans. 

## Core Features

- **Data Generation**: Generate synthetic tables with specified row counts and column configurations. Data for columns can be generated using various distributions:
  - `sequential`: For creating unique IDs
  - `uniform`: For evenly distributed data
  - `gaussian`: For normally distributed data
  - `zipf`: For skewed, real-world-like data distributions
- **Join Plan Generation**: Generate different types of join query plans based on your generated table schemas. Supported patterns include:
  - `random`: A random, connected graph of joins
  - `star`: A central table joined with multiple satellite tables
  - `linear`: A linear chain of joins (A-B, B-C, C-D)
  - `cyclic`: A linear plan with an additional join between the first and last tables
- **Join Execution & Analysis**: Using Dask, it performs the joins on the generated data to learn the exact cardinality of each join stage. 
- **Configuration-Driven**: Configured via a single TOML configuration file. This allows for repeatable and shareable experimental setups.
- **Scalable**: Built on Dask and Parquet, it can handle datasets that are larger than memory, distributing the workload efficiently on a local machine. 
    - Note: Since the analysis is exact, it will not be as scalable. **Disable the analysis if you are finding the execution to be slow.**

## How It Works

The tool operates in three distinct, sequential phases for each "iteration" defined in your configuration file:

1. **Data Generation (Datagen)**: It reads the `datagen` section of your configuration and creates the specified tables. Each table is generated as a Dask DataFrame and saved as a Parquet file in the output directory. This phase handles creating columns with the requested data types and distributions.
2. **Plan Generation (Plangen)**: Using the table schemas from the previous step, it generates join plans according to the patterns (`star`, `linear`, etc.) you've specified. For each pattern, it creates both a `binary` (step-by-step) and `n-ary` (grouped by key) join plan, saving them as `.json` files.
3. **Analysis**: This is the execution phase. The tool reads the generated data (Parquet files) and a join plan (JSON file). It then uses Dask to execute the joins one stage at a time, calculating the number of rows in the resulting DataFrame after each merge. The results, including the output size of each stage, are saved to an analysis JSON file.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sarrisv/djp-generator.git
    cd djp-generator
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the package and its dependencies:**
    ```bash
    pip install requirements.txt 
    ```

## Usage

### 1. Create a Configuration File

The behavior of the generator is controlled by a TOML file. You can define one or more "iterations," each with its own data, plans, and analysis steps.

Here is an example `config.toml` that defines one iteration. This iteration generates two tables, `table0` and `table1`, and then creates and analyzes `star` and `linear` join plans for them.

```toml
[project]
name = "Example Config"
output_dir = "output"

[[iterations]]
name = "example"

[iterations.datagen]
enabled = true
tables = [
    { name = "table0", num_rows = 100000, columns = [ { name = "attr0", dtype = "int64", distribution = { type = "sequential", start = 1 } }, { name = "attr1", dtype = "int32", distribution = { type = "gaussian", mean = 40, std = 10 } }, { name = "attr2", dtype = "int32", distribution = { type = "zipf", skew = 1.2 }, low = 10000, high = 99999 } ] },
    { name = "table1", num_rows = 500000, columns = [ { name = "attr0", dtype = "int64", distribution = { type = "sequential", start = 1 } }, { name = "attr1", dtype = "int64", distribution = { type = "uniform" }, low = 1, high = 100000 }, { name = "attr2", dtype = "float64", distribution = { type = "uniform" }, low = 5.0, high = 500.0 } ] }
]

[iterations.plangen]
enabled = true
base_plans = [
    { name = "star_join", pattern = "star" },
    { name = "linear_join", pattern = "linear" }
]

[iterations.analysis]
enabled = true
```

More examples can be found in the `config` directory.

### 2. Run the Generator

Execute the tool from your terminal, pointing it to your configuration file. The `run` mode sets up a Dask cluster for parallel execution.

```bash
python -m src.main run config/config_full.toml
```

You can also use `debug` mode, which runs the process sequentially without a Dask cluster, which is useful for debugging with tools like `pdb` or in an IDE.

```bash
python -m src.main debug config/config_full.toml
```

### 3. Inspect the Output

The tool will create an output directory structure as specified in your configuration. For the example above, the structure would be:

```
output/
└── example/
    ├── data/
    │   ├── customers/ (Parquet files)
    │   └── orders/    (Parquet files)
    ├── plans/
    │   ├── star_join0_binary.json
    │   ├── star_join0_nary.json
    │   ├── linear_join0_binary.json
    │   └── linear_join0_nary.json
    └── analysis/
        ├── star_join0_binary_analysis.json
        ├── star_join0_nary_analysis.json
        ├── linear_join0_binary_analysis.json
        └── linear_join0_nary_analysis.json
```

The `*_analysis.json` files contain the results of the join execution, including the output size at each stage.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.