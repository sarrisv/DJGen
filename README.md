# Data & Join Plan Generator (djp-generator)

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
  - `cyclic`: A chain of joins with cyclic attribute data dependencies
- **Join Plan Analysis**: Generate an analysis of the generated including details like exact cardinality. Cardinality is computed via the exact join using Dask for more parallel computation.
- **Join Plan Visualizations**: Generate visual representations of join plans and their execution graphs in multiple formats (PNG, SVG, etc.) using GraphViz.
- **Configuration-Driven**: Configured via a single TOML configuration file. This allows for repeatable and shareable experimental setups.
- **Scalable**: Built on Dask and Parquet, it can handle datasets that are larger than memory, distributing the workload efficiently on a local machine.
  - Note: Since the analysis is exact, it will not be as scalable. **Disable the analysis if you are finding the execution to be slow.**

## How It Works

The tool operates in three distinct, sequential phases for each "iteration" defined in your configuration file:

1. **Data Generation (Datagen)**: It reads the `datagen` section of your configuration and creates the specified tables. Each table is generated as a Dask DataFrame and saved as a Parquet file in the output directory. This phase handles creating columns with the requested data types and distributions.
2. **Plan Generation (Plangen)**: Using the table schemas from the previous step, it generates join plans according to the patterns (`star`, `linear`, etc.) you've specified. For each pattern, it creates both a `binary` (table-at-a-time) and `n-ary` (attribute-at-a-time) join plan, saving them as `.json` files.
3. **Analysis**: The tool reads the generated data (Parquet files) and a join plan (JSON file). It then uses Dask to execute the joins one stage at a time, calculating the number of rows in the resulting DataFrame after each merge. The results, including the output size of each stage, are saved to an analysis JSON file.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sarrisv/djp-generator.git
   cd djp-generator
   ```

2. **Create virtual environment and install dependencies:**

   Using uv (recommended):

   ```bash
   uv sync
   ```

   Or using manually using venv and pip:

   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
   ```

## Usage

### 1. Create a Configuration File

The behavior of the generator is controlled by a TOML file. You can define one or more "iterations," each with its own data, plans, and analysis steps.

Here is an example `config.toml` that defines one iteration. This iteration generates two tables, `table0` and `table1`, and then creates, analyzes, and visualizes `star` and `linear` join plans for them.

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
visualize = true
visualization_format = "png"
base_plans = [
    { pattern = "star", num_plans = 1, permutations = false },
    { pattern = "linear", num_plans = 1, permutations = true }
]

[iterations.analysis]
enabled = true
```

#### Plan Generation Configuration

The `plangen` section supports the following options:

- `enabled`: Whether to generate join plans (if the `plangen` section is included, defaults to true)
- `visualize`: Whether to generate visualizations of the join plans
- `visualization_format`: Format for GraphViz visualizations (`png`, `svg`, etc.)
- `base_plans`: Array of plan configurations, each containing:
  - `pattern`: Type of join pattern (`star`, `linear`, `cyclic`, `random`)
  - `num_plans`: Number of instances of this pattern to generate
  - `permutations`: Controls permutation generation:
    - `false`: No permutations (generate base plan only)
    - `true`: Generate all possible permutations
    - `N` (integer): Generate up to N permutations

More examples can be found in the `config` directory.

## Project Structure

```
src/
├── main.py              # Main entry point and CLI interface
├── datagen/             # Synthetic data generation
│   ├── generator.py     # Core data generation logic
│   └── distributions.py # Statistical distribution implementations
├── plangen/             # Join plan generation
│   ├── generator.py     # Plan generation orchestration
│   └── patterns.py      # Join pattern implementations (star, linear, etc.)
├── analysis/            # Join execution and analysis
│   └── analyzer.py      # Dask-based join execution and cardinality analysis
├── visualization/       # Plan visualization
│   └── generator.py     # Graph visualization using matplotlib/graphviz
└── utils/               # Shared utilities
    └── toml_parser.py   # Configuration file parsing
```

### 2. Run the Generator

View available options:

```bash
python -m src.main --help
```

Execute the tool from your terminal, pointing it to your configuration file:

**Standard mode** (recommended - uses Dask cluster for parallel execution):

```bash
python -m src.main run config/config_full.toml
```

**Debug mode** (sequential execution, better for development/debugging):

```bash
python -m src.main debug config/config_full.toml
```

**GUI mode** (standard mode with visual interface)

```bash
python -m streamlit run src/ui.py
```

The tool will display a Dask dashboard URL in run mode for monitoring execution progress.

### 3. Inspect the Output

The tool will create an output directory structure as specified in your configuration. For the example above, the structure would be:

```
output/
└── example/
    ├── data/
    │   ├── table0/ (Parquet files)
    │   └── table1/ (Parquet files)
    ├── plans/
    │   ├── star0_base.json
    │   ├── star0_binary.json
    │   ├── star0_nary.json
    │   ├── linear0_base.json
    │   ├── linear0_p0_binary.json (permutation 0)
    │   ├── linear0_p0_nary.json
    │   ├── linear0_p1_binary.json (permutation 1)
    │   └── linear0_p1_nary.json
    ├── analysis/
    │   ├── star0_binary_analysis.json
    │   ├── star0_nary_analysis.json
    │   ├── linear0_p0_binary_analysis.json
    │   ├── linear0_p0_nary_analysis.json
    │   ├── linear0_p1_binary_analysis.json
    │   └── linear0_p1_nary_analysis.json
    └── visualizations/
        ├── star0_binary.png
        ├── star0_nary.png
        ├── linear0_p0_binary.png
        ├── linear0_p0_nary.png
        ├── linear0_p1_binary.png
        └── linear0_p1_nary.png
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
