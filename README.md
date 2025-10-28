# Data & Join Plan Generator (DJGen)

A command-line tool and web interface for generating configurable synthetic datasets and fully conjunctive join plans. Data generation and join plan analysis are parallelized via Dask for scalability.

## Core Features

- **Data Generation**: Generate synthetic tables (relations) with specified row counts and attribute configurations. Data for columns can be generated using various distributions: - `sequential`: For creating unique IDs
  - `sequential`: For creating unique IDs
  - `uniform`: For evenly distributed data
  - `gaussian`: For normally distributed data
  - `zipf`: For skewed, real-world-like data distributions
- **Join Plan Generation**: Generate binary and n-ary join queries for your generated data set using various join patterns. Supported patterns include:
  - `random`: A random, connected graph of joins
  - `star`: A central table joined with multiple satellite tables
  - `linear`: A linear chain of joins (A-B, B-C, C-D)
  - `cyclic`: A chain of joins with cyclic attribute data dependencies
  - `custom`: A user-defined sequence of joins
- SQL Query Generation: For each generated join plan, an equivalent standard SQL query is also created, allowing for easy portability and testing on different database systems.
- **Join Plan Analysis**: Execute generated plans using Dask, appending detailed stage-by-stage performance metrics (cardinality, selectivity, intermediate result sizes) directly into the plan files.
- **Join Plan Visualizations**: Generate visual representations of join plans and their execution graphs in multiple formats (PNG, SVG, etc.) using GraphViz.
- **Configuration-Driven**: All behavior is controlled by a single TOML file, allowing for repeatable and shareable experimental setups.
- **Scalable**: Built on Dask and Parquet, it can handle datasets that are larger than memory, distributing the workload efficiently on a local machine.
  - Note: Since the analysis is exact, and thus requires computing the actual joins, it will not be as scalable.
- **Web Interface**: Interactive Streamlit-based GUI for configuration, execution, results visualization, plan comparison, and data download.

## How It Works

The tool operates in three distinct, sequential phases for each "iteration" defined in your configuration file:

1. **Data Generation (Datagen)**: Reads the datagen section of your configuration and creates synthetic tables (relations) with specified schemas and distributions. Each table is generated as a Dask DataFrame and saved as Parquet files in the output directory.
2. **Plan Generation (Plangen)**: Using the table schemas from the previous step, generates join plans according to the specified patterns (`star`, `linear`, `cyclic`, `random`, `custom`). For each pattern, it creates both execution strategies:

   - **Binary plans**: A traditional, table-at-a-time-style execution with one join per stage
   - **N-ary plans**: A Worst-Case Optimal Join, attribute-at-a-time-style execution where multiple relations can be joined in a single stage on a common attribute

3. **Analysis**: Executes each plan file using Dask. It computes the exact cardinality and selectivity at every stage of the join process. The performance metrics are then written back into the original JSON plan file under an `analysis` key, allowing for direct comparison of the different strategies.

## Prerequisites

- Python 3.11 or higher
- GraphViz (system dependency for visualizations)
  - Linux:
    - Arch: `pacman -S graphviz`
    - Ubuntu/Debian: `sudo apt-get install graphviz`
  - macOS: `brew install graphviz`
  - Windows: Download from [https://graphviz.org/download/](https://graphviz.org/download/)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sarrisv/DJGen.git
   cd DJGen
   ```

2. **Create virtual environment and install dependencies:**

   Automatically install dependencies locally using uv (recommended):

   ```bash
   uv sync
   ```

   Or manually install dependencies locally using venv and pip:

   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\Activate.ps1`
    pip install -r requirements.txt
   ```

## Usage

### 1. Create a Configuration File

The generator's behavior is controlled by a TOML file. The example below defines one iteration that generates two relations, then creates, analyzes, and visualizes star and linear join plans for them.

```toml
[project]
name = "Example Config"
output_dir = "output"

[[iterations]]
name = "example"

[iterations.datagen]
enabled = true
relations = [
    { name = "rel0", num_rows = 100000, attributes = [ { name = "attr0", dtype = "int64", null_ratio = 0.0, distribution = { type = "sequential", start = 1 } }, { name = "attr1", dtype = "int32", distribution = { type = "gaussian", mean = 40, std = 10 } }, { name = "attr2", dtype = "int32", distribution = { type = "zipf", skew = 1.2, low = 10000, high = 99999 } } ] },
    { name = "rel1", num_rows = 500000, attributes = [ { name = "attr0", dtype = "int64", null_ratio = 0.5, distribution = { type = "sequential", start = 1 } }, { name = "attr1", dtype = "int64", distribution = { type = "uniform", low = 1, high = 100000 } }, { name = "attr2", dtype = "float64", distribution = { type = "uniform", low = 5.0, high = 500.0 } } ] }
]

[iterations.plangen]
enabled = true
visualize = true
visualization_format = "png"
base_plans = [
    { pattern = "star", num_plans = 1, permutations = false },
    { pattern = "linear", num_plans = 1, permutations = true },
    { pattern = "custom", base_plan = [
        ["rel0", "rel1", "attr0"],
        ["rel1", "rel2", "attr1"],
        ["rel0", "rel2", "attr2"],
    ]},
]

[iterations.analysis]
enabled = true
```

#### Iteration Configuration

The `[[iterations]]` section defines a workload to be generated and/or analyzed.

- `name` (string): A unique name for this iteration
- `seed` (integer, optional): A seed value to make data and plan generation reproducible. If omitted, the output will be different on each run.

#### Data Generation Configuration

Data Generation Configuration

The `datagen` section controls the creation of synthetic data.

- `enabled` (boolean): Set to true to run this phase. If the datagen section is present, this defaults to true.
- `relations` (array): A list of tables to generate. Each table is an object with the following keys:
  - `name` (string): The name of the relation (e.g., "rel0").
  - `num_rows` (integer): The number of rows to generate.
  - `attributes` (array): A list of attribute (column) configurations:
    - `name` (string): The name of the attribute (e.g., "attr1").
    - `dtype` (string, optional): Data type. Supported types: `int32`, `int64`, `float32`, `float64`. Defaults to `int64`.
    - `null_ratio` (float, optional): Proportion of null values (from 0.0 to 1.0). Defaults to 0.
    - `distribution` (object): A dictionary specifying the data distribution with the following keys:
      - `type` (string): Can be `sequential`, `uniform`, `gaussian`, or `zipf`.
      - Distribution-specific parameters:
        - For `sequential`: `start` (integer) - The starting value.
        - For `uniform`: `low` and `high` (integer or float) - The range for random values.
        - For `gaussian`: `mean` and `std` (float) - The mean and standard deviation.
        - For `zipf`: `skew` (float), `low` and `high` (integer or float) - The skewness parameter and value range.

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
  - `base_plan` (array): Required only for the `custom` pattern. A list of joins, where each join is an array of `["relation1", "relation2", "attribute"]`.

Complete configuration examples can be found in the `config/` directory, including `config_full.toml`, `quick.toml`, and specialized examples for different use cases.

### 2. Run the Generator

The tool can be run from the command line or via the web interface. If using uv, add `uv run` before any command.

View available options:

```bash
python -m src.main --help
```

Execute the tool from your terminal, pointing it to your configuration file:

**Standard mode** (recommended - uses Dask cluster for parallel execution):

```bash
python -m src.main run <config>
```

**Debug mode** (sequential execution, better for development/debugging):

```bash
python -m src.main debug <config>
```

**GUI mode** (standard mode with Streamlit interface)

```bash
python -m streamlit run src/ui/main.py
```

The tool will display a Dask dashboard URL in run mode for monitoring execution progress.

### 3. Inspect the Output

The tool will create an output directory structure as specified in your configuration. For the example above, the structure would be:

```
output/
└── example/
    ├── data/
    │   ├── rel0/ (Parquet files)
    │   └── rel1/ (Parquet files)
    ├── plans/
    │   ├── linear0_p0_binary.json  # Plan with analysis results inside
    │   ├── linear0_p0_nary.json
    │   ├── linear0_p1_binary.json
    │   ├── linear0_p1_nary.json
    │   ├── star0_p0_binary.json
    │   └── star0_p0_nary.json
    └── visualizations/
        ├── linear0_p0_binary.png
        ├── linear0_p0_nary.png
        ├── linear0_p1_binary.png
        ├── linear0_p1_nary.png
        ├── star0_p0_binary.png
        └── star0_p0_nary.png
```

### 4. Understanding the Output File Structure

Each JSON file is a self-contained description of a query, its execution plan, and (optionally) its performance analysis.

The structure of the JSON output is as follows:

- `plan_id` (string): A unique name for the plan, like `random0_p0_binary`.
- `catalog` (object): Metadata about all relations involved in the query. For each relation, it includes:
  - `schema`: A list of attributes with their names and data types.
  - `statistics`: Key metrics like cardinality (the number of rows).
  - `location`: The path to the generated Parquet data file.
- `query` (object): The high-level definition of the query.
  - `base_plan`: An abstract list of joins, represented as `[relation1, relation2, attribute]`, defining the logical connections without enforcing an execution order.
  - `sql`: The complete, runnable SQL query that is equivalent to the base plan.
- `execution_plan` (object): The physical plan that defines the exact sequence of operations.
  - `type`: The strategy, either binary (one join at a time) or nary (multi-way joins on a single attribute).
  - `stages`: An ordered array of join operations. Each stage object contains:
    - `stage_id`: The sequence number of the stage (0, 1, 2...).
    - `name`: The name of the intermediate result (e.g., result_1).
    - `input_relations`: The relations or intermediate results being joined in this stage.
    - `on_attributes`: The specific keys being used for the join.
- `analysis` (object): **This section only appears if the analysis phase is run.** It contains performance metrics for each execution stage.
  - `stages`: An array of analysis results that corresponds directly to the `execution_plan` stages. Each stage object includes:
    - `output_size`: The number of rows produced by this stage's join.
    - `total_intermediates`: The cumulative sum of output sizes from all stages up to and including the current one.
    - `selectivity`: The ratio of `output_size` to the product of the input sizes, indicating how much the join filtered the data.

## Project Structure

```
src/
├── main.py                # CLI entry point
├── analysis/
│   └── analyzer.py        # Dask-based join execution and cardinality analysis
├── datagen/
│   ├── generator.py       # Core data generation logic
│   └── distributions.py   # Statistical distribution implementations
├── plangen/
│   ├── generator.py       # Plan generation orchestration
│   └── patterns.py        # Join pattern implementations (star, linear, etc.)
├── ui/
│   ├── main.py            # Streamlit application entry point
│   ├── components.py      # UI components and widgets
│   ├── models.py          # UI state management and data models
│   ├── analysis_tab.py    # Renders the analysis comparison view
│   ├── data_tab.py        # Renders the data schema and query view
│   └── utils.py           # UI-specific helper functions
├── utils/
│   └── toml_parser.py     # Configuration file parsing
└── visualization/
    └── generator.py       # Graph visualization using Graphviz
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
