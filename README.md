# Data & Join Plan Generator (djp-generator)

A command-line tool and web interface for generating configurable synthetic datasets and fully conjunctive join plans. Data generation and join plan analysis are parallelized via Dask for scalability.

## Core Features

- **Data Generation**: Generate synthetic tables (relations) with specified row counts and attribute configurations. Data for columns can be generated using various distributions: - `sequential`: For creating unique IDs
  - `uniform`: For evenly distributed data
  - `gaussian`: For normally distributed data
  - `zipf`: For skewed, real-world-like data distributions
- **Join Plan Generation**: Generate binary and n-ary join queries for your generated data set using various join patterns. Supported patterns include:
  - `random`: A random, connected graph of joins
  - `star`: A central table joined with multiple satellite tables
  - `linear`: A linear chain of joins (A-B, B-C, C-D)
  - `cyclic`: A chain of joins with cyclic attribute data dependencies
- **Join Plan Analysis**: Execute generated plans using Dask, appending detailed stage-by-stage performance metrics (cardinality, selectivity, intermediate result sizes) directly into the plan files.
- **Join Plan Visualizations**: Generate visual representations of join plans and their execution graphs in multiple formats (PNG, SVG, etc.) using GraphViz.
- **Configuration-Driven**: All behavior is controlled by a single TOML file, allowing for repeatable and shareable experimental setups.
- **Scalable**: Built on Dask and Parquet, it can handle datasets that are larger than memory, distributing the workload efficiently on a local machine.
  - Note: Since the analysis is exact, and thus requires computing the actual joins, it will not be as scalable.
- **Web Interface**: Interactive Streamlit-based GUI for configuration, execution, results visualization, plan comparison, and data download.

## How It Works

The tool operates in three distinct, sequential phases for each "iteration" defined in your configuration file:

1. **Data Generation (Datagen)**: Reads the datagen section of your configuration and creates synthetic tables (relations) with specified schemas and distributions. Each table is generated as a Dask DataFrame and saved as Parquet files in the output directory.
2. **Plan Generation (Plangen)**: \*\*: Using the table schemas from the previous step, generates join plans according to the specified patterns (`star`, `linear`, `cyclic`, `random`). For each pattern, it creates both execution strategies:

   - **Binary plans**: A traditional, table-at-a-time-style execution with one join per stage
   - **N-ary plans**: A Worst-Case Optimal Join, attribute-at-a-time-style execution where multiple relations can be joined in a single stage on a common attribute

3. **Analysis**: Executes each plan file using Dask. It computes the exact cardinality and selectivity at every stage of the join process. The performance metrics are then written back into the original JSON plan file under an analysis key, allowing for direct comparison of the different strategies.

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
   git clone https://github.com/sarrisv/djp-generator.git
   cd djp-generator
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
    { name = "rel0", num_rows = 100000, attributes = [ { name = "attr0", dtype = "int64", distribution = { type = "sequential", start = 1 } }, { name = "attr1", dtype = "int32", distribution = { type = "gaussian", mean = 40, std = 10 } }, { name = "attr2", dtype = "int32", distribution = { type = "zipf", skew = 1.2, low = 10000, high = 99999 } } ] },
    { name = "rel1", num_rows = 500000, attributes = [ { name = "attr0", dtype = "int64", distribution = { type = "sequential", start = 1 } }, { name = "attr1", dtype = "int64", distribution = { type = "uniform", low = 1, high = 100000 } }, { name = "attr2", dtype = "float64", distribution = { type = "uniform", low = 5.0, high = 500.0 } } ] }
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

**GUI mode** (standard mode with visual interface)

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
├── ui/                  # Web-based user interface
│   ├── main.py          # Streamlit application entry point
│   ├── components.py    # UI components and widgets
│   └── models.py        # UI state management and data models
└── utils/               # Shared utilities
    └── toml_parser.py   # Configuration file parsing
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
