import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import tempfile
import toml
import re
import zipfile
import io
from typing import Optional, List, Dict, Any

# Import existing modules
from src.datagen import generate_data_for_iteration
from src.plangen import generate_join_plans_for_iteration
from src.analysis import generate_analysis_for_iteration
from src.visualization import create_visualizations_for_analyses

# ===== CONFIGURATION CONSTANTS =====
CONFIG = {
    "temp_prefix": "djp_web_",
    "iteration_name": "web_generated",
    "patterns": ["star", "linear", "cyclic", "random"],
    "default_patterns": ["star", "linear"],
    "table_limits": {"min": 3, "max": 7, "default": 5},
    "row_options": [10, 100, 1000, 10000, 100000],
    "viz_formats": ["png", "svg"],
    "column_limits": {"min": 3, "max": 7, "default": 5},
    "distribution_types": ["sequential", "uniform", "gaussian", "zipf"],
}

SORT_OPTIONS = [
    "Name (alphabetical)",
    "Total Intermediates ↓",
    "Total Intermediates ↑",
    "Average Selectivity ↓",
    "Average Selectivity ↑",
]


# ===== HELPER FUNCTIONS =====
def get_distribution_config(dist_type: str) -> dict:
    """Return distribution config based on type"""
    configs = {
        "uniform": {"type": "uniform", "low": 1, "high": 1000},
        "gaussian": {"type": "gaussian", "mean": 100.0, "std": 20.0},
        "zipf": {"type": "zipf", "skew": 1.5, "low": 1, "high": 1000},
        "sequential": {"type": "sequential", "start": 1},
    }
    return configs.get(dist_type, configs["uniform"])


def create_table_config(name: str, num_rows: int, columns: list) -> dict:
    """Create a table configuration dictionary"""
    return {"name": name, "num_rows": num_rows, "columns": columns}


def create_simple_tables(
    num_tables: int, rows_per_table: int, columns_per_table: int, distribution: str
) -> list:
    """Create simple table configurations with uniform settings"""
    tables = []
    for i in range(num_tables):
        columns = []
        for j in range(columns_per_table):
            columns.append(
                {
                    "name": f"attr{j}",
                    "dtype": "int64",
                    "distribution": get_distribution_config(distribution),
                }
            )
        tables.append(create_table_config(f"rel{i}", rows_per_table, columns))
    return tables


def create_project_config(
    project_name: str,
    tables: list,
    patterns: list,
    enable_analysis: bool,
    enable_visualization: bool,
    visualization_format: str,
    pattern_settings: dict,
) -> dict:
    """Generate a complete project configuration"""

    # Create base plans from selected patterns
    base_plans = []
    for pattern in patterns:
        base_plans.append(
            {
                "pattern": pattern,
                "num_plans": pattern_settings.get(f"num_plans_{pattern}", 1),
                "permutations": pattern_settings.get(f"permutations_{pattern}", False),
            }
        )

    return {
        "project": {"name": project_name, "output_dir": "output"},
        "iterations": [
            {
                "name": CONFIG["iteration_name"],
                "datagen": {"enabled": True, "tables": tables},
                "plangen": {
                    "enabled": True,
                    "visualize": enable_visualization,
                    "visualization_format": visualization_format,
                    "base_plans": base_plans,
                },
                "analysis": {"enabled": enable_analysis},
            }
        ],
    }


# ===== PLAN METADATA CLASS (keeping this as it's complex) =====
class PlanMetadata:
    """Metadata and analysis for a single join plan"""

    def __init__(self, filename: str, analysis_data: dict):
        self.filename = filename
        self.pattern, self.index, self.permutation, self.type = self.parse_filename(
            filename
        )
        self.base_plan = f"{self.pattern}{self.index}"
        self.analysis_data = analysis_data

        # Calculate key metrics
        stages = analysis_data.get("stages", [])
        self.total_intermediates = (
            stages[-1].get("total_intermediates", 0) if stages else 0
        )
        self.avg_selectivity = (
            sum(stage.get("selectivity", 0) for stage in stages) / len(stages)
            if stages
            else 0
        )

    def parse_filename(self, filename: str):
        """Parse filename to extract plan components"""
        # Remove _analysis.json suffix
        base = filename.replace("_analysis.json", "")

        # Extract type (binary or nary)
        if base.endswith("_binary"):
            plan_type = "binary"
            base = base.replace("_binary", "")
        elif base.endswith("_nary"):
            plan_type = "nary"
            base = base.replace("_nary", "")
        else:
            plan_type = "unknown"

        # Parse permutation if present
        permutation = None
        if "_p" in base:
            parts = base.split("_p")
            base = parts[0]
            try:
                permutation = int(parts[1])
            except (ValueError, IndexError):
                permutation = None

        # Extract pattern and index (e.g., "linear0" -> pattern="linear", index=0)
        match = re.match(r"([a-zA-Z]+)(\d+)", base)
        if match:
            pattern = match.group(1)
            index = int(match.group(2))
        else:
            pattern = base
            index = 0

        return pattern, index, permutation, plan_type

    def get_display_name(self) -> str:
        """Get human-readable display name"""
        if self.permutation is not None:
            return f"{self.base_plan}_p{self.permutation}_{self.type}"
        return f"{self.base_plan}_{self.type}"

    def get_sort_key(self, sort_by: str):
        """Get sort key for the specified sorting method"""
        if sort_by == "Total Intermediates ↓":
            return -self.total_intermediates
        elif sort_by == "Total Intermediates ↑":
            return self.total_intermediates
        elif sort_by == "Average Selectivity ↓":
            return -self.avg_selectivity
        elif sort_by == "Average Selectivity ↑":
            return self.avg_selectivity
        else:
            return self.get_display_name()


# ===== ZIP ARCHIVE HELPER FUNCTIONS =====
def create_zip_archive(file_paths: List[tuple], archive_name: str) -> bytes:
    """Create a ZIP archive from a list of (file_path, archive_path) tuples"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path, archive_path in file_paths:
            if os.path.exists(file_path):
                zip_file.write(file_path, archive_path)

    return zip_buffer.getvalue()


def get_all_data_files(output_dir: str) -> List[tuple]:
    """Get all data files from the output directory"""
    files = []
    iteration_dir = os.path.join(output_dir, CONFIG["iteration_name"])

    if not os.path.exists(iteration_dir):
        return files

    # Add parquet data files
    data_dir = os.path.join(iteration_dir, "data")
    if os.path.exists(data_dir):
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    file_path = os.path.join(root, filename)
                    # Create archive path relative to iteration dir
                    rel_path = os.path.relpath(file_path, iteration_dir)
                    files.append((file_path, rel_path))

    # Add plan JSON files
    plans_dir = os.path.join(iteration_dir, "plans")
    if os.path.exists(plans_dir):
        for filename in os.listdir(plans_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(plans_dir, filename)
                files.append((file_path, f"plans/{filename}"))

    # Add analysis JSON files
    analysis_dir = os.path.join(iteration_dir, "analysis")
    if os.path.exists(analysis_dir):
        for filename in os.listdir(analysis_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(analysis_dir, filename)
                files.append((file_path, f"analysis/{filename}"))

    # Add visualization files
    viz_dir = os.path.join(iteration_dir, "visualizations")
    if os.path.exists(viz_dir):
        for filename in os.listdir(viz_dir):
            if filename.endswith((".png", ".svg")):
                file_path = os.path.join(viz_dir, filename)
                files.append((file_path, f"visualizations/{filename}"))

    return files


def get_filtered_data_files(
    plans: List[PlanMetadata], output_dir: str, viz_format: str
) -> List[tuple]:
    """Get data files for filtered plans only"""
    files = []
    iteration_dir = os.path.join(output_dir, CONFIG["iteration_name"])

    if not os.path.exists(iteration_dir):
        return files

    # Get unique base plans for parquet files
    unique_base_plans = set()
    for plan in plans:
        # Extract table names from the plan - this is a simplified approach
        # In practice, you might need to parse the plan JSON to get exact table relationships
        unique_base_plans.add(plan.base_plan)

    # Add parquet data files (include all since plans reference these tables)
    data_dir = os.path.join(iteration_dir, "data")
    if os.path.exists(data_dir):
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, iteration_dir)
                    files.append((file_path, rel_path))

    # Add specific plan and analysis files for filtered plans
    for plan in plans:
        # Add plan JSON (derive from analysis filename)
        plan_filename = plan.filename.replace("_analysis.json", ".json")
        plan_path = os.path.join(iteration_dir, "plans", plan_filename)
        if os.path.exists(plan_path):
            files.append((plan_path, f"plans/{plan_filename}"))

        # Add analysis JSON
        analysis_path = os.path.join(iteration_dir, "analysis", plan.filename)
        if os.path.exists(analysis_path):
            files.append((analysis_path, f"analysis/{plan.filename}"))

        # Add visualization
        viz_filename = plan.filename.replace(".json", f".{viz_format}")
        viz_path = os.path.join(iteration_dir, "visualizations", viz_filename)
        if os.path.exists(viz_path):
            files.append((viz_path, f"visualizations/{viz_filename}"))

    return files


# ===== SESSION STATE MANAGEMENT =====
def init_session_state():
    """Initialize session state with defaults"""
    defaults = {
        "analysis_results": None,
        "output_dir": None,
        "running_analysis": False,
        "selected_left_plan": None,
        "selected_right_plan": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def is_analysis_running() -> bool:
    return st.session_state.get("running_analysis", False)


def start_analysis():
    st.session_state.running_analysis = True


def complete_analysis(output_dir: str):
    st.session_state.update(
        {
            "running_analysis": False,
            "output_dir": output_dir,
            "analysis_results": output_dir,
        }
    )


# ===== FILE OPERATIONS =====
def find_visualization_file(analysis_filename: str, viz_dir: str) -> Optional[str]:
    """Find visualization file using simple logic"""
    if not os.path.exists(viz_dir):
        return None

    base_name = analysis_filename.replace("_analysis.json", "")

    for ext in CONFIG["viz_formats"]:
        viz_file = f"{base_name}.{ext}"
        if os.path.exists(os.path.join(viz_dir, viz_file)):
            return viz_file
    return None


def load_analysis_plans(analysis_dir: str) -> List[PlanMetadata]:
    """Load and parse analysis files into plan objects"""
    if not os.path.exists(analysis_dir):
        return []

    plans = []
    for filename in os.listdir(analysis_dir):
        if filename.endswith(".json"):
            try:
                filepath = os.path.join(analysis_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                plans.append(PlanMetadata(filename, data))
            except Exception as e:
                st.warning(f"Failed to load {filename}: {e}")

    return plans


def filter_and_sort_plans(
    plans: List[PlanMetadata], base_plan_filter: str, type_filter: str, sort_by: str
) -> List[PlanMetadata]:
    """Filter and sort plans based on user selections"""
    filtered_plans = plans

    # Apply base plan filter
    if base_plan_filter != "All":
        filtered_plans = [p for p in filtered_plans if p.base_plan == base_plan_filter]

    # Apply type filter
    if type_filter != "All":
        filtered_plans = [p for p in filtered_plans if p.type == type_filter]

    # Sort plans
    filtered_plans.sort(key=lambda p: p.get_sort_key(sort_by))

    return filtered_plans


# ===== ANALYSIS RUNNER =====
def run_djp_generator(config_dict: dict) -> Optional[str]:
    """Run the DJP analysis pipeline with the given configuration"""

    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp(prefix=CONFIG["temp_prefix"])
    config_dict["project"]["output_dir"] = temp_dir

    try:
        for iter_config in config_dict["iterations"]:
            iter_name = iter_config["name"]
            output_dir = os.path.join(temp_dir, iter_name)

            # Data generation
            datagen_config = iter_config.get("datagen", {})
            if datagen_config.get("enabled", False):
                with st.spinner(text="Generating Data..."):
                    generate_data_for_iteration(datagen_config, output_dir)

            # Plan generation
            plangen_config = iter_config.get("plangen", {})
            if plangen_config.get("enabled", False):
                with st.spinner(text="Generating Join Plans..."):
                    generate_join_plans_for_iteration(
                        plangen_config, datagen_config, output_dir
                    )

            # Analysis
            analysis_config = iter_config.get("analysis", {})
            if analysis_config.get("enabled", False):
                with st.spinner(text="Generating Join Plans Analyses..."):
                    generate_analysis_for_iteration(output_dir)

            # Visualization
            if plangen_config.get("visualize", False):
                with st.spinner(text="Generating Join Plans Visualizations..."):
                    analysis_dir = os.path.join(output_dir, "analysis")
                    visualizations_dir = os.path.join(output_dir, "visualizations")
                    visualization_format = plangen_config.get(
                        "visualization_format", "png"
                    )
                    create_visualizations_for_analyses(
                        analysis_dir, visualizations_dir, visualization_format
                    )

        return temp_dir

    except Exception as e:
        st.error(f"DJP Generator failed: {str(e)}")
        return None


# ===== UI RENDERING FUNCTIONS =====
def render_simple_table_config(num_tables: int) -> list:
    """Render simple table configuration"""
    rows = st.sidebar.selectbox("Rows per Table", CONFIG["row_options"], index=1)
    columns = st.sidebar.slider(
        "Columns per Table",
        CONFIG["column_limits"]["min"],
        CONFIG["column_limits"]["max"],
        CONFIG["column_limits"]["default"],
    )
    distribution = st.sidebar.selectbox(
        "Default Distribution", CONFIG["distribution_types"], index=1
    )

    return create_simple_tables(num_tables, rows, columns, distribution)


def render_advanced_table_config(num_tables: int) -> list:
    """Render advanced table configuration with per-table customization"""
    table_configs = []

    for i in range(num_tables):
        with st.sidebar.expander(f"Table {i} Settings", expanded=False):
            table_name = st.text_input(
                f"Table {i} Name", value=f"rel{i}", key=f"table_name_{i}"
            )
            table_rows = st.number_input(
                f"Table {i} Rows",
                min_value=100,
                max_value=100000,
                value=1000,
                key=f"table_rows_{i}",
            )
            num_columns = st.slider(
                f"Table {i} Columns", 3, 7, 5, key=f"table_cols_{i}"
            )

            columns = []
            for j in range(num_columns):
                col_name = st.text_input(
                    f"Column {j} Name", value=f"attr{j}", key=f"col_name_{i}_{j}"
                )
                dtype = st.selectbox(
                    f"Column {j} Type",
                    ["int32", "int64", "float64"],
                    key=f"col_dtype_{i}_{j}",
                )

                # Distribution configuration
                dist_type = st.selectbox(
                    f"Column {j} Distribution",
                    CONFIG["distribution_types"],
                    key=f"col_dist_{i}_{j}",
                )

                distribution = {"type": dist_type}
                if dist_type == "sequential":
                    start_val = st.number_input(
                        f"Start Value", value=1, key=f"seq_start_{i}_{j}"
                    )
                    distribution["start"] = start_val
                elif dist_type == "uniform":
                    low = st.number_input(
                        f"Min Value", value=1, key=f"uniform_low_{i}_{j}"
                    )
                    high = st.number_input(
                        f"Max Value", value=1000, key=f"uniform_high_{i}_{j}"
                    )
                    distribution.update({"low": low, "high": high})
                elif dist_type == "gaussian":
                    mean = st.number_input(
                        f"Mean", value=100.0, key=f"gauss_mean_{i}_{j}"
                    )
                    std = st.number_input(
                        f"Std Dev", value=20.0, key=f"gauss_std_{i}_{j}"
                    )
                    distribution.update({"mean": mean, "std": std})
                elif dist_type == "zipf":
                    skew = st.number_input(f"Skew", value=1.5, key=f"zipf_skew_{i}_{j}")
                    low = st.number_input(
                        f"Min Value", value=1, key=f"zipf_low_{i}_{j}"
                    )
                    high = st.number_input(
                        f"Max Value", value=1000, key=f"zipf_high_{i}_{j}"
                    )
                    distribution.update({"skew": skew, "low": low, "high": high})

                # Null ratio
                null_ratio = st.slider(
                    f"Column {j} Null %", 0.0, 0.5, 0.0, key=f"null_ratio_{i}_{j}"
                )

                col_config = {
                    "name": col_name,
                    "dtype": dtype,
                    "distribution": distribution,
                }
                if null_ratio > 0:
                    col_config["null_ratio"] = null_ratio

                columns.append(col_config)

            table_configs.append(create_table_config(table_name, table_rows, columns))

    return table_configs


def render_pattern_settings(patterns: list) -> dict:
    """Render pattern-specific settings and return configuration"""
    pattern_settings = {}

    if patterns:
        with st.sidebar.expander("Pattern Settings", expanded=False):
            for pattern in patterns:
                st.write(f"**{pattern.title()} Pattern**")
                pattern_settings[f"num_plans_{pattern}"] = st.number_input(
                    f"Number of {pattern} plans",
                    min_value=1,
                    max_value=5,
                    value=1,
                    key=f"num_plans_{pattern}",
                )
                pattern_settings[f"permutations_{pattern}"] = st.checkbox(
                    f"Enable {pattern} permutations",
                    value=pattern in CONFIG["patterns"],
                    key=f"permutations_{pattern}",
                )

    return pattern_settings


def render_sidebar_config() -> dict:
    """Render sidebar and return configuration dict"""
    st.sidebar.title("Configuration")

    # Project settings
    st.sidebar.subheader("Project Settings")
    project_name = st.sidebar.text_input(
        "Project Name", value="Synthetic Data & Join Plans"
    )

    # Table configuration
    st.sidebar.subheader("Table Configuration")
    num_tables = st.sidebar.slider(
        "Number of Tables",
        CONFIG["table_limits"]["min"],
        CONFIG["table_limits"]["max"],
        CONFIG["table_limits"]["default"],
    )

    # Table config method
    if st.sidebar.checkbox("Advanced Table Configuration", value=False):
        tables = render_advanced_table_config(num_tables)
    else:
        tables = render_simple_table_config(num_tables)

    # Join patterns configuration
    st.sidebar.subheader("Join Patterns")
    patterns = st.sidebar.multiselect(
        "Select Join Patterns", CONFIG["patterns"], default=CONFIG["default_patterns"]
    )

    # Pattern-specific settings
    pattern_settings = render_pattern_settings(patterns)

    # Analysis options
    st.sidebar.subheader("Analysis Options")
    enable_analysis = st.sidebar.checkbox("Enable Analysis", value=True)
    enable_visualization = st.sidebar.checkbox("Generate Visualizations", value=True)
    visualization_format = st.sidebar.selectbox(
        "Visualization Format", CONFIG["viz_formats"], index=0
    )

    # Configuration summary
    st.sidebar.divider()
    st.sidebar.subheader("Current Configuration")
    if patterns:
        st.sidebar.success(f"{len(patterns)} pattern(s) selected")
        st.sidebar.info(f"{num_tables} tables, {CONFIG['row_options'][1]:,} rows each")
    else:
        st.sidebar.warning("Please select join patterns")

    # Download config
    if patterns:
        config_dict = create_project_config(
            project_name,
            tables,
            patterns,
            enable_analysis,
            enable_visualization,
            visualization_format,
            pattern_settings,
        )
        config_toml = toml.dumps(config_dict)
        st.sidebar.download_button(
            label="Download Config",
            data=config_toml,
            file_name=f"{project_name.lower().replace(' ', '_')}.toml",
            mime="text/plain",
        )

    st.sidebar.divider()

    # Run button
    if st.sidebar.button(
        "Run",
        type="primary",
        disabled=not patterns or is_analysis_running(),
        width="stretch",
    ):
        if patterns:
            start_analysis()
            st.rerun()

    return {
        "project_name": project_name,
        "tables": tables,
        "patterns": patterns,
        "enable_analysis": enable_analysis,
        "enable_visualization": enable_visualization,
        "visualization_format": visualization_format,
        "pattern_settings": pattern_settings,
    }


def render_plan_filters(plans: List[PlanMetadata]) -> tuple[List[PlanMetadata], str]:
    """Render filtering controls and return filtered plans and sort option"""
    unique_base_plans = sorted(set(p.base_plan for p in plans))
    unique_types = sorted(set(p.type for p in plans if p.type != "unknown"))

    st.subheader("Plan Selection & Filtering")
    col1, col2, col3 = st.columns(3)

    with col1:
        base_filter = st.selectbox("Base Plan", ["All"] + unique_base_plans)
    with col2:
        type_filter = st.selectbox("Type", ["All"] + unique_types)
    with col3:
        sort_by = st.selectbox("Sort Plans By", SORT_OPTIONS)

    filtered_plans = filter_and_sort_plans(plans, base_filter, type_filter, sort_by)
    return filtered_plans, sort_by


def create_plan_charts(plan: PlanMetadata, title_prefix: str):
    """Create plotly charts for a plan"""
    stages = plan.analysis_data.get("stages", [])
    if not stages:
        return None, None, None

    # Prepare data
    stage_data = []
    for stage in stages:
        stage_data.append(
            {
                "Stage": stage["stage"],
                "Output Size": stage["output_size"],
                "Selectivity": stage.get("selectivity", 0),
                "Total Intermediates": stage.get("total_intermediates", 0),
                "Tables": ", ".join(stage["tables"]),
            }
        )

    df = pd.DataFrame(stage_data)

    # Create charts
    fig1 = px.bar(
        df,
        x="Stage",
        y="Total Intermediates",
        hover_data=["Tables"],
        title=f"{title_prefix} - Cumulative Intermediates",
    )
    fig1.update_layout(height=300)

    fig2 = px.line(
        df,
        x="Stage",
        y="Selectivity",
        markers=True,
        hover_data=["Tables"],
        title=f"{title_prefix} - Selectivity by Stage",
    )
    fig2.update_layout(height=300)

    return fig1, fig2, df


def render_plan_comparison_controls(
    filtered_plans: List[PlanMetadata], sort_by: str
) -> tuple:
    """Render plan selection controls and return selected plans"""
    st.subheader("Side-by-Side Comparison")

    plan_names = [p.get_display_name() for p in filtered_plans]

    # Get current selections - ensure indices are updated after button clicks
    current_left_idx = 0
    current_right_idx = min(1, len(plan_names) - 1)

    # Update indices based on session state (this handles button clicks)
    if (
        "selected_left_plan" in st.session_state
        and st.session_state["selected_left_plan"] in plan_names
    ):
        current_left_idx = plan_names.index(st.session_state["selected_left_plan"])

    if (
        "selected_right_plan" in st.session_state
        and st.session_state["selected_right_plan"] in plan_names
    ):
        current_right_idx = plan_names.index(st.session_state["selected_right_plan"])

    # Plan selection dropdowns
    col1, col2 = st.columns(2)
    with col1:
        left_plan_name = st.selectbox(
            "Left Plan", plan_names, index=current_left_idx, key="left_plan_select"
        )
    with col2:
        right_plan_name = st.selectbox(
            "Right Plan", plan_names, index=current_right_idx, key="right_plan_select"
        )

    # Update session state when selectboxes change
    st.session_state["selected_left_plan"] = left_plan_name
    st.session_state["selected_right_plan"] = right_plan_name

    # Quick action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Find Pair for Left Plan",
            help="Find the binary/nary pair for the currently selected left plan",
        ):
            current_left_plan = next(
                (p for p in filtered_plans if p.get_display_name() == left_plan_name),
                None,
            )
            if current_left_plan:
                target_type = "nary" if current_left_plan.type == "binary" else "binary"
                matching_plan = next(
                    (
                        p
                        for p in filtered_plans
                        if p.base_plan == current_left_plan.base_plan
                        and p.permutation == current_left_plan.permutation
                        and p.type == target_type
                    ),
                    None,
                )
                if matching_plan:
                    st.session_state["selected_left_plan"] = (
                        current_left_plan.get_display_name()
                    )
                    st.session_state["selected_right_plan"] = (
                        matching_plan.get_display_name()
                    )
                    st.rerun()
                else:
                    st.warning(
                        f"No {target_type} pair found for {current_left_plan.get_display_name()}"
                    )

    with col2:
        is_numeric_sort = any(
            metric in sort_by
            for metric in ["Total Intermediates", "Average Selectivity"]
        )

        if is_numeric_sort and st.button(
            "Best Binary vs Best Nary",
            help="Select highest rated binary and nary plans",
        ):
            binary_plans = [p for p in filtered_plans if p.type == "binary"]
            nary_plans = [p for p in filtered_plans if p.type == "nary"]

            if binary_plans and nary_plans:
                best_binary = binary_plans[0]
                best_nary = nary_plans[0]

                st.session_state["selected_left_plan"] = best_binary.get_display_name()
                st.session_state["selected_right_plan"] = best_nary.get_display_name()
                st.rerun()
            else:
                st.warning("Need both binary and nary plans in filtered results")

    # Get selected plans
    left_plan = next(
        (p for p in filtered_plans if p.get_display_name() == left_plan_name), None
    )
    right_plan = next(
        (p for p in filtered_plans if p.get_display_name() == right_plan_name), None
    )

    return left_plan, right_plan


def render_plan_metrics_comparison(left_plan: PlanMetadata, right_plan: PlanMetadata):
    """Render side-by-side metrics comparison"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{left_plan.get_display_name()}")
        lcol1, lcol2 = st.columns(2)
        with lcol1:
            st.metric("Total Intermediates", f"{left_plan.total_intermediates:,}")
        with lcol2:
            st.metric("Avg Selectivity", f"{left_plan.avg_selectivity:.4f}")

    with col2:
        st.subheader(f"{right_plan.get_display_name()}")
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            delta_int = right_plan.total_intermediates - left_plan.total_intermediates
            st.metric(
                "Total Intermediates",
                f"{right_plan.total_intermediates:,}",
                delta=f"{delta_int:+,}" if delta_int != 0 else None,
            )
        with rcol2:
            delta_sel = right_plan.avg_selectivity - left_plan.avg_selectivity
            st.metric(
                "Avg Selectivity",
                f"{right_plan.avg_selectivity:.4f}",
                delta=f"{delta_sel:+.4f}" if abs(delta_sel) > 0.0001 else None,
            )


def render_plan_charts_comparison(left_plan: PlanMetadata, right_plan: PlanMetadata):
    """Render charts comparison"""
    st.subheader("Performance Comparison")

    # Generate charts for both plans
    left_sel_fig, left_int_fig, left_df = create_plan_charts(left_plan, "Left Plan")
    right_sel_fig, right_int_fig, right_df = create_plan_charts(
        right_plan, "Right Plan"
    )

    if left_sel_fig and right_sel_fig:
        # Display charts in tabs
        tab1, tab2 = st.tabs(
            ["Intermediate Results Comparison", "Selectivity Comparison"]
        )

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(left_sel_fig, width="stretch")
            with col2:
                st.plotly_chart(right_sel_fig, width="stretch")

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(left_int_fig, width="stretch")
            with col2:
                st.plotly_chart(right_int_fig, width="stretch")

    return left_df, right_df


def render_stage_details_comparison(
    left_plan: PlanMetadata, right_plan: PlanMetadata, left_df, right_df
):
    """Render detailed stage comparison"""
    st.subheader("Detailed Stage Comparison")

    col1, col2 = st.columns(2)

    with col1:
        if left_df is not None:
            left_display = pd.DataFrame(
                [
                    {
                        "Stage": stage["stage"],
                        "Output Size": f"{stage['output_size']:,}",
                        "Selectivity": f"{stage.get('selectivity', 0):.4f}",
                        "Total Int": f"{stage.get('total_intermediates', 0):,}",
                        "Tables": ", ".join(stage["tables"]),
                    }
                    for stage in left_plan.analysis_data.get("stages", [])
                ]
            )
            st.dataframe(left_display, width="stretch")

    with col2:
        if right_df is not None:
            right_display = pd.DataFrame(
                [
                    {
                        "Stage": stage["stage"],
                        "Output Size": f"{stage['output_size']:,}",
                        "Selectivity": f"{stage.get('selectivity', 0):.4f}",
                        "Total Int": f"{stage.get('total_intermediates', 0):,}",
                        "Tables": ", ".join(stage["tables"]),
                    }
                    for stage in right_plan.analysis_data.get("stages", [])
                ]
            )
            st.dataframe(right_display, width="stretch")


def render_visualizations_comparison(
    left_plan: PlanMetadata,
    right_plan: PlanMetadata,
    viz_dir: str,
    viz_format: str = "png",
):
    """Render visualization comparison"""
    if not os.path.exists(viz_dir):
        return

    st.subheader("Join Plan Visualizations")

    # Use the specific format that was configured
    # Visualization files keep the "_analysis" part: e.g., "star0_p0_binary_analysis.png"
    left_viz_file = f"{left_plan.filename.replace('.json', f'.{viz_format}')}"
    right_viz_file = f"{right_plan.filename.replace('.json', f'.{viz_format}')}"

    col1, col2 = st.columns(2)

    with col1:
        left_viz_path = os.path.join(viz_dir, left_viz_file)
        if os.path.exists(left_viz_path):
            st.image(left_viz_path, caption=left_plan.get_display_name())
        else:
            st.info(f"Visualization not found: {left_viz_file}")

    with col2:
        right_viz_path = os.path.join(viz_dir, right_viz_file)
        if os.path.exists(right_viz_path):
            st.image(right_viz_path, caption=right_plan.get_display_name())
        else:
            st.info(f"Visualization not found: {right_viz_file}")


def get_current_comparison_files(
    left_plan: PlanMetadata, right_plan: PlanMetadata, output_dir: str, viz_format: str
) -> List[tuple]:
    """Get files for the current left and right plan comparison"""
    files = []
    iteration_dir = os.path.join(output_dir, CONFIG["iteration_name"])

    if not os.path.exists(iteration_dir):
        return files

    # Add all parquet data files (since both plans reference these tables)
    data_dir = os.path.join(iteration_dir, "data")
    if os.path.exists(data_dir):
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, iteration_dir)
                    files.append((file_path, rel_path))

    # Add files for both left and right plans
    for plan in [left_plan, right_plan]:
        # Add plan JSON (derive from analysis filename)
        plan_filename = plan.filename.replace("_analysis.json", ".json")
        plan_path = os.path.join(iteration_dir, "plans", plan_filename)
        if os.path.exists(plan_path):
            files.append((plan_path, f"plans/{plan_filename}"))

        # Add analysis JSON
        analysis_path = os.path.join(iteration_dir, "analysis", plan.filename)
        if os.path.exists(analysis_path):
            files.append((analysis_path, f"analysis/{plan.filename}"))

        # Add visualization
        viz_filename = plan.filename.replace(".json", f".{viz_format}")
        viz_path = os.path.join(iteration_dir, "visualizations", viz_filename)
        if os.path.exists(viz_path):
            files.append((viz_path, f"visualizations/{viz_filename}"))

    return files


def render_bulk_download_section(
    filtered_plans: List[PlanMetadata],
    output_dir: str,
    viz_format: str,
    left_plan: PlanMetadata = None,
    right_plan: PlanMetadata = None,
):
    """Render download section"""
    st.subheader("Downloads")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Complete Dataset**")

        # Get all files
        all_files = get_all_data_files(output_dir)

        if all_files:
            # Count files by type
            parquet_count = len([f for f in all_files if f[1].endswith(".parquet")])
            json_count = len([f for f in all_files if f[1].endswith(".json")])
            viz_count = len([f for f in all_files if f[1].endswith((".png", ".svg"))])

            st.info(
                f"{parquet_count} data files, {json_count} JSON files, {viz_count} visualizations"
            )

            # Create ZIP archive
            zip_data = create_zip_archive(all_files, "complete_dataset")

            st.download_button(
                label="Download All Data (ZIP)",
                data=zip_data,
                file_name="djp_complete_dataset.zip",
                mime="application/zip",
                key="download_all_data",
                help="Downloads all generated data including parquet files, plans, analysis results, and visualizations",
            )
        else:
            st.warning("No data files found")

    with col2:
        st.write("**Filtered Dataset**")

        if filtered_plans:
            # Get filtered files
            filtered_files = get_filtered_data_files(
                filtered_plans, output_dir, viz_format
            )

            if filtered_files:
                # Count files by type
                parquet_count = len(
                    [f for f in filtered_files if f[1].endswith(".parquet")]
                )
                json_count = len([f for f in filtered_files if f[1].endswith(".json")])
                viz_count = len(
                    [f for f in filtered_files if f[1].endswith((".png", ".svg"))]
                )

                st.info(
                    f"{parquet_count} data files, {json_count} JSON files, {viz_count} visualizations"
                )

                # Create ZIP archive
                zip_data = create_zip_archive(filtered_files, "filtered_dataset")

                st.download_button(
                    label="Download Filtered Data (ZIP)",
                    data=zip_data,
                    file_name="djp_filtered_dataset.zip",
                    mime="application/zip",
                    key="download_filtered_data",
                    help="Downloads data for currently filtered/displayed plans only",
                )
            else:
                st.warning("No files found for filtered plans")
        else:
            st.warning("No plans available for filtering")

    with col3:
        st.write("**Current Dataset**")

        if left_plan and right_plan:
            # Get files for current comparison
            current_files = get_current_comparison_files(
                left_plan, right_plan, output_dir, viz_format
            )

            if current_files:
                # Count files by type
                parquet_count = len(
                    [f for f in current_files if f[1].endswith(".parquet")]
                )
                json_count = len([f for f in current_files if f[1].endswith(".json")])
                viz_count = len(
                    [f for f in current_files if f[1].endswith((".png", ".svg"))]
                )

                st.info(
                    f"{parquet_count} data files, {json_count} JSON files, {viz_count} visualizations"
                )

                # Create ZIP archive
                zip_data = create_zip_archive(current_files, "current_comparison")

                comparison_name = (
                    f"{left_plan.get_display_name()}_vs_{right_plan.get_display_name()}"
                )

                st.download_button(
                    label="Download Current Dataset (ZIP)",
                    data=zip_data,
                    file_name=f"djp_comparison_{comparison_name}.zip",
                    mime="application/zip",
                    key="download_current_comparison",
                    help="Downloads all data, plans, analysis, and visualizations for the current left & right plan comparison",
                )
            else:
                st.warning("No files found for current comparison")
        else:
            st.info("Select plans for comparison to enable current dataset download")


def render_plan_comparison(
    filtered_plans: List[PlanMetadata],
    output_dir: str,
    sort_by: str,
    viz_format: str = "png",
):
    """Render the complete plan comparison section"""
    left_plan, right_plan = render_plan_comparison_controls(filtered_plans, sort_by)

    if not left_plan or not right_plan:
        st.error("Selected plans not found.")
        return

    st.divider()

    # Metrics comparison
    render_plan_metrics_comparison(left_plan, right_plan)

    # Charts comparison
    left_df, right_df = render_plan_charts_comparison(left_plan, right_plan)

    # Stage details
    render_stage_details_comparison(left_plan, right_plan, left_df, right_df)

    # Visualizations
    viz_dir = os.path.join(output_dir, CONFIG["iteration_name"], "visualizations")
    render_visualizations_comparison(left_plan, right_plan, viz_dir, viz_format)

    st.divider()

    # Downloads
    render_bulk_download_section(
        filtered_plans, output_dir, viz_format, left_plan, right_plan
    )


# ===== PAGE RENDERING FUNCTIONS =====
def render_analysis_running_page(config: dict):
    """Handle the analysis running state"""
    # Save visualization format to session state for later use
    st.session_state.visualization_format = config["visualization_format"]

    # Convert flattened config to the expected format
    config_dict = create_project_config(
        config["project_name"],
        config["tables"],
        config["patterns"],
        config["enable_analysis"],
        config["enable_visualization"],
        config["visualization_format"],
        config["pattern_settings"],
    )

    output_dir = run_djp_generator(config_dict)

    if output_dir:
        complete_analysis(output_dir)
        st.success("Analysis completed successfully!")
        st.rerun()
    else:
        st.session_state.running_analysis = False
        st.error("Analysis failed. Please check your configuration.")


def render_results_page():
    """Render the results analysis page"""
    st.header("Analysis Results")

    output_dir = st.session_state.output_dir
    analysis_dir = os.path.join(output_dir, CONFIG["iteration_name"], "analysis")

    if not os.path.exists(analysis_dir):
        st.error("Analysis directory not found.")
        return

    # Load and parse all plans
    plans = load_analysis_plans(analysis_dir)

    if not plans:
        st.error("No analysis files found.")
        return

    # Filter and sort plans
    filtered_plans, sort_by = render_plan_filters(plans)

    if not filtered_plans:
        st.warning("No plans match the current filters.")
        return

    # Get the visualization format from session state or default to png
    viz_format = st.session_state.get("visualization_format", "png")

    # Plan comparison
    render_plan_comparison(filtered_plans, output_dir, sort_by, viz_format)


def render_welcome_page():
    st.markdown("""
    # Data & Join Plan Generator
    ### Get Started:
    1. **Configure** your data and join plan generation parameters in the sidebar
    2. **Click** the "Run" button to generate results
    3. **View** interactive charts and download results
    """)


def main():
    st.set_page_config(
        page_title="DJP Generator", layout="wide", initial_sidebar_state="expanded"
    )

    init_session_state()

    # Always render sidebar
    config = render_sidebar_config()

    # Render appropriate page based on state
    if is_analysis_running():
        render_analysis_running_page(config)
    elif st.session_state.analysis_results:
        render_results_page()
    else:
        render_welcome_page()


if __name__ == "__main__":
    main()
