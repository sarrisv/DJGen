import io
import json
import os
import re
import tempfile
import zipfile
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st

from src.analysis import generate_analysis_for_iteration
from src.datagen import generate_data_for_iteration
from src.plangen import generate_join_plans_for_iteration
from src.visualization import create_visualizations_for_analyses

CONFIG = {
    "temp_prefix": "djp_web_",
    "iteration_name": "web_generated",
    "patterns": ["star", "linear", "cyclic", "random"],
    "default_patterns": ["star", "linear"],
    "table_limits": {"min": 3, "max": 7, "default": 5},
    "row_options": [10, 100, 1000, 10000, 100000],
    "viz_formats": ["png", "svg"],
    "column_limits": {"min": 3, "max": 10, "default": 5},
    "distribution_types": ["sequential", "uniform", "gaussian", "zipf"],
    "data_types": ["int32", "int64", "float32", "float64"],
}

SORT_OPTIONS = [
    "Name (alphabetical)",
    "Total Intermediates ↓",
    "Total Intermediates ↑",
    "Average Selectivity ↓",
    "Average Selectivity ↑",
]

VALIDATION_RULES = {
    "uniform": {
        "low": {"min": 0, "help": "Minimum value for uniform distribution"},
        "high": {"min": 1, "help": "Maximum value (must be > minimum)"},
    },
    "gaussian": {
        "mean": {"min": 0, "help": "Mean (center) of the distribution"},
        "std": {"min": 0.1, "help": "Standard deviation (spread)"},
    },
    "zipf": {
        "skew": {"min": 1.0, "help": "Skewness parameter (higher = more skewed)"},
        "low": {"min": 0, "help": "Minimum value"},
        "high": {"min": 1, "help": "Maximum value (must be > minimum)"},
    },
    "sequential": {
        "start": {"min": 0, "help": "Starting value for sequential numbering"}
    },
}


class PlanMetadata:
    """Enhanced plan metadata with consistent metric calculation"""

    def __init__(self, filename: str, analysis_data: dict):
        self.filename = filename
        self.analysis_data = analysis_data
        self.pattern, self.index, self.permutation, self.type = self._parse_filename(
            filename
        )
        self.base_plan = f"{self.pattern}{self.index}"

        # Calculate metrics consistently
        stages = analysis_data.get("stages", [])
        self.total_intermediates = (
            stages[-1].get("total_intermediates", 0) if stages else 0
        )
        self.final_output_size = stages[-1].get("output_size", 0) if stages else 0

        if self.type == "binary":
            self.max_intermediate_size = (
                max((stage.get("output_size", 0) for stage in stages), default=0)
                if stages
                else 0
            )
        else:
            self.max_intermediate_size = self.final_output_size

        self.avg_selectivity = (
            sum(stage.get("selectivity", 0) for stage in stages) / len(stages)
            if stages
            else 0
        )

    def _parse_filename(self, filename: str):
        """Parse filename consistently"""
        # Remove the analysis suffix to get the base name
        base = filename.replace("_analysis.json", "")

        # Extract plan type from suffix
        if base.endswith("_binary"):
            plan_type = "binary"
            base = base.replace("_binary", "")
        elif base.endswith("_nary"):
            plan_type = "nary"
            base = base.replace("_nary", "")
        else:
            plan_type = "unknown"

        # Extract permutation number if present (e.g., "star1_p2" -> permutation=2)
        permutation = None
        if "_p" in base:
            parts = base.split("_p")
            base = parts[0]
            try:
                permutation = int(parts[1])
            except (ValueError, IndexError):
                # Invalid permutation format, ignore
                pass

        # Extract pattern name and index (e.g., "star1" -> pattern="star", index=1)
        match = re.match(r"([a-zA-Z]+)(\d+)", base)
        if match:
            pattern = match.group(1)
            index = int(match.group(2))
        else:
            # No numeric suffix found
            pattern = base
            index = 0

        return pattern, index, permutation, plan_type

    def get_display_name(self) -> str:
        """Get consistent display name"""
        name = f"{self.base_plan}_{self.type}"
        if self.permutation is not None:
            name = f"{self.base_plan}_p{self.permutation}_{self.type}"
        return name

    def get_sort_key(self, sort_by: str):
        """Get sort key consistently"""
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


def init_session_state() -> None:
    """Initialize all session state with comprehensive defaults"""
    defaults = {
        "analysis_results": None,
        "output_dir": None,
        "running_analysis": False,
        "selected_left_plan": None,
        "selected_right_plan": None,
        "advanced_mode": False,
        "advanced_tables": [],
        "visualization_format": "png",
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def create_project_config(
    project_name: str,
    tables: List[Dict[str, Any]],
    patterns: List[str],
    enable_analysis: bool,
    enable_visualization: bool,
    visualization_format: str,
    pattern_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """Create project configuration with consistent structure"""
    base_plans = []
    for pattern in patterns:
        base_plans.append(
            {
                "pattern": pattern,
                "num_plans": pattern_settings.get(f"pattern_num_plans_{pattern}", 1),
                "permutations": pattern_settings.get(
                    f"pattern_permutations_{pattern}", False
                ),
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


def run_djp_generator(config_dict: Dict[str, Any]) -> Optional[str]:
    """Run DJP generator with consistent error handling"""
    temp_dir = tempfile.mkdtemp(prefix=CONFIG["temp_prefix"])
    config_dict["project"]["output_dir"] = temp_dir

    try:
        for iter_config in config_dict["iterations"]:
            iter_name = iter_config["name"]
            output_dir = os.path.join(temp_dir, iter_name)

            datagen_config = iter_config.get("datagen", {})
            if datagen_config.get("enabled", False):
                with st.spinner("Generating data..."):
                    generate_data_for_iteration(datagen_config, output_dir)

            plangen_config = iter_config.get("plangen", {})
            if plangen_config.get("enabled", False):
                with st.spinner("Generating join plans..."):
                    generate_join_plans_for_iteration(
                        plangen_config, datagen_config, output_dir
                    )

            analysis_config = iter_config.get("analysis", {})
            if analysis_config.get("enabled", False):
                with st.spinner("Analyzing plans..."):
                    generate_analysis_for_iteration(output_dir)

            if plangen_config.get("visualize", False):
                with st.spinner("Creating visualizations..."):
                    analysis_dir = os.path.join(output_dir, "analysis")
                    viz_dir = os.path.join(output_dir, "visualizations")
                    viz_format = plangen_config.get("visualization_format", "png")
                    create_visualizations_for_analyses(
                        analysis_dir, viz_dir, viz_format
                    )

        return temp_dir
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None


def get_best_plans_by_type(
    plans: List[PlanMetadata], sort_by: str
) -> Tuple[Optional[PlanMetadata], Optional[PlanMetadata]]:
    """Find the best binary and n-ary plans based on current sorting"""
    binary_plans = [p for p in plans if p.type == "binary"]
    nary_plans = [p for p in plans if p.type == "nary"]

    if binary_plans:
        binary_plans.sort(key=lambda p: p.get_sort_key(sort_by))
    if nary_plans:
        nary_plans.sort(key=lambda p: p.get_sort_key(sort_by))

    best_binary = binary_plans[0] if binary_plans else None
    best_nary = nary_plans[0] if nary_plans else None

    return best_binary, best_nary


def load_analysis_plans(analysis_dir: str) -> List[PlanMetadata]:
    """Load analysis plans with consistent error handling"""
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


def create_zip_archive(file_paths: List[Tuple[str, str]], archive_name: str) -> bytes:
    """Create ZIP archive from file paths"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path, archive_path in file_paths:
            if os.path.exists(file_path):
                zip_file.write(file_path, archive_path)
    return zip_buffer.getvalue()


def get_all_data_files(output_dir: str) -> List[Tuple[str, str]]:
    """Get all data files from output directory"""
    files = []
    iteration_dir = os.path.join(output_dir, CONFIG["iteration_name"])

    if not os.path.exists(iteration_dir):
        return files

    # Add parquet data files
    data_dir = os.path.join(iteration_dir, "data")
    if os.path.exists(data_dir):
        for root, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, iteration_dir)
                    files.append((file_path, rel_path))

    # Add plan and analysis JSON files
    for subdir in ["plans", "analysis"]:
        target_dir = os.path.join(iteration_dir, subdir)
        if os.path.exists(target_dir):
            for filename in os.listdir(target_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(target_dir, filename)
                    files.append((file_path, f"{subdir}/{filename}"))

    # Add visualization files
    viz_dir = os.path.join(iteration_dir, "visualizations")
    if os.path.exists(viz_dir):
        for filename in os.listdir(viz_dir):
            if filename.endswith((".png", ".svg")):
                file_path = os.path.join(viz_dir, filename)
                files.append((file_path, f"visualizations/{filename}"))

    return files

