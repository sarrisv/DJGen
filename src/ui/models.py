import io
import json
import os
import re
import tempfile
import zipfile
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st
import toml
import tomllib

from src.analysis import generate_analysis_for_iteration
from src.datagen import generate_data_for_iteration
from src.plangen import generate_join_plans_for_iteration
from src.visualization import create_visualizations_for_plans
from src.utils.toml_parser import get_default_config

CONFIG = {
    "temp_prefix": "djp_web_",
    "iteration_name": "web_generated",
    "patterns": ["star", "linear", "cyclic", "random"],
    "default_patterns": ["star", "linear"],
    "rel_limits": {"min": 3, "max": 10, "default": 5},
    "row_options": [10, 100, 1000, 10000, 100000],
    "viz_formats": ["png", "svg"],
    "attribute_limits": {"min": 3, "max": 10, "default": 5},
    "distribution_types": ["sequential", "uniform", "gaussian", "zipf"],
    "data_types": ["int32", "int64", "float32", "float64"],
}

SORT_OPTIONS = [
    "Name (alphabetical)",
    "Total Results ↓",
    "Total Results ↑",
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


def _load_and_process_defaults() -> Dict[str, Any]:
    """
    Loads the master default configuration from the toml_parser
    and flattens it into a simple structure for UI comparisons.
    """
    full_defaults = get_default_config()

    # Extract the template objects from the full config
    default_iteration = full_defaults["iterations"][0]
    default_attribute = default_iteration["datagen"]["relations"][0]["attributes"][0]
    default_plangen = default_iteration["plangen"]
    default_base_plan = default_plangen["base_plans"][0]

    return {
        "attribute": {
            "dtype": default_attribute["dtype"],
            "distribution": default_attribute["distribution"],
        },
        "base_plan": {
            "num_plans": default_base_plan["num_plans"],
            "permutations": default_base_plan["permutations"],
        },
        "plangen": {
            "visualize": default_plangen["visualize"],
            "visualization_format": default_plangen["visualization_format"],
        },
    }


DEFAULTS = _load_and_process_defaults()


class PlanMetadata:
    """Enhanced plan metadata with consistent metric calculation"""

    def __init__(self, filename: str, plan: dict):
        self.filename = filename
        self.plan_data = plan
        self.pattern, self.index, self.permutation, self.type = self._parse_filename(
            filename
        )
        self.base_plan = f"{self.pattern}{self.index}"

        analysis_section = plan.get("analysis", {})
        stages = analysis_section.get("stages", [])
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
        base = filename.replace(".json", "")

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
        if sort_by == "Total Results ↓":
            return -self.total_intermediates
        elif sort_by == "Total Results ↑":
            return self.total_intermediates
        elif sort_by == "Average Selectivity ↓":
            return -self.avg_selectivity
        elif sort_by == "Average Selectivity ↑":
            return self.avg_selectivity
        else:
            return self.get_display_name()


def create_toml_string_from_config(config: Dict[str, Any]) -> str:
    """Generate a TOML string from the current UI configuration."""
    datagen_relations = []
    for rel in config["relations"]:
        new_rel = {"name": rel["name"], "num_rows": rel["num_rows"], "attributes": []}
        for attr in rel.get("attributes", []):
            attr_to_write = {"name": attr["name"], "distribution": attr["distribution"]}
            if attr.get("dtype") != DEFAULTS["attribute"]["dtype"]:
                attr_to_write["dtype"] = attr["dtype"]
            new_rel["attributes"].append(attr_to_write)
        datagen_relations.append(new_rel)

    base_plans = []
    plan_settings = config["pattern_settings"]
    for pattern in config["patterns"]:
        if pattern == "custom":
            # Handle the custom plan from session state
            custom_plan_data = st.session_state.get("custom_joins", [])
            # Filter out incomplete joins before saving
            valid_joins = [j for j in custom_plan_data if all(j)]
            if valid_joins:
                base_plans.append(
                    {
                        "pattern": "custom",
                        "base_plan": valid_joins,
                        "permutations": plan_settings.get(
                            "pattern_permutations_custom", False
                        ),
                    }
                )
        else:
            # Handle standard patterns
            base_plans.append(
                {
                    "pattern": pattern,
                    "num_plans": plan_settings.get(f"pattern_num_plans_{pattern}", 1),
                    "permutations": plan_settings.get(
                        f"pattern_permutations_{pattern}", False
                    ),
                }
            )

    plangen_config = {
        "enabled": True,
        "visualize": config["enable_visualization"],
        "visualization_format": config["visualization_format"],
        "base_plans": base_plans,
    }

    toml_config = {
        "project": {"name": config["project_name"], "output_dir": "output"},
        "iterations": [
            {
                "name": CONFIG["iteration_name"],
                "datagen": {"enabled": True, "relations": datagen_relations},
                "plangen": plangen_config,
                "analysis": {"enabled": config["enable_analysis"]},
            }
        ],
    }
    return toml.dumps(toml_config)


def update_session_state_from_toml(file_content: bytes) -> None:
    """Parse a TOML file and update the session state to match."""
    try:
        user_config = tomllib.loads(file_content.decode("utf-8"))

        st.session_state.project_name = user_config.get("project", {}).get(
            "name", "Imported Project"
        )
        st.session_state.advanced_mode = True
        iteration = user_config.get("iterations", [{}])[0]

        # Update relations
        relations = iteration.get("datagen", {}).get("relations", [])
        advanced_relations = []
        for rel in relations:
            new_rel = {
                "name": rel["name"],
                "num_rows": rel["num_rows"],
                "attributes": [],
            }
            for attr in rel.get("attributes", []):
                new_attr = {
                    "name": attr["name"],
                    "dtype": attr.get("dtype", "int64"),
                    "distribution": attr.get("distribution", {"type": "uniform"}),
                }
                new_rel["attributes"].append(new_attr)
            advanced_relations.append(new_rel)
        st.session_state.advanced_relations = advanced_relations

        # Update patterns and their settings
        base_plans = iteration.get("plangen", {}).get("base_plans", [])
        loaded_patterns = []
        st.session_state.custom_joins = []
        for bp in base_plans:
            pattern = bp["pattern"]
            if pattern == "custom":
                loaded_patterns.append("custom")
                st.session_state.custom_joins = bp.get("base_plan", [])
                st.session_state["perms_custom"] = bp.get("permutations", False)
            else:
                loaded_patterns.append(pattern)
                st.session_state[f"num_{pattern}"] = bp.get("num_plans", 1)
                st.session_state[f"perms_{pattern}"] = bp.get("permutations", False)

        # This key will be picked up by the multiselect widget on the next rerun
        st.session_state.selected_patterns = loaded_patterns

        plangen_config = iteration.get("plangen", {})
        st.session_state.enable_visualization = plangen_config.get("visualize", True)
        st.session_state.visualization_format = plangen_config.get(
            "visualization_format", "png"
        )

        st.success("Configuration loaded successfully!")

    except Exception as e:
        st.error(f"Failed to parse configuration file: {e}")


def init_session_state() -> None:
    """Initialize all session state with comprehensive defaults"""
    defaults = {
        "analysis_results": None,
        "output_dir": None,
        "running_analysis": False,
        "selected_left_plan": None,
        "selected_right_plan": None,
        "advanced_mode": False,
        "advanced_relations": [],
        "visualization_format": "png",
        "custom_joins": [],
        "selected_patterns": CONFIG["default_patterns"],
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def create_project_config(
    project_name: str,
    relations: List[Dict[str, Any]],
    patterns: List[str],
    enable_analysis: bool,
    enable_visualization: bool,
    visualization_format: str,
    pattern_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """Create project configuration with consistent structure"""
    base_plans = []
    for pattern in patterns:
        if pattern == "custom":
            custom_plan_data = st.session_state.get("custom_joins", [])
            valid_joins = [j for j in custom_plan_data if all(j)]
            if valid_joins:
                base_plans.append(
                    {
                        "pattern": "custom",
                        "base_plan": valid_joins,
                        "permutations": pattern_settings.get(
                            "pattern_permutations_custom", False
                        ),
                    }
                )
        else:
            base_plans.append(
                {
                    "pattern": pattern,
                    "num_plans": pattern_settings.get(
                        f"pattern_num_plans_{pattern}", 1
                    ),
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
                "datagen": {"enabled": True, "relations": relations},
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
                    plans_dir = os.path.join(output_dir, "plans")
                    viz_dir = os.path.join(output_dir, "visualizations")
                    viz_format = plangen_config.get("visualization_format", "png")
                    create_visualizations_for_plans(plans_dir, viz_dir, viz_format)

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


def load_plans(analysis_dir: str) -> List[PlanMetadata]:
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

    # Add plan JSON files
    plans_dir = os.path.join(iteration_dir, "plans")
    if os.path.exists(plans_dir):
        for filename in os.listdir(plans_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(plans_dir, filename)
                files.append((file_path, f"plans/{filename}"))

    # Add visualization files
    viz_dir = os.path.join(iteration_dir, "visualizations")
    if os.path.exists(viz_dir):
        for filename in os.listdir(viz_dir):
            if filename.endswith((".png", ".svg")):
                file_path = os.path.join(viz_dir, filename)
                files.append((file_path, f"visualizations/{filename}"))

    return files
