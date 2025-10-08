import os
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st

from src.ui.models import (
    CONFIG,
    SORT_OPTIONS,
    VALIDATION_RULES,
    PlanMetadata,
    create_project_config,
    run_djp_generator,
    load_plans,
    create_toml_string_from_config,
    update_session_state_from_toml,
)
from src.ui.data_tab import render_data_tab
from src.ui.analysis_tab import render_analysis_tab
from src.ui.utils import (
    _render_two_column_layout,
    _render_standard_button_pair,
    _render_standard_input_pair,
)


# ==============================================================================
# 1. High-Level Page Orchestrators
# ==============================================================================


def render_sidebar() -> Dict[str, Any]:
    """Main sidebar orchestrator with clean separation of concerns"""
    st.sidebar.title("DJP Generator Configuration")

    project_name = _render_project_settings()
    advanced_mode, rels = _render_mode_and_relations()

    patterns, pattern_settings = render_pattern_configuration(rels)
    enable_analysis, enable_visualization, viz_format = render_analysis_options()

    current_config = {
        "project_name": project_name,
        "advanced_mode": advanced_mode,
        "relations": rels,
        "patterns": patterns,
        "pattern_settings": pattern_settings,
        "enable_analysis": enable_analysis,
        "enable_visualization": enable_visualization,
        "visualization_format": viz_format,
    }

    _render_config_management_buttons(current_config)
    run_clicked = _render_sidebar_summary_and_run(patterns, rels)

    current_config["run_clicked"] = run_clicked
    return current_config


def render_main_content(config: Dict[str, Any]) -> None:
    """Render main content based on application state"""
    if config["run_clicked"]:
        _handle_analysis_execution(config)
    elif st.session_state.get("running_analysis"):
        _render_analysis_in_progress()
    elif st.session_state.get("analysis_results"):
        _render_analysis_results()
    else:
        _render_welcome_content()


# ==============================================================================
# 2. Main Content Area Renderers
# ==============================================================================


def _render_welcome_content() -> None:
    """Render the welcome/landing page content"""
    st.markdown("""
    # Data & Join Plan Generator

    **Generate and then analyze synthetic datasets and conjunctive join plans**

    ### Quick Start:
    + **Configure** your data generation parameters in the sidebar
    + **Select** join patterns to analyze
    + **Click** "Generate Analysis" to run the complete pipeline
    + **Compare** different join strategies and download results

    Use **Simple Mode** for quick setup or **Advanced Mode** for detailed control.
    """)


def _render_analysis_in_progress() -> None:
    """Display analysis in progress indicator"""
    st.info("Analysis in progress...")


def _render_analysis_results() -> None:
    """Render the complete analysis results interface with a tabbed layout."""
    st.header("Analysis Results")

    plans_dir = os.path.join(
        st.session_state.output_dir, CONFIG["iteration_name"], "plans"
    )
    plans = load_plans(plans_dir)

    if not plans:
        st.error("No analysis files found.")
        return

    analysis_was_run = "analysis" in plans[0].plan_data

    # Render top-level filtering controls that apply to both tabs
    filtered_plans, sort_by = _render_plan_filtering(plans, analysis_was_run)

    if not filtered_plans:
        st.warning("No plans match the current filters.")
        return

    # Create the tabs and delegate rendering to the view modules
    data_tab, analysis_tab = st.tabs(["Data", "Analysis"])

    with data_tab:
        render_data_tab(filtered_plans, st.session_state.output_dir)

    with analysis_tab:
        render_analysis_tab(filtered_plans, sort_by, analysis_was_run)


# ==============================================================================
# 3. Sidebar Renderers
# ==============================================================================


def _render_project_settings() -> str:
    """Render project name input and return the value"""
    st.sidebar.subheader("Project")
    project_name = st.sidebar.text_input(
        "Project Name",
        value=st.session_state.get("project_name", "Synthetic Data & Join Plans"),
        key="project_name_input",
    )
    st.session_state.project_name = project_name
    return project_name


def _render_mode_and_relations() -> Tuple[bool, list]:
    """Render mode selection and relation configuration, return mode and relations"""
    st.sidebar.subheader("Configuration Mode")
    advanced_mode = st.sidebar.toggle(
        "Advanced Mode", value=st.session_state.get("advanced_mode", False)
    )
    st.session_state.update({"advanced_mode": advanced_mode})

    # Relation configuration
    if advanced_mode:
        relations = render_advanced_mode_config()
    else:
        relations = render_simple_mode_config()

    return advanced_mode, relations


def render_simple_mode_config():
    """Render simple mode configuration with consistent styling"""
    st.sidebar.subheader("Simple Configuration")

    num_relations = st.sidebar.slider(
        "Number of Relations",
        CONFIG["rel_limits"]["min"],
        CONFIG["rel_limits"]["max"],
        CONFIG["rel_limits"]["default"],
    )

    rows = st.sidebar.selectbox("Rows per Relation", CONFIG["row_options"], index=2)

    attributes = st.sidebar.slider(
        "Attributes per Relation",
        CONFIG["attribute_limits"]["min"],
        CONFIG["attribute_limits"]["max"],
        CONFIG["attribute_limits"]["default"],
    )

    distribution = st.sidebar.selectbox(
        "Distribution Type", CONFIG["distribution_types"], index=1
    )

    # Consistent distribution parameters
    with st.sidebar:
        params = _render_distribution_params(distribution, "simple_config")

    # Create relations with consistent structure
    relations = []
    for i in range(num_relations):
        rel_attrs = []
        for j in range(attributes):
            rel_attrs.append(
                {
                    "name": f"attr{j}",
                    "dtype": "int64",
                    "distribution": {"type": distribution, **params},
                }
            )
        relations.append({"name": f"rel{i}", "num_rows": rows, "attributes": rel_attrs})

    return relations


def render_advanced_mode_config():
    """Render advanced mode configuration with standardized components"""
    st.sidebar.subheader("Advanced Configuration")

    if "advanced_relations" not in st.session_state:
        st.session_state.update({"advanced_relations": []})

    # Standardized relation management buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        add_clicked = st.button(
            "Add Relation", use_container_width=True, key="advanced_add_relation"
        )
    with col2:
        remove_clicked = st.button(
            "Remove Relation",
            disabled=len(st.session_state["advanced_relations"]) == 0,
            use_container_width=True,
            key="advanced_remove_relation",
        )

    if add_clicked:
        st.session_state["advanced_relations"].append(
            {
                "name": f"rel{len(st.session_state['advanced_relations'])}",
                "num_rows": 1000,
                "attributes": [
                    {
                        "name": "id",
                        "dtype": "int64",
                        "distribution": {"type": "sequential", "start": 1},
                    }
                ],
            }
        )
        st.rerun()

    if remove_clicked and st.session_state["advanced_relations"]:
        st.session_state["advanced_relations"].pop()
        st.rerun()

    # Render each relation with consistent styling
    for i, rel in enumerate(st.session_state["advanced_relations"]):
        with st.sidebar.expander(f"Relation: {rel['name']}", expanded=False):
            render_relation_configuration(i, rel)

    return st.session_state["advanced_relations"]


def render_relation_configuration(rel_idx: int, rel: Dict[str, Any]) -> None:
    """Orchestrate relation configuration with focused helpers"""
    _render_relation_basics(rel_idx, rel)
    _handle_attribute_management(rel_idx, rel)

    # Render each attribute
    for j, attr in enumerate(rel["attributes"]):
        st.write(f"*Attribute {j + 1}:*")
        render_attribute_configuration(rel_idx, j, attr)


def render_attribute_configuration(
    rel_idx: int, attr_idx: int, attr: Dict[str, Any]
) -> None:
    """Render individual attribute configuration with consistent components"""
    # Attribute basic settings
    name, dtype = _render_standard_input_pair(
        lambda: st.text_input(
            "Name", value=attr["name"], key=f"attribute_name_r{rel_idx}_a{attr_idx}"
        ),
        lambda: st.selectbox(
            "Data Type",
            CONFIG["data_types"],
            index=CONFIG["data_types"].index(attr.get("dtype", "int64")),
            key=f"attribute_dtype_r{rel_idx}_a{attr_idx}",
        ),
    )

    current_distribution = attr.get("distribution", {"type": "uniform"})
    distribution_type = st.selectbox(
        "Distribution",
        CONFIG["distribution_types"],
        index=CONFIG["distribution_types"].index(
            current_distribution.get("type", "uniform")
        ),
        key=f"attribute_dist_r{rel_idx}_a{attr_idx}",
    )

    key_prefix = f"advanced_attribute_r{rel_idx}_a{attr_idx}_{distribution_type}"
    params = _render_distribution_params(
        distribution_type, key_prefix, current_distribution
    )

    attr.update(
        {
            "name": name,
            "dtype": dtype,
            "distribution": {"type": distribution_type, **params},
        }
    )


def render_pattern_configuration(relations: list) -> Tuple[list, dict]:
    """Render pattern configuration, including the custom plan editor."""
    st.sidebar.subheader("Join Patterns")

    all_patterns = CONFIG["patterns"] + ["custom"]

    # Use a persistent session state key for the multiselect default
    default_patterns = st.session_state.get(
        "selected_patterns", CONFIG["default_patterns"]
    )

    patterns = st.sidebar.multiselect(
        "Select Patterns", all_patterns, default=default_patterns
    )
    # Immediately update the session state with the user's selection
    st.session_state.selected_patterns = patterns

    pattern_settings = {}

    # Render settings for standard patterns
    standard_patterns = [p for p in patterns if p != "custom"]
    if standard_patterns:
        with st.sidebar.expander("Standard Pattern Settings", expanded=False):
            for pattern in standard_patterns:
                st.write(f"**{pattern.title()}**")
                num_plans, perms = _render_two_column_layout(
                    lambda: st.number_input(
                        f"Number of plans",
                        min_value=1,
                        max_value=5,
                        value=st.session_state.get(f"num_{pattern}", 1),
                        key=f"num_{pattern}",
                    ),
                    lambda: st.toggle(
                        "Permutations",
                        value=st.session_state.get(f"perms_{pattern}", True),
                        key=f"perms_{pattern}",
                    ),
                )
                pattern_settings[f"pattern_num_plans_{pattern}"] = num_plans
                pattern_settings[f"pattern_permutations_{pattern}"] = perms

    # Render the custom plan editor if "custom" is selected
    if "custom" in patterns:
        custom_settings = _render_custom_plan_editor(relations)
        pattern_settings.update(custom_settings)

    return patterns, pattern_settings


def _render_custom_plan_editor(relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Render the UI for creating a custom join plan."""
    settings = {}
    with st.sidebar.expander("Custom Plan Editor", expanded=True):
        if "custom_joins" not in st.session_state:
            st.session_state.custom_joins = []

        # Add/Remove buttons
        add, remove = _render_two_column_layout(
            lambda: st.button("Add Join", use_container_width=True),
            lambda: st.button(
                "Remove Last Join",
                use_container_width=True,
                disabled=not st.session_state.custom_joins,
            ),
        )
        if add:
            st.session_state.custom_joins.append(["", "", ""])
            st.rerun()
        if remove:
            st.session_state.custom_joins.pop()
            st.rerun()

        # Render each join row
        for i, join in enumerate(st.session_state.custom_joins):
            st.markdown(f"**Join {i + 1}**")
            rel_names = [r["name"] for r in relations]

            # Use columns for layout
            col1, col2, col3 = st.columns(3)
            with col1:
                join[0] = st.selectbox(
                    "Rel 1",
                    rel_names,
                    index=rel_names.index(join[0]) if join[0] in rel_names else 0,
                    key=f"cust_rel1_{i}",
                )
            with col2:
                join[1] = st.selectbox(
                    "Rel 2",
                    rel_names,
                    index=rel_names.index(join[1]) if join[1] in rel_names else 0,
                    key=f"cust_rel2_{i}",
                )

            common_attrs = _get_common_attributes(join[0], join[1], relations)
            with col3:
                join[2] = st.selectbox(
                    "On Attr",
                    common_attrs,
                    index=common_attrs.index(join[2]) if join[2] in common_attrs else 0,
                    key=f"cust_attr_{i}",
                )

        # Permutation settings for the custom plan
        st.markdown("**Permutations**")
        perms = st.toggle(
            "Permutations",
            value=st.session_state.get("perms_custom", False),
            key="perms_custom",
        )
        settings["pattern_permutations_custom"] = perms

    return settings


def _get_common_attributes(
    rel1_name: str, rel2_name: str, relations: List[Dict[str, Any]]
) -> List[str]:
    """Helper to find common attributes between two relations."""
    if not rel1_name or not rel2_name or not relations:
        return []

    try:
        rel1 = next((r for r in relations if r["name"] == rel1_name), None)
        rel2 = next((r for r in relations if r["name"] == rel2_name), None)
        if not rel1 or not rel2:
            return []

        rel1_attrs = {attr["name"] for attr in rel1.get("attributes", [])}
        rel2_attrs = {attr["name"] for attr in rel2.get("attributes", [])}

        return sorted(list(rel1_attrs.intersection(rel2_attrs)))
    except (StopIteration, KeyError):
        return []


def render_analysis_options():
    """Render analysis options with consistent styling"""
    st.sidebar.subheader("Analysis & Visualization")

    enable_analysis = st.sidebar.toggle("Run Performance Analysis", value=True)
    enable_visualization = st.sidebar.toggle("Visualize Join Plans", value=True)

    if not enable_analysis and enable_visualization:
        st.sidebar.info(
            "Visualizations will not include performance metrics if analysis is disabled."
        )

    viz_format = st.sidebar.selectbox(
        "Visualization Format", CONFIG["viz_formats"], index=0
    )

    return enable_analysis, enable_visualization, viz_format


def _render_config_management_buttons(config: Dict[str, Any]) -> None:
    """Render config upload and download buttons."""
    st.sidebar.subheader("Configuration")

    # Upload Button
    st.sidebar.file_uploader(
        "Upload Config",
        type=["toml"],
        accept_multiple_files=False,
        key="config_uploader",
        on_change=_handle_config_upload,
    )

    # Download Button
    toml_string = create_toml_string_from_config(config)
    st.sidebar.download_button(
        label="Download Config",
        data=toml_string,
        file_name="config.toml",
        mime="text/toml",
        use_container_width=True,
    )


def _render_sidebar_summary_and_run(patterns: list, relations: list) -> bool:
    """Render sidebar summary validation and run button, return run state"""
    # Configuration summary
    st.sidebar.divider()
    st.sidebar.subheader("Summary")
    if patterns and relations:
        st.sidebar.success(f"{len(patterns)} pattern(s) selected")
        st.sidebar.info(f"{len(relations)} relation(s) configured")
    else:
        if not patterns:
            st.sidebar.warning("Select join patterns")
        if not relations:
            st.sidebar.warning("Configure relations")

    # Run button
    st.sidebar.divider()
    return st.sidebar.button(
        "Generate",
        type="primary",
        disabled=not patterns
        or not relations
        or st.session_state.get("running_analysis", False),
        use_container_width=True,
    )


# ==============================================================================
# 4. Component-Specific Helpers
# ==============================================================================


def _render_plan_filtering(
    plans: List[PlanMetadata], analysis_was_run: bool
) -> Tuple[List[PlanMetadata], str]:
    """Render plan filtering controls, conditionally showing sort options."""
    unique_base_plans = sorted(set(p.base_plan for p in plans))
    unique_types = sorted(set(p.type for p in plans if p.type != "unknown"))
    sort_by = "Name (alphabetical)"

    st.subheader("Filter Plans")

    # Adjust layout based on whether analysis was run
    num_cols = 3 if analysis_was_run else 2
    cols = st.columns(num_cols)

    with cols[0]:
        base_filter = st.selectbox("Base Plan", ["All"] + unique_base_plans)
    with cols[1]:
        type_filter = st.selectbox("Type", ["All"] + unique_types)
    if analysis_was_run:
        with cols[2]:
            sort_by = st.selectbox("Sort By", SORT_OPTIONS)

    # Apply filters
    filtered_plans = plans
    if base_filter != "All":
        filtered_plans = [p for p in filtered_plans if p.base_plan == base_filter]
    if type_filter != "All":
        filtered_plans = [p for p in filtered_plans if p.type == type_filter]

    # Apply sort
    filtered_plans.sort(key=lambda p: p.get_sort_key(sort_by))

    return filtered_plans, sort_by


def _render_distribution_params(
    dist_type: str, key_prefix: str, defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Standardized distribution parameter input with consistent validation"""
    defaults = defaults or {}
    rules = VALIDATION_RULES.get(dist_type, {})

    if dist_type == "uniform":

        def left():
            return st.number_input(
                "Min Value",
                value=defaults.get("low", 1),
                min_value=rules.get("low", {}).get("min", 0),
                key=f"{key_prefix}_low",
            )

        def right():
            low_val = st.session_state.get(f"{key_prefix}_low", defaults.get("low", 1))
            return st.number_input(
                "Max Value",
                value=defaults.get("high", 1000),
                min_value=max(low_val + 1, rules.get("high", {}).get("min", 1)),
                key=f"{key_prefix}_high",
            )

        low, high = _render_two_column_layout(left, right)
        return {"low": low, "high": high}

    elif dist_type == "gaussian":

        def left():
            return st.number_input(
                "Mean",
                value=defaults.get("mean", 100.0),
                min_value=float(rules.get("mean", {}).get("min", 0.0)),
                key=f"{key_prefix}_mean",
            )

        def right():
            return st.number_input(
                "Std Deviation",
                value=defaults.get("std", 20.0),
                min_value=float(rules.get("std", {}).get("min", 0.1)),
                key=f"{key_prefix}_std",
            )

        mean, std = _render_two_column_layout(left, right)
        return {"mean": mean, "std": std}

    elif dist_type == "zipf":
        skew = st.number_input(
            "Skewness",
            value=defaults.get("skew", 1.5),
            min_value=float(rules.get("skew", {}).get("min", 1.0)),
            key=f"{key_prefix}_skew",
        )

        def left():
            return st.number_input(
                "Min Value",
                value=defaults.get("low", 1),
                min_value=rules.get("low", {}).get("min", 0),
                key=f"{key_prefix}_low",
            )

        def right():
            low_val = st.session_state.get(f"{key_prefix}_low", defaults.get("low", 1))
            return st.number_input(
                "Max Value",
                value=defaults.get("high", 1000),
                min_value=max(low_val + 1, rules.get("high", {}).get("min", 1)),
                key=f"{key_prefix}_high",
            )

        low, high = _render_two_column_layout(left, right)
        return {"skew": skew, "low": low, "high": high}

    elif dist_type == "sequential":
        start = st.number_input(
            "Start Value",
            value=defaults.get("start", 1),
            min_value=rules.get("start", {}).get("min", 0),
            key=f"{key_prefix}_start",
        )
        return {"start": start}


def _render_relation_basics(rel_idx: int, rel: Dict[str, Any]) -> None:
    """Render relation name and row count inputs"""
    name, rows = _render_standard_input_pair(
        lambda: st.text_input("Name", value=rel["name"], key=f"rel_name_r{rel_idx}"),
        lambda: st.selectbox(
            "Rows",
            CONFIG["row_options"],
            index=CONFIG["row_options"].index(rel.get("num_rows", 1000)),
            key=f"rel_rows_r{rel_idx}",
        ),
    )
    rel.update({"name": name, "num_rows": rows})


def _handle_attribute_management(rel_idx: int, rel: Dict[str, Any]) -> None:
    """Handle add/remove attribute buttons and logic"""
    st.write("**Attributes:**")
    add_clicked, remove_clicked = _render_standard_button_pair(
        "Add Attribute",
        "Remove Attribute",
        f"rel_add_attr_t{rel_idx}",
        f"rel_remove_attr_t{rel_idx}",
        right_disabled=len(rel["attributes"]) <= 1,
    )

    if add_clicked:
        rel["attributes"].append(
            {
                "name": f"attr{len(rel['attributes'])}",
                "dtype": "int64",
                "distribution": {"type": "uniform"},
            }
        )
        st.rerun()

    if remove_clicked and len(rel["attributes"]) > 1:
        rel["attributes"].pop()
        st.rerun()


# ==============================================================================
# 6. Callback and Handler Functions
# ==============================================================================


def _handle_config_upload():
    """Callback to process the uploaded TOML file from session_state."""
    uploaded_file = st.session_state.get("config_uploader")
    if uploaded_file is not None:
        file_content = uploaded_file.getvalue()
        update_session_state_from_toml(file_content)


def _handle_analysis_execution(config: Dict[str, Any]) -> None:
    """Handle analysis execution pipeline and state management"""
    st.session_state.update({"running_analysis": True})

    try:
        config_dict = create_project_config(
            config["project_name"],
            config["relations"],
            config["patterns"],
            config["enable_analysis"],
            config["enable_visualization"],
            config["visualization_format"],
            config["pattern_settings"],
        )

        output_dir = run_djp_generator(config_dict)

        if output_dir:
            st.session_state.update(
                {
                    "running_analysis": False,
                    "output_dir": output_dir,
                    "analysis_results": output_dir,
                    "visualization_format": config["visualization_format"],
                }
            )
            st.success("Generation completed successfully!")
            st.rerun()
        else:
            st.session_state.update({"running_analysis": False})
    except Exception as e:
        st.error(f"Failed to execute analysis: {str(e)}")
        st.session_state.update({"running_analysis": False})
