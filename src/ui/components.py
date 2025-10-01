import os
from typing import List, Dict, Any, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from .models import (
    CONFIG,
    SORT_OPTIONS,
    VALIDATION_RULES,
    PlanMetadata,
    create_project_config,
    run_djp_generator,
    get_best_plans_by_type,
    load_analysis_plans,
    create_zip_archive,
    get_all_data_files,
)


def __render_two_column_layout(left_component, right_component):
    """Standard two-column layout for paired form elements."""
    col1, col2 = st.columns(2)
    with col1:
        left_result = left_component()
    with col2:
        right_result = right_component()
    return left_result, right_result


def __render_distribution_params(
    dist_type: str, key_prefix: str, defaults: Dict = None
) -> Dict:
    """Standardized distribution parameter input with consistent validation."""
    defaults = defaults or {}
    rules = VALIDATION_RULES[dist_type]

    if dist_type == "uniform":

        def left():
            return st.number_input(
                "Min Value",
                value=defaults.get("low", 1),
                min_value=rules["low"]["min"],
                key=f"{key_prefix}_low",
            )

        def right():
            low_val = st.session_state.get(f"{key_prefix}_low", defaults.get("low", 1))
            return st.number_input(
                "Max Value",
                value=defaults.get("high", 1000),
                min_value=max(low_val + 1, rules["high"]["min"]),
                key=f"{key_prefix}_high",
            )

        low, high = _render_two_column_layout(left, right)
        return {"low": low, "high": high}

    elif dist_type == "gaussian":

        def left():
            return st.number_input(
                "Mean",
                value=defaults.get("mean", 100.0),
                min_value=float(rules["mean"]["min"]),
                key=f"{key_prefix}_mean",
            )

        def right():
            return st.number_input(
                "Std Deviation",
                value=defaults.get("std", 20.0),
                min_value=float(rules["std"]["min"]),
                key=f"{key_prefix}_std",
            )

        mean, std = _render_two_column_layout(left, right)
        return {"mean": mean, "std": std}

    elif dist_type == "zipf":
        skew = st.number_input(
            "Skewness",
            value=defaults.get("skew", 1.5),
            min_value=float(rules["skew"]["min"]),
            key=f"{key_prefix}_skew",
        )

        def left():
            return st.number_input(
                "Min Value",
                value=defaults.get("low", 1),
                min_value=rules["low"]["min"],
                key=f"{key_prefix}_low",
            )

        def right():
            low_val = st.session_state.get(f"{key_prefix}_low", defaults.get("low", 1))
            return st.number_input(
                "Max Value",
                value=defaults.get("high", 1000),
                min_value=max(low_val + 1, rules["high"]["min"]),
                key=f"{key_prefix}_high",
            )

        low, high = _render_two_column_layout(left, right)
        return {"skew": skew, "low": low, "high": high}

    elif dist_type == "sequential":
        start = st.number_input(
            "Start Value",
            value=defaults.get("start", 1),
            min_value=rules["start"]["min"],
            key=f"{key_prefix}_start",
        )
        return {"start": start}


def __render_standard_button_pair(
    left_text: str,
    right_text: str,
    left_key: str,
    right_key: str,
    right_disabled: bool = False,
) -> Tuple[bool, bool]:
    """Standard button pair layout with consistent styling."""

    def left():
        return st.button(left_text, use_container_width=True, key=left_key)

    def right():
        return st.button(
            right_text, disabled=right_disabled, use_container_width=True, key=right_key
        )

    return _render_two_column_layout(left, right)


def __render_standard_input_pair(left_component, right_component):
    """Standard input pair"""

    def left():
        return left_component()

    def right():
        return right_component()

    return _render_two_column_layout(left, right)


def _handle_analysis_execution(config: Dict[str, Any]) -> None:
    """Handle analysis execution pipeline and state management."""
    st.session_state.update({"running_analysis": True})

    try:
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
            st.session_state.update(
                {
                    "running_analysis": False,
                    "output_dir": output_dir,
                    "analysis_results": output_dir,
                    "visualization_format": config["visualization_format"],
                }
            )
            st.success("Analysis completed successfully!")
            st.rerun()
        else:
            st.session_state.update({"running_analysis": False})
    except Exception as e:
        st.error(f"Failed to execute analysis: {str(e)}")
        st.session_state.update({"running_analysis": False})


def _render_analysis_in_progress() -> None:
    """Display analysis in progress indicator."""
    st.info("Analysis in progress...")


def _render_welcome_content() -> None:
    """Render the welcome/landing page content."""
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


def _render_plan_filtering(plans: List[PlanMetadata]) -> Tuple[List[PlanMetadata], str]:
    """Render plan filtering controls and return filtered plans with sort option."""
    unique_base_plans = sorted(set(p.base_plan for p in plans))
    unique_types = sorted(set(p.type for p in plans if p.type != "unknown"))

    st.subheader("Plan Selection")
    col1, col2, col3 = st.columns(3)

    with col1:
        base_filter = st.selectbox("Base Plan", ["All"] + unique_base_plans)
    with col2:
        type_filter = st.selectbox("Type", ["All"] + unique_types)
    with col3:
        sort_by = st.selectbox("Sort By", SORT_OPTIONS)

    # Apply filters and sort
    filtered_plans = plans
    if base_filter != "All":
        filtered_plans = [p for p in filtered_plans if p.base_plan == base_filter]
    if type_filter != "All":
        filtered_plans = [p for p in filtered_plans if p.type == type_filter]
    filtered_plans.sort(key=lambda p: p.get_sort_key(sort_by))

    return filtered_plans, sort_by


def _render_plan_comparison_controls(
    filtered_plans: List[PlanMetadata], sort_by: str
) -> Tuple[PlanMetadata, PlanMetadata]:
    """Render plan comparison controls and return selected plans."""
    # Select best plans button
    best_binary, best_nary = get_best_plans_by_type(filtered_plans, sort_by)
    can_select_best = best_binary is not None and best_nary is not None

    if st.button("Select Best Plans", disabled=not can_select_best, type="secondary"):
        if best_binary and best_nary:
            st.session_state.update(
                {
                    "selected_left_plan": best_binary.get_display_name(),
                    "selected_right_plan": best_nary.get_display_name(),
                }
            )
            st.rerun()

    # Plan selection with consistent styling
    st.subheader("Comparison")
    plan_names = [p.get_display_name() for p in filtered_plans]

    # Get current selections
    current_left_idx = 0
    current_right_idx = min(1, len(plan_names) - 1)

    if st.session_state.get("selected_left_plan") in plan_names:
        current_left_idx = plan_names.index(st.session_state.get("selected_left_plan"))
    if st.session_state.get("selected_right_plan") in plan_names:
        current_right_idx = plan_names.index(
            st.session_state.get("selected_right_plan")
        )

    def left():
        return st.selectbox(
            "Left Plan",
            plan_names,
            index=current_left_idx,
            key="plan_select_left",
        )

    def right():
        return st.selectbox(
            "Right Plan",
            plan_names,
            index=current_right_idx,
            key="plan_select_right",
        )

    left_name, right_name = _render_two_column_layout(left, right)

    # Update session state
    st.session_state.update(
        {"selected_left_plan": left_name, "selected_right_plan": right_name}
    )

    left_plan = next(p for p in filtered_plans if p.get_display_name() == left_name)
    right_plan = next(p for p in filtered_plans if p.get_display_name() == right_name)

    return left_plan, right_plan


def _create_plan_charts(plan: PlanMetadata, title_prefix: str):
    """Create plotly charts for a plan with consistent styling."""
    stages = plan.analysis_data.get("stages", [])
    if not stages:
        return None, None

    # Prepare data for charts
    stage_data = []
    for stage in stages:
        stage_data.append(
            {
                "Stage": stage["stage"],
                "Output Size": stage["output_size"],
                "Selectivity": stage.get("selectivity", 0),
                "Total Intermediates": stage.get("total_intermediates", 0),
                "Materialized Intermediates": 0
                if stage["type"] == "n-ary" and stage != stages[-1]
                else stage.get("total_intermediates", 0),
                "Tables": ", ".join(stage["tables"]),
            }
        )

    df = pd.DataFrame(stage_data)

    # Create intermediates chart
    fig1 = px.bar(
        df,
        x="Stage",
        y="Materialized Intermediates",
        hover_data=["Tables"],
        title=f"{title_prefix} - Cumulative Materialized Intermediates",
    )
    fig1.update_layout(height=300)

    # Create selectivity chart
    fig2 = px.line(
        df,
        x="Stage",
        y="Selectivity",
        markers=True,
        hover_data=["Tables"],
        title=f"{title_prefix} - Selectivity by Stage",
    )
    fig2.update_layout(height=300)

    return fig1, fig2


def render_metrics_comparison(left_plan: PlanMetadata, right_plan: PlanMetadata):
    """Render metrics with standardized 4-column layout."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{left_plan.get_display_name()}")
        lcol1, lcol2, lcol3, lcol4 = st.columns(4)
        with lcol1:
            st.metric("Final Size", f"{left_plan.final_output_size:,}")
        with lcol2:
            st.metric("Total Results", f"{left_plan.total_intermediates:,}")
        with lcol3:
            st.metric("Max Size", f"{left_plan.max_intermediate_size:,}")
        with lcol4:
            st.metric("Avg Selectivity", f"{left_plan.avg_selectivity:.4f}")

    with col2:
        st.subheader(f"{right_plan.get_display_name()}")
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)

        with rcol1:
            delta = right_plan.final_output_size - left_plan.final_output_size
            st.metric(
                "Final Size",
                f"{right_plan.final_output_size:,}",
                delta=f"{delta:+,}" if delta != 0 else None,
            )
        with rcol2:
            delta = right_plan.total_intermediates - left_plan.total_intermediates
            st.metric(
                "Total Results",
                f"{right_plan.total_intermediates:,}",
                delta=f"{delta:+,}" if delta != 0 else None,
            )
        with rcol3:
            delta = right_plan.max_intermediate_size - left_plan.max_intermediate_size
            st.metric(
                "Max Size",
                f"{right_plan.max_intermediate_size:,}",
                delta=f"{delta:+,}" if delta != 0 else None,
            )
        with rcol4:
            delta = right_plan.avg_selectivity - left_plan.avg_selectivity
            st.metric(
                "Avg Selectivity",
                f"{right_plan.avg_selectivity:.4f}",
                delta=f"{delta:+.4f}" if abs(delta) >= 0.0001 else None,
            )


def render_charts_comparison(left_plan: PlanMetadata, right_plan: PlanMetadata):
    """Render charts comparison with consistent layout."""
    st.subheader("Performance Charts")

    # Generate charts for both plans
    left_fig1, left_fig2 = _create_plan_charts(left_plan, "Left Plan")
    right_fig1, right_fig2 = _create_plan_charts(right_plan, "Right Plan")

    if left_fig1 and right_fig1:
        # Display charts in tabs
        tab1, tab2 = st.tabs(["Intermediate Results", "Selectivity"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(left_fig1, use_container_width=True)
            with col2:
                st.plotly_chart(right_fig1, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(left_fig2, use_container_width=True)
            with col2:
                st.plotly_chart(right_fig2, use_container_width=True)


def render_visualizations(
    left_plan: PlanMetadata, right_plan: PlanMetadata, output_dir: str, viz_format: str
):
    """Render join plan visualizations side by side."""
    viz_dir = os.path.join(output_dir, CONFIG["iteration_name"], "visualizations")
    if not os.path.exists(viz_dir):
        return

    st.subheader("Join Plan Visualizations")

    col1, col2 = st.columns(2)

    # Left plan visualization
    with col1:
        left_viz_file = f"{left_plan.filename.replace('.json', f'.{viz_format}')}"
        left_viz_path = os.path.join(viz_dir, left_viz_file)
        if os.path.exists(left_viz_path):
            st.image(left_viz_path, caption=left_plan.get_display_name())
        else:
            st.info(f"Visualization not found: {left_viz_file}")

    # Right plan visualization
    with col2:
        right_viz_file = f"{right_plan.filename.replace('.json', f'.{viz_format}')}"
        right_viz_path = os.path.join(viz_dir, right_viz_file)
        if os.path.exists(right_viz_path):
            st.image(right_viz_path, caption=right_plan.get_display_name())
        else:
            st.info(f"Visualization not found: {right_viz_file}")


def render_downloads_section(output_dir: str):
    """Render download section for all analysis results."""
    st.subheader("Downloads")

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
        )
    else:
        st.warning("No data files found")


def _render_comparison_dashboard(
    left_plan: PlanMetadata, right_plan: PlanMetadata
) -> None:
    """Render the complete comparison dashboard with metrics, charts, and visualizations."""
    st.divider()
    render_metrics_comparison(left_plan, right_plan)

    st.divider()
    render_charts_comparison(left_plan, right_plan)

    st.divider()
    viz_format = st.session_state.get("visualization_format", "png")
    render_visualizations(
        left_plan, right_plan, st.session_state.output_dir, viz_format
    )

    st.divider()
    render_downloads_section(st.session_state.output_dir)


def _render_analysis_results() -> None:
    """Render the complete analysis results interface."""
    st.header("Analysis Results")

    analysis_dir = os.path.join(
        st.session_state.output_dir, CONFIG["iteration_name"], "analysis"
    )
    plans = load_analysis_plans(analysis_dir)

    if plans:
        filtered_plans, sort_by = _render_plan_filtering(plans)

        if filtered_plans:
            left_plan, right_plan = _render_plan_comparison_controls(
                filtered_plans, sort_by
            )
            _render_comparison_dashboard(left_plan, right_plan)
        else:
            st.warning("No plans match the current filters.")
    else:
        st.error("No analysis files found.")


def _render_project_settings() -> str:
    """Render project name input and return the value."""
    st.sidebar.subheader("Project")
    return st.sidebar.text_input("Project Name", value="Synthetic Data & Join Plans")


def _render_mode_and_tables() -> Tuple[bool, list]:
    """Render mode selection and table configuration, return mode and tables."""
    st.sidebar.subheader("Configuration Mode")
    advanced_mode = st.sidebar.toggle(
        "Advanced Mode", value=st.session_state.get("advanced_mode", False)
    )
    st.session_state.update({"advanced_mode": advanced_mode})

    # Table configuration
    if advanced_mode:
        tables = render_advanced_mode_config()
    else:
        tables = render_simple_mode_config()

    return advanced_mode, tables


def _render_patterns_and_analysis() -> Tuple[list, dict]:
    """Render pattern configuration and return patterns with settings."""
    return render_pattern_configuration()


def _render_sidebar_summary_and_run(patterns: list, tables: list) -> bool:
    """Render sidebar summary validation and run button, return run state."""
    # Configuration summary
    st.sidebar.divider()
    st.sidebar.subheader("Summary")
    if patterns and tables:
        st.sidebar.success(f"{len(patterns)} pattern(s) selected")
        st.sidebar.info(f"{len(tables)} table(s) configured")
    else:
        if not patterns:
            st.sidebar.warning("Select join patterns")
        if not tables:
            st.sidebar.warning("Configure tables")

    # Run button
    st.sidebar.divider()
    return st.sidebar.button(
        "Run",
        type="primary",
        disabled=not patterns
        or not tables
        or st.session_state.get("running_analysis", False),
        use_container_width=True,
    )


def render_simple_mode_config():
    """Render simple mode configuration with consistent styling."""
    st.sidebar.subheader("Simple Configuration")

    num_tables = st.sidebar.slider(
        "Number of Tables",
        CONFIG["table_limits"]["min"],
        CONFIG["table_limits"]["max"],
        CONFIG["table_limits"]["default"],
    )

    rows = st.sidebar.selectbox("Rows per Table", CONFIG["row_options"], index=2)

    columns = st.sidebar.slider(
        "Columns per Table",
        CONFIG["column_limits"]["min"],
        CONFIG["column_limits"]["max"],
        CONFIG["column_limits"]["default"],
    )

    distribution = st.sidebar.selectbox(
        "Distribution Type", CONFIG["distribution_types"], index=1
    )

    # Consistent distribution parameters
    with st.sidebar:
        params = _render_distribution_params(distribution, "simple_config")

    # Create tables with consistent structure
    tables = []
    for i in range(num_tables):
        table_columns = []
        for j in range(columns):
            table_columns.append(
                {
                    "name": f"attr{j}",
                    "dtype": "int64",
                    "distribution": {"type": distribution, **params},
                }
            )
        tables.append({"name": f"rel{i}", "num_rows": rows, "columns": table_columns})

    return tables


def render_advanced_mode_config():
    """Render advanced mode configuration with standardized components."""
    st.sidebar.subheader("Advanced Configuration")

    if "advanced_tables" not in st.session_state:
        st.session_state.update({"advanced_tables": []})

    # Standardized table management buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        add_clicked = st.button(
            "Add Table", use_container_width=True, key="advanced_add_table"
        )
    with col2:
        remove_clicked = st.button(
            "Remove Table",
            disabled=len(st.session_state["advanced_tables"]) == 0,
            use_container_width=True,
            key="advanced_remove_table",
        )

    if add_clicked:
        st.session_state["advanced_tables"].append(
            {
                "name": f"table_{len(st.session_state['advanced_tables'])}",
                "num_rows": 1000,
                "columns": [
                    {
                        "name": "id",
                        "dtype": "int64",
                        "distribution_type": "sequential",
                        "distribution_params": {},
                    }
                ],
            }
        )
        st.rerun()

    if remove_clicked and st.session_state["advanced_tables"]:
        st.session_state["advanced_tables"].pop()
        st.rerun()

    # Render each table with consistent styling
    for i, table in enumerate(st.session_state["advanced_tables"]):
        with st.sidebar.expander(f"Table: {table['name']}", expanded=False):
            render_table_configuration(i, table)

    return st.session_state["advanced_tables"]


def render_pattern_configuration():
    """Render pattern configuration with consistent styling."""
    st.sidebar.subheader("Join Patterns")

    patterns = st.sidebar.multiselect(
        "Select Patterns", CONFIG["patterns"], default=CONFIG["default_patterns"]
    )

    pattern_settings = {}
    if patterns:
        with st.sidebar.expander("Pattern Settings", expanded=False):
            for pattern in patterns:
                st.write(f"**{pattern.title()}**")

                def left():
                    return st.number_input(
                        f"Number of {pattern} plans",
                        min_value=1,
                        max_value=5,
                        value=1,
                        key=f"pattern_plans_{pattern}",
                    )

                def right():
                    return st.toggle(
                        "Permutations", value=True, key=f"pattern_perms_{pattern}"
                    )

                num_plans, permutations = _render_two_column_layout(left, right)
                pattern_settings[f"pattern_num_plans_{pattern}"] = num_plans
                pattern_settings[f"pattern_permutations_{pattern}"] = permutations

    return patterns, pattern_settings


def render_analysis_options():
    """Render analysis options with consistent styling."""
    st.sidebar.subheader("Analysis Options")

    enable_analysis = True  # Always enabled
    enable_visualization = True  # Always enabled

    viz_format = st.sidebar.selectbox(
        "Visualization Format", CONFIG["viz_formats"], index=0
    )

    return enable_analysis, enable_visualization, viz_format


def _render_table_basics(table_idx: int, table: Dict) -> None:
    """Render table name and row count inputs."""

    def left():
        return st.text_input(
            "Name", value=table["name"], key=f"table_name_t{table_idx}"
        )

    def right():
        return st.selectbox(
            "Rows", CONFIG["row_options"], index=2, key=f"table_rows_t{table_idx}"
        )

    st.write("**Table:**")
    name, rows = _render_standard_input_pair(left, right)
    table.update({"name": name, "num_rows": rows})


def _handle_column_management(table_idx: int, table: Dict) -> None:
    """Handle add/remove column buttons and logic."""
    st.write("**Columns:**")
    add_clicked, remove_clicked = _render_standard_button_pair(
        "Add Column",
        "Remove Column",
        f"table_add_col_t{table_idx}",
        f"table_remove_col_t{table_idx}",
        right_disabled=len(table["columns"]) <= 1,
    )

    if add_clicked:
        table["columns"].append(
            {
                "name": f"col_{len(table['columns'])}",
                "dtype": "int64",
                "distribution_type": "uniform",
                "distribution_params": {},
            }
        )
        st.rerun()

    if remove_clicked and len(table["columns"]) > 1:
        table["columns"].pop()
        st.rerun()


def render_table_configuration(table_idx: int, table: Dict) -> None:
    """Orchestrate table configuration with focused helpers."""
    _render_table_basics(table_idx, table)
    _handle_column_management(table_idx, table)

    # Render each column
    for j, column in enumerate(table["columns"]):
        st.write(f"*Column {j + 1}:*")
        render_column_configuration(table_idx, j, column)


def render_column_configuration(table_idx: int, col_idx: int, column: Dict):
    """Render individual column configuration with consistent components."""

    # Column basic settings
    def left():
        return st.text_input(
            "Name", value=column["name"], key=f"column_name_t{table_idx}_c{col_idx}"
        )

    def right():
        return st.selectbox(
            "Data Type",
            CONFIG["data_types"],
            index=CONFIG["data_types"].index(column.get("dtype", "int64")),
            key=f"column_dtype_t{table_idx}_c{col_idx}",
        )

    name, dtype = _render_standard_input_pair(left, right)

    distribution_type = st.selectbox(
        "Distribution",
        CONFIG["distribution_types"],
        index=CONFIG["distribution_types"].index(
            column.get("distribution_type", "uniform")
        ),
        key=f"column_dist_t{table_idx}_c{col_idx}",
    )

    # Distribution parameters with consistent styling
    key_prefix = f"advanced_column_t{table_idx}_c{col_idx}_{distribution_type}"
    params = _render_distribution_params(
        distribution_type, key_prefix, column.get("distribution_params", {})
    )

    column.update(
        {
            "name": name,
            "dtype": dtype,
            "distribution_type": distribution_type,
            "distribution_params": params,
        }
    )


def render_sidebar() -> Dict[str, Any]:
    """Main sidebar orchestrator with clean separation of concerns."""
    st.sidebar.title("DJP Generator Configuration")

    project_name = _render_project_settings()
    advanced_mode, tables = _render_mode_and_tables()
    patterns, pattern_settings = _render_patterns_and_analysis()
    enable_analysis, enable_visualization, viz_format = render_analysis_options()
    run_clicked = _render_sidebar_summary_and_run(patterns, tables)

    return {
        "project_name": project_name,
        "advanced_mode": advanced_mode,
        "tables": tables,
        "patterns": patterns,
        "pattern_settings": pattern_settings,
        "enable_analysis": enable_analysis,
        "enable_visualization": enable_visualization,
        "visualization_format": viz_format,
        "run_clicked": run_clicked,
    }


def render_main_content(config: Dict[str, Any]) -> None:
    """Render main content based on application state."""
    if config["run_clicked"]:
        _handle_analysis_execution(config)
    elif st.session_state.get("running_analysis"):
        _render_analysis_in_progress()
    elif st.session_state.get("analysis_results"):
        _render_analysis_results()
    else:
        _render_welcome_content()
