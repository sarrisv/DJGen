import os
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui.models import (
    CONFIG,
    PlanMetadata,
    get_best_plans_by_type,
    get_all_data_files,
    create_zip_archive,
)
from src.ui.utils import _render_two_column_layout, _render_standard_input_pair


def _create_plan_charts(plan: PlanMetadata, title_prefix: str):
    """Create plotly charts for a plan with consistent styling"""
    analysis_section = plan.plan_data.get("analysis", {})
    stages = analysis_section.get("stages", [])
    plan_type = plan.plan_data.get("execution_plan", {}).get("type")

    if not stages:
        return None, None

    # Prepare data for charts
    stage_data = []
    for stage in stages:
        stage_data.append(
            {
                "Stage": stage["stage_id"],
                "Output Size": stage["output_size"],
                "Selectivity": stage.get("selectivity", 0),
                "Total Results": stage.get("total_intermediates", 0),
                "Materialized Results": 0
                if plan_type == "nary" and stage != stages[-1]
                else stage.get("total_intermediates", 0),
            }
        )

    df = pd.DataFrame(stage_data)

    # Create intermediates chart
    fig1 = px.bar(
        df,
        x="Stage",
        y="Materialized Results",
        title=f"{title_prefix} - Cumulative Materialized Results",
    )
    fig1.update_layout(height=300)

    # Create selectivity chart
    fig2 = px.line(
        df,
        x="Stage",
        y="Selectivity",
        markers=True,
        title=f"{title_prefix} - Selectivity by Stage",
    )
    fig2.update_layout(height=300)

    return fig1, fig2


def _render_plan_comparison_controls(
    filtered_plans: List[PlanMetadata], sort_by: str, analysis_was_run: bool
) -> Tuple[PlanMetadata, PlanMetadata]:
    """Render plan comparison controls, conditionally showing the 'Best Plans' button."""

    if analysis_was_run:
        best_binary, best_nary = get_best_plans_by_type(filtered_plans, sort_by)
        can_select_best = best_binary is not None and best_nary is not None
        if st.button("Select Best Plans", disabled=not can_select_best, type="secondary"):
            if best_binary and best_nary:
                st.session_state.plan_select_left = best_binary.get_display_name()
                st.session_state.plan_select_right = best_nary.get_display_name()
                st.rerun()

    plan_names = [p.get_display_name() for p in filtered_plans]

    if "plan_select_left" not in st.session_state and plan_names:
        st.session_state.plan_select_left = plan_names[0]
    if "plan_select_right" not in st.session_state and len(plan_names) > 1:
        st.session_state.plan_select_right = plan_names[1]

    def left_plan_selector():
        return st.selectbox("Left Plan", plan_names, key="plan_select_left")

    def right_plan_selector():
        return st.selectbox("Right Plan", plan_names, key="plan_select_right")

    _render_standard_input_pair(left_plan_selector, right_plan_selector)

    left_name = st.session_state.plan_select_left
    right_name = st.session_state.plan_select_right

    left_plan = next((p for p in filtered_plans if p.get_display_name() == left_name), None)
    right_plan = next((p for p in filtered_plans if p.get_display_name() == right_name), None)

    if not left_plan: left_plan = filtered_plans[0]
    if not right_plan: right_plan = filtered_plans[-1]

    return left_plan, right_plan


def render_metrics_comparison(left_plan: PlanMetadata, right_plan: PlanMetadata):
    """Render metrics with standardized 4-column layout"""
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
    """Render charts comparison with consistent layout"""
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
    """Render join plan visualizations side by side, or show an info message."""
    viz_dir = os.path.join(output_dir, CONFIG["iteration_name"], "visualizations")

    # If the directory doesn't exist, it means visualization was disabled.
    if not os.path.exists(viz_dir):
        st.info(
            "Join plan visualizations were not generated. "
            "To view them, enable 'Visualize Join Plans' in the sidebar and generate again."
        )
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
            st.warning(f"Visualization not found: {left_viz_file}")

    # Right plan visualization
    with col2:
        right_viz_file = f"{right_plan.filename.replace('.json', f'.{viz_format}')}"
        right_viz_path = os.path.join(viz_dir, right_viz_file)
        if os.path.exists(right_viz_path):
            st.image(right_viz_path, caption=right_plan.get_display_name())
        else:
            st.warning(f"Visualization not found: {right_viz_file}")


def render_downloads_section(output_dir: str):
    """Render download section for all analysis results"""
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


def _render_performance_metrics_and_charts(
    left_plan: PlanMetadata, right_plan: PlanMetadata
) -> None:
    """Renders the performance-related metrics and charts of the dashboard."""
    render_metrics_comparison(left_plan, right_plan)
    st.divider()
    render_charts_comparison(left_plan, right_plan)


def render_analysis_tab(
    filtered_plans: List[PlanMetadata], sort_by: str, analysis_was_run: bool
):
    """
    Orchestrates the rendering of the 'Analysis' tab.
    Conditionally shows performance metrics if analysis was run.
    """
    st.subheader("Plan Comparison")
    left_plan, right_plan = _render_plan_comparison_controls(
        filtered_plans, sort_by, analysis_was_run
    )

    st.divider()

    # Conditionally render performance metrics and charts
    if analysis_was_run:
        _render_performance_metrics_and_charts(left_plan, right_plan)
    else:
        st.info("Performance analysis was not run. Metrics and charts are unavailable.")

    # Always attempt to render visualizations and downloads
    st.divider()
    viz_format = st.session_state.get("visualization_format", "png")
    render_visualizations(
        left_plan, right_plan, st.session_state.output_dir, viz_format
    )

    st.divider()
    render_downloads_section(st.session_state.output_dir)
