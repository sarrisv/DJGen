import os
import json
from typing import List, Dict, Any
import streamlit as st

from src.ui.models import PlanMetadata, CONFIG


def _render_data_and_join_plan_summary(
    plan: PlanMetadata, output_dir: str, viz_format: str
) -> None:
    """
    Renders a summary of the data schema and base join plan for a selected plan.
    """
    plan_data = plan.plan_data
    catalog = plan_data.get("catalog", {})
    query = plan_data.get("query", {})

    if not catalog or not query:
        st.warning("Could not load catalog or query information.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Schema**")
        for rel_name, rel_details in catalog.items():
            with st.expander(f"Relation: `{rel_name}`"):
                schema_data = [
                    {"Attribute": attr["name"], "Data Type": attr["type"]}
                    for attr in rel_details.get("schema", [])
                ]
                st.table(schema_data)

    with col2:
        st.markdown("**Base Join Plan Graph**")

        # Find and display the pre-rendered base plan image
        base_plan_name = f"{plan.pattern}{plan.index}"
        viz_file = f"{base_plan_name}_base_plan.{viz_format}"
        viz_path = os.path.join(
            output_dir, CONFIG["iteration_name"], "visualizations", viz_file
        )

        if os.path.exists(viz_path):
            st.image(viz_path, caption=f"{base_plan_name} Base Join Plan")
        else:
            st.info(f"Base plan visualization not found: {viz_file}")

    st.markdown("**Generated SQL Query**")
    st.code(query.get("sql", "SQL query not found."), language="sql")


def _render_plan_file_content(plan: PlanMetadata) -> None:
    """
    Renders an expandable section showing the raw plan file content.
    """
    with st.expander("Plan File Content", expanded=False):
        st.markdown(f"**File:** `{plan.filename}`")
        
        # Format the JSON data with proper indentation
        formatted_json = json.dumps(plan.plan_data, indent=2, ensure_ascii=False)
        
        st.code(formatted_json, language="json")


def render_data_tab(filtered_plans: List[PlanMetadata], output_dir: str):
    """
    Orchestrates the rendering of the 'Data' tab.
    """
    st.subheader("Plan Details")
    plan_names = [p.get_display_name() for p in filtered_plans]
    selected_plan_name = st.selectbox(
        "Select a plan to view its schema, base plan, and SQL query",
        plan_names,
        key="data_tab_plan_selector",
    )
    selected_plan = next(
        (p for p in filtered_plans if p.get_display_name() == selected_plan_name),
        None,
    )

    if selected_plan:
        # Pass selected plan, output_dir, and viz_format to the summary renderer
        viz_format = st.session_state.get("visualization_format", "png")
        _render_data_and_join_plan_summary(selected_plan, output_dir, viz_format)
        
        # Add expandable section for plan file content
        _render_plan_file_content(selected_plan)
    else:
        st.warning("Could not find the selected plan.")
