import json
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple

from graphviz import Digraph

logger = logging.getLogger("djp")

# Color scheme constants
FILL_COLOR = "#DBEEFF"
OUTLINE_COLOR = "#003594"
ARROW_COLOR = "#00205B"
BACKGROUND_COLOR = "#f8f9fa"


def _create_base_graph(plan_name: str, suffix: str) -> Digraph:
    """Create styled base graph"""
    graph = Digraph(
        f"{plan_name}_{suffix}", comment=f"{suffix.title()} Join Plan: {plan_name}"
    )
    graph.attr(
        rankdir="BT" if suffix == "binary" else "LR",
        bgcolor=BACKGROUND_COLOR,
        fontsize="24",
        fontcolor=OUTLINE_COLOR,
        fontname="Arial Bold",
        labelloc="t",
    )
    graph.attr("node", fontsize="12", fontname="Arial Bold")
    graph.attr("edge", fontsize="10", fontname="Arial Bold")

    # Add plan name as graph title
    graph.attr(
        label=f"{plan_name.replace('_', ' ').title()}",
        fontsize="16",
        fontname="Arial Bold",
        labelloc="t",
    )

    return graph


def _add_node(graph: Digraph, node_id: str, label: str = ""):
    if label == "":
        label = node_id

    graph.node(
        node_id,
        label=label,
        shape="box",
        style="filled,rounded",
        fillcolor=FILL_COLOR,
        color=OUTLINE_COLOR,
        penwidth="2",
        fontsize="12",
        fontweight="bold",
        fontcolor=OUTLINE_COLOR,
        margin="0.4,0.3",
    )


def _add_arrow(graph: Digraph, source: str, target: str):
    graph.edge(source, target, color=ARROW_COLOR, penwidth="2", arrowhead="vee")


def _extract_unique_attrs(on_attributes: List[List[str]]) -> List[str]:
    unique_attrs = set()
    for key_list in on_attributes:
        for key in key_list:
            parts = key.split("_")
            if len(parts) > 1:
                unique_attrs.add(parts[1])
    return sorted(list(unique_attrs))


def generate_base_plan_visualization(
    base_plan: List[Tuple[str, str, str]], plan_name: str
) -> Digraph:
    """Creates a Graphviz visualization of a base join plan."""
    graph = _create_base_graph(plan_name, "base_plan")
    graph.attr(
        rankdir="LR",
        labelloc="t",
        label=f"{plan_name.replace('_', ' ').title()} Base Plan",
    )

    relations = set()
    for rel1, rel2, _ in base_plan:
        relations.add(rel1)
        relations.add(rel2)

    for rel in sorted(list(relations)):
        _add_node(graph, rel)

    for rel1, rel2, attr in base_plan:
        graph.edge(rel1, rel2, label=attr)

    return graph


def generate_binary_join_visualization(
    stages: List[Dict[str, Any]], plan_name: str, base_relations: Dict[str, int]
) -> Digraph:
    graph = _create_base_graph(plan_name, "binary")
    graph.attr(ranksep="2.0", nodesep="1.0")

    for rel_name, output_size in base_relations.items():
        rel_label = f"{rel_name}\n({output_size} rows)"
        _add_node(graph, rel_name, rel_label)

    for i, stage in enumerate(stages):
        rel0 = stage["input_relations"][0]
        rel1 = stage["input_relations"][1]
        result_name = stage["name"]
        on_attributes = _extract_unique_attrs(stage["on_attributes"])
        join_attrs = ", ".join(on_attributes)

        # Conditionally get analysis data
        output_size = stage.get("output_size")
        selectivity = stage.get("selectivity")

        # Build label string conditionally
        result_label = f"Result {i + 1}"
        if output_size is not None:
            result_label += f"\n({output_size} rows)"
        result_label += f"\n\nJoined On: {join_attrs}"
        if selectivity is not None:
            result_label += f"\nSelectivity: {selectivity}"

        _add_node(graph, result_name, result_label)
        _add_arrow(graph, rel0, result_name)
        _add_arrow(graph, rel1, result_name)
    return graph


def generate_nary_join_visualization(
    stages: List[Dict[str, Any]], plan_name: str, base_relations: Dict[str, int]
) -> Digraph:
    graph = _create_base_graph(plan_name, "nary")
    graph.attr(ranksep="2.0", nodesep="1.0", compound="true")
    fake_nodes = []

    for i, stage in enumerate(stages):
        relations = stage["base_relations"]
        on_attributes = _extract_unique_attrs(stage["on_attributes"])
        cluster_name = f"cluster_{i}"
        fake_node = f"fake_node_{i}"
        join_attrs_str = ", ".join(on_attributes)

        # Conditionally get analysis data
        output_size = stage.get("output_size")
        selectivity = stage.get("selectivity")

        # Build label string conditionally
        attribute_label = f"Stage {i + 1}"
        if output_size is not None:
            attribute_label += f"\n({output_size} prefixes)"
        attribute_label += f"\n\nJoined On: {join_attrs_str}"
        if selectivity is not None:
            attribute_label += f"\nSelectivity: {selectivity}"

        with graph.subgraph(name=cluster_name) as cluster:
            cluster.attr(
                label=attribute_label,
                fontname="Arial Bold",
                fontsize="14",
                style="filled,rounded",
                fillcolor=FILL_COLOR,
                color=OUTLINE_COLOR,
                penwidth="2",
                margin="25",
            )
            for rel in relations:
                rel_node = f"{rel}_s{i}"
                _add_node(cluster, rel_node, rel)
            cluster.node(fake_node, style="invis", width="0", height="0")
        fake_nodes.append((fake_node, cluster_name))

    for i in range(len(fake_nodes) - 1):
        current_fake, current_cluster = fake_nodes[i]
        next_fake, next_cluster = fake_nodes[i + 1]
        graph.edge(
            current_fake,
            next_fake,
            ltail=current_cluster,
            lhead=next_cluster,
            style="bold",
            color=ARROW_COLOR,
            penwidth="3",
        )
    return graph


def generate_graphviz_from_plan(
    plan_type: str,
    stages: List[Dict[str, Any]],
    plan_name: str,
    base_relations: Dict[str, int],
) -> Digraph:
    """Generate join plan graph"""
    if not stages:
        return Digraph(plan_name, comment=f"Empty Join Plan: {plan_name}")

    if plan_type == "binary":
        return generate_binary_join_visualization(stages, plan_name, base_relations)
    else:
        return generate_nary_join_visualization(stages, plan_name, base_relations)


def create_visualization(
    plan_path: str, output_dir: str, output_format: str = "png"
) -> Optional[str]:
    """Create join analysis visualization, adapting to presence of analysis data."""
    with open(plan_path, "r") as f:
        plan = json.load(f)

    plan_name = plan["query_id"]
    execution_plan = plan["execution_plan"]
    base_relations = {
        name: details["statistics"]["cardinality"]
        for name, details in plan["catalog"].items()
    }

    # Merge execution and analysis stages if analysis exists
    exec_stages = execution_plan.get("stages", [])
    analysis_stages = plan.get("analysis", {}).get("stages", [])

    if analysis_stages:
        merged_stages = []
        for i, exec_stage in enumerate(exec_stages):
            analysis_stage = next(
                (
                    s
                    for s in analysis_stages
                    if s.get("stage_id") == exec_stage.get("stage_id")
                ),
                {},
            )
            merged_stage = {**exec_stage, **analysis_stage}
            merged_stages.append(merged_stage)
    else:
        # If no analysis, just use the execution stages
        merged_stages = exec_stages

    graph = generate_graphviz_from_plan(
        execution_plan["type"], merged_stages, plan_name, base_relations
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, plan_name)

    try:
        graph.render(output_path, format=output_format, cleanup=True)
        final_path = f"{output_path}.{output_format}"
        logger.debug(f"\t\t\tVisualization written to {final_path}")
        return final_path
    except Exception as e:
        logger.debug(f"\t\t\tError generating visualization: {e}")
        return None


def create_visualizations_for_plans(
    plans_dir: str, visualizations_dir: str, output_format: str = "png"
) -> None:
    if not os.path.exists(plans_dir):
        logger.debug(f"\t\tPlans directory does not exist: {plans_dir}")
        return

    os.makedirs(visualizations_dir, exist_ok=True)
    plan_files = [f for f in os.listdir(plans_dir) if f.endswith(".json")]

    if not plan_files:
        logger.debug(f"\t\tNo plan files found in {plans_dir}")
        return

    logger.debug(f"\t\tCreating visualizations for {len(plan_files)} plans...")
    generated_base_plans = set()

    for plan_file in sorted(plan_files):
        # Generate the execution plan visualization (now handles both cases)
        create_visualization(
            os.path.join(plans_dir, plan_file), visualizations_dir, output_format
        )

        # Generate the base plan visualization (once per base plan)
        match = re.match(r"([a-zA-Z]+)(\d+)", plan_file)
        if not match:
            continue

        base_plan_name = match.group(0)

        if base_plan_name not in generated_base_plans:
            try:
                plan_path = os.path.join(plans_dir, plan_file)
                with open(plan_path, "r") as f:
                    plan = json.load(f)

                base_plan_data = plan.get("query", {}).get("base_plan")
                if base_plan_data:
                    graph = generate_base_plan_visualization(
                        base_plan_data, base_plan_name
                    )
                    output_path = os.path.join(
                        visualizations_dir, f"{base_plan_name}_base_plan"
                    )
                    graph.render(output_path, format=output_format, cleanup=True)
                    final_path = f"{output_path}.{output_format}"
                    logger.debug(
                        f"\t\t\tBase plan visualization written to {final_path}"
                    )
                    generated_base_plans.add(base_plan_name)

            except Exception as e:
                logger.debug(
                    f"\t\t\tError generating base plan visualization for {base_plan_name}: {e}"
                )
