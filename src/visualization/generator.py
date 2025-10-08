import json
import logging
import os
from typing import List, Dict, Any, Optional

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
    # Iterate through each relations's key list (e.g., ['rel0_attr1', 'rel0_attr2'])
    for key_list in on_attributes:
        # Iterate through each key in the list (e.g., 'rel0_attr1')
        for key in key_list:
            # Split the key by '_' and take the attribute part
            parts = key.split("_")
            if len(parts) > 1:
                unique_attrs.add(parts[1])

    return sorted(list(unique_attrs))


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
        output_size = stage["output_size"]
        selectivity = stage["selectivity"]

        on_attributes = _extract_unique_attrs(stage["on_attributes"])

        join_attrs = ", ".join(on_attributes)
        result_label = f"Result {i + 1}\n({output_size} rows)\n\nJoined On: {join_attrs}\nSelectivity: {selectivity}"
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
        output_size = stage["output_size"]
        selectivity = stage["selectivity"]

        on_attributes = _extract_unique_attrs(stage["on_attributes"])

        cluster_name = f"cluster_{i}"
        fake_node = f"fake_node_{i}"

        # Use join attributes as cluster label
        join_attrs_str = ", ".join(on_attributes)
        attribute_label = f"Stage {i + 1}\n({output_size} prefixes)\n\nJoined On: {join_attrs_str}\nSelectivity: {selectivity}"

        with graph.subgraph(name=cluster_name) as cluster:
            cluster.attr(
                label=attribute_label,
                fontname="Arial Bold",
                fontsize="14",
                fontweight="bold",
                fontcolor=OUTLINE_COLOR,
                style="filled,rounded",
                fillcolor=FILL_COLOR,
                color=OUTLINE_COLOR,
                penwidth="2",
                margin="25",
            )

            for rel in relations:
                rel_node = f"{rel}_s{i}"
                _add_node(cluster, rel_node, rel)

            # Add invisible connection node
            cluster.node(fake_node, style="invis", width="0", height="0")

        fake_nodes.append((fake_node, cluster_name))

    # Connect clusters
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
            arrowsize="1.2",
            arrowhead="vee",
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
    """Create join analysis visualization"""
    with open(plan_path, "r") as f:
        plan = json.load(f)

    if "analysis" not in plan:
        logger.debug(
            f"\t\t\tSkipping visualization for {os.path.basename(plan_path)}: no analysis section found."
        )
        return None

    plan_name = plan["query_id"]
    execution_plan = plan["execution_plan"]
    analysis = plan["analysis"]
    base_relations = {
        name: details["statistics"]["cardinality"]
        for name, details in plan["catalog"].items()
    }

    merged_stages = []
    for i, exec_stage in enumerate(execution_plan["stages"]):
        analysis_stage = analysis["stages"][i]
        merged_stage = {**exec_stage, **analysis_stage}
        merged_stages.append(merged_stage)

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
    for plan_file in sorted(plan_files):
        create_visualization(
            os.path.join(plans_dir, plan_file), visualizations_dir, output_format
        )
