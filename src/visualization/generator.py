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
    """Create base graph with styling"""
    graph = Digraph(
        f"{plan_name}_{suffix}", comment=f"{suffix.title()} Join Plan: {plan_name}"
    )
    graph.attr(
        rankdir="LR",
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


def _extract_unique_attributes(on_attributes: List[List[str]]) -> List[str]:
    unique_attrs = set()
    # Iterate through each table's key list (e.g., ['rel0_attr1', 'rel0_attr2'])
    for key_list in on_attributes:
        # Iterate through each key in the list (e.g., 'rel0_attr1')
        for key in key_list:
            # Split the key by '_' and take the attribute part
            parts = key.split("_")
            if len(parts) > 1:
                unique_attrs.add(parts[1])

    return sorted(list(unique_attrs))


def generate_binary_join_visualization(
    stages: List[Dict[str, Any]], analysis_name: str, base_tables: Dict[str, int]
) -> Digraph:
    graph = _create_base_graph(analysis_name, "binary")
    graph.attr(ranksep="2.0", nodesep="1.0")

    for table_name, output_size in base_tables.items():
        table_label = f"{table_name}\n({output_size} rows)"
        _add_node(graph, table_name, table_label)

    for i, stage in enumerate(stages):
        table0 = stage["tables"][0]
        table1 = stage["tables"][1]

        result_name = stage["name"]
        output_size = stage["output_size"]
        selectivity = stage["selectivity"]

        on_attributes = _extract_unique_attributes(stage["on_attributes"])

        join_attrs = ", ".join(on_attributes)
        result_label = f"Result {i + 1}\n({output_size} rows)\nJoined On: {join_attrs}\nSelectivity: {selectivity}"
        _add_node(graph, result_name, result_label)

        _add_arrow(graph, table0, result_name)
        _add_arrow(graph, table1, result_name)
    return graph


def generate_nary_join_visualization(
    stages: List[Dict[str, Any]], analysis_name: str, base_tables: Dict[str, int]
) -> Digraph:
    graph = _create_base_graph(analysis_name, "nary")
    graph.attr(ranksep="2.0", nodesep="1.0", compound="true")

    fake_nodes = []

    for i, stage in enumerate(stages):
        tables = stage["contains"]
        output_size = stage["output_size"]
        selectivity = stage["selectivity"]

        on_attributes = _extract_unique_attributes(stage["on_attributes"])

        cluster_name = f"cluster_{i}"
        fake_node = f"fake_node_{i}"

        # Use join attributes as cluster label
        join_attrs_str = ", ".join(on_attributes)
        attribute_label = f"Stage {i + 1}\n({output_size} rows)\nJoined On: {join_attrs_str}\nSelectivity: {selectivity}"

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

            for table in tables:
                table_node = f"{table}_s{i}"
                _add_node(cluster, table_node, table)

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


def generate_graphviz_from_analysis(
    stages: List[Dict[str, Any]], analysis_name: str, base_tables: Dict[str, int]
) -> Digraph:
    """Generate a graph representation of the given join analysis"""
    if not stages:
        return Digraph(analysis_name, comment=f"Empty Join Analysis: {analysis_name}")

    stage_type = stages[0].get("type", "")

    if stage_type == "binary":
        return generate_binary_join_visualization(stages, analysis_name, base_tables)
    elif stage_type == "n-ary":
        return generate_nary_join_visualization(stages, analysis_name, base_tables)
    else:
        # Fallback for unknown types
        graph = _create_base_graph(analysis_name, "unknown")
        tables = {
            table for stage in stages for table in stage.get("tables_captured", [])
        }
        for table in sorted(tables):
            _add_node(graph, table)
        return graph


def create_visualization(
    analysis_file_path: str, output_dir: str, output_format: str = "png"
) -> Optional[str]:
    """Create a visual representation of given join analysis"""
    with open(analysis_file_path, "r") as f:
        analysis_data = json.load(f)

    analysis_name = os.path.splitext(os.path.basename(analysis_file_path))[0]
    # Extract stages from analysis data
    stages = analysis_data["stages"]
    base_tables = analysis_data["base_relations"]
    graph = generate_graphviz_from_analysis(stages, analysis_name, base_tables)

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, analysis_name)

    try:
        graph.render(output_file_path, format=output_format, cleanup=True)
        final_path = f"{output_file_path}.{output_format}"
        logger.debug(f"\t\t\tVisualization written to {final_path}")
        return final_path
    except Exception as e:
        logger.debug(f"\t\t\tError generating visualization: {e}")
        return None


def create_visualizations_for_analyses(
    analysis_dir: str, visualizations_dir: str, output_format: str = "png"
) -> None:
    if not os.path.exists(analysis_dir):
        logger.debug(f"\t\tAnalysis directory does not exist: {analysis_dir}")
        return

    os.makedirs(visualizations_dir, exist_ok=True)
    analysis_files = [
        f for f in os.listdir(analysis_dir) if f.endswith("_analysis.json")
    ]

    if not analysis_files:
        logger.debug(f"\t\tNo analysis files found in {analysis_dir}")
        return

    logger.debug(f"\t\tCreating visualizations for {len(analysis_files)} analyses...")
    for analysis_file in sorted(analysis_files):
        create_visualization(
            os.path.join(analysis_dir, analysis_file), visualizations_dir, output_format
        )
