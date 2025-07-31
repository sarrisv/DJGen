import os
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from src.plangen import patterns

PATTERN_FUNCTIONS = {
    "linear": patterns.create_linear_plan,
    "star": patterns.create_star_plan,
    "cyclic": patterns.create_cyclic_plan,
    "random": patterns.create_random_plan,
}

BasePlan = List[Tuple[str, str, str]]


def _generate_binary_plan_from_base(base_plan: BasePlan) -> List[Dict[str, Any]]:
    """Generates a list of binary join stages from a base plan"""

    binary_stages = []
    for table_a, table_b, attribute in base_plan:
        stage = {
            "type": "binary",
            "tables": [table_a, table_b],
            "on_attribute": attribute,
        }
        binary_stages.append(stage)
    return binary_stages


def _generate_nary_plan_from_base(base_plan: BasePlan) -> List[Dict[str, Any]]:
    """Generates a list of n-ary join stages from a base plan"""

    groups = defaultdict(set)
    for table_a, table_b, attribute in base_plan:
        groups[attribute].add(table_a)
        groups[attribute].add(table_b)

    nary_stages = []
    for attribute, tables_set in groups.items():
        if len(tables_set) >= 2:
            stage = {
                "type": "n-ary",
                "tables": sorted(list(tables_set)),
                "on_attribute": attribute,
            }
            nary_stages.append(stage)
    return nary_stages


def generate_join_plans_for_iteration(
    plan_gen_config: Dict[str, Any], data_gen_config: Dict[str, Any], output_dir: str
) -> None:
    """Orchestrates the join plan generation for all patterns within a single iteration"""

    plans_output_path = os.path.join(output_dir, "plans")
    os.makedirs(plans_output_path, exist_ok=True)

    table_configs = data_gen_config.get("tables", [])

    for i in range(0, plan_gen_config.get("num_plans", 1)):
        for plan_config in plan_gen_config.get("base_plans", []):
            pattern = plan_config["pattern"]
            print(f"\t\tGenerating plan: {pattern}")

            pattern_func = PATTERN_FUNCTIONS[pattern]
            base_plan = pattern_func(table_configs)

            binary_plan = _generate_binary_plan_from_base(base_plan)
            binary_output_path = os.path.join(
                plans_output_path, f"{pattern}{i}_binary.json"
            )
            with open(binary_output_path, "w") as f:
                json.dump(binary_plan, f, indent=4)
            print(f"\t\t\tBinary plan written to {binary_output_path}")

            nary_plan = _generate_nary_plan_from_base(base_plan)
            nary_output_path = os.path.join(
                plans_output_path, f"{pattern}{i}_nary.json"
            )
            with open(nary_output_path, "w") as f:
                json.dump(nary_plan, f, indent=4)
            print(f"\t\t\tN-ary plan written to {nary_output_path}")
