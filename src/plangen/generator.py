import json
import logging
import os
import random
from collections import defaultdict, OrderedDict
from itertools import permutations
from typing import List, Dict, Any, Tuple, Optional

from src.plangen import patterns

logger = logging.getLogger("djp")

PATTERN_FUNCTIONS = {
    "linear": patterns.create_linear_plan,
    "star": patterns.create_star_plan,
    "cyclic": patterns.create_cyclic_plan,
    "random": patterns.create_random_plan,
}

BasePlan = List[Tuple[str, str, str]]


def _generate_binary_plan_from_base(base_plan: BasePlan) -> List[Dict[str, Any]]:
    stages = []

    for t1, t2, attr in base_plan:
        stage = {
            "type": "binary",
            "tables": [t1, t2],
            "on_attribute": attr,
        }
        stages.append(stage)

    return stages


def _generate_nary_plan_from_base(base_plan: BasePlan) -> List[Dict[str, Any]]:
    groups = {}
    attribute_order = []

    for t1, t2, attr in base_plan:
        if attr not in groups:
            groups[attr] = []
            attribute_order.append(attr)

        if t1 not in groups[attr]:
            groups[attr].append(t1)
        if t2 not in groups[attr]:
            groups[attr].append(t2)

    stages = []
    for attr in attribute_order:
        tables = groups[attr]
        stage = {
            "type": "n-ary",
            "tables": tables,
            "on_attribute": attr,
        }
        stages.append(stage)

    return stages


def _generate_plan_permutations(
    base_plan: BasePlan, max_permutations: Optional[int] = None
) -> List[BasePlan]:
    all_perms = list(permutations(base_plan))

    if max_permutations is not None and len(all_perms) > max_permutations:
        all_perms = random.sample(all_perms, k=max_permutations)

    return [list(perm) for perm in all_perms]


def generate_join_plans_for_iteration(
    plan_gen_config: Dict[str, Any], data_gen_config: Dict[str, Any], output_dir: str
) -> None:
    plans_output_path = os.path.join(output_dir, "plans")
    os.makedirs(plans_output_path, exist_ok=True)

    table_configs = data_gen_config.get("tables", [])

    for plan_config in plan_gen_config.get("base_plans", []):
        pattern = plan_config["pattern"]
        num_plans = plan_config.get("num_plans", 1)
        permutations = plan_config.get("permutations", False)

        logger.debug(f"\t\tGenerating {num_plans} instances of pattern: {pattern}")

        pattern_func = PATTERN_FUNCTIONS[pattern]

        for i in range(num_plans):
            # Generate a single base plan for this pattern instance
            base_plan = pattern_func(table_configs)

            base_output_path = os.path.join(
                plans_output_path, f"{pattern}{i}_base.json"
            )
            with open(base_output_path, "w") as f:
                json.dump(base_plan, f, indent=4)
            logger.debug(f"\t\t\tBase plan written to {base_output_path}")

            plans = [base_plan]

            # Generate permutations if applicable
            if permutations is not False:
                max_permutations = None if permutations is True else permutations
                logger.debug(
                    f"\t\t\tGenerating {'all' if max_permutations is None else max_permutations} permutations"
                )

                plans = _generate_plan_permutations(base_plan, max_permutations)

            for j, plan in enumerate(plans):
                binary_plan = _generate_binary_plan_from_base(plan)
                binary_output_path = os.path.join(
                    plans_output_path, f"{pattern}{i}_p{j}_binary.json"
                )
                with open(binary_output_path, "w") as f:
                    json.dump(binary_plan, f, indent=4)
                logger.debug(
                    f"\t\t\t\tBinary permutation {j} written to {binary_output_path}"
                )

                nary_plan = _generate_nary_plan_from_base(plan)
                nary_output_path = os.path.join(
                    plans_output_path, f"{pattern}{i}_p{j}_nary.json"
                )
                with open(nary_output_path, "w") as f:
                    json.dump(nary_plan, f, indent=4)
                logger.debug(
                    f"\t\t\t\tN-ary permutation {j} written to {nary_output_path}"
                )
