import json
import logging
import os
import random
from collections import defaultdict, OrderedDict
from itertools import permutations
from typing import List, Dict, Any, Tuple, Optional, Set

from src.plangen import patterns

logger = logging.getLogger("djp")

PATTERN_FUNCTIONS = {
    "linear": patterns.create_linear_plan,
    "star": patterns.create_star_plan,
    "cyclic": patterns.create_cyclic_plan,
    "random": patterns.create_random_plan,
}

BasePlan = List[Tuple[str, str, str]]


def _create_binary_execution_plan(base_plan: BasePlan) -> List[Dict[str, Any]]:
    plan_stages = []
    for rel0, rel1, attr in base_plan:
        stage = {
            "relations": [rel0, rel1],
            "on_attribute": attr,
        }
        plan_stages.append(stage)

    execution_stages: List[Dict[str, Any]] = []
    earliest_stage: Dict[frozenset[str], int] = {}
    latest_stage: Dict[str, int] = {}
    result_counter = 1

    # used to track which relations contain which relations
    rel_composition: Dict[str, Set[str]] = defaultdict(set)
    for stage in plan_stages:
        for rel in stage["relations"]:
            rel_composition[rel].add(rel)

    # used to track join branches so we can join them together at the end if needed
    join_branches = []

    for stage in plan_stages:
        attr = stage["on_attribute"]
        base_rel0 = stage["relations"][0]
        base_rel1 = stage["relations"][1]

        rel_pair = frozenset({base_rel0, base_rel1})
        earliest_stage_idx = earliest_stage.get(rel_pair, None)

        # Case 1: this pair of relations have been joined before, add the attr to the earliest stage they both exist
        if earliest_stage_idx is not None:
            earliest_execution_stage = execution_stages[earliest_stage_idx]

            rel0_key = f"{base_rel0}_{attr}"
            rel1_key = f"{base_rel1}_{attr}"

            # rel0 is / is in relations[0] => rel1 is / is in relations[1]
            if (
                base_rel0
                in rel_composition[earliest_execution_stage["input_relations"][0]]
            ):
                earliest_execution_stage["on_attributes"][0].append(rel0_key)
                earliest_execution_stage["on_attributes"][1].append(rel1_key)

            # rel1 is / is in relations[0] => rel0 is / is in relations[1]
            else:
                earliest_execution_stage["on_attributes"][0].append(rel1_key)
                earliest_execution_stage["on_attributes"][1].append(rel0_key)

            continue

        # Case 2: this pair of relations has not been joined before, create a new execution stage
        rel0_stage = latest_stage.get(base_rel0, None)
        rel1_stage = latest_stage.get(base_rel1, None)

        current_rel0 = (
            base_rel0 if rel0_stage is None else execution_stages[rel0_stage]["name"]
        )
        current_rel1 = (
            base_rel1 if rel1_stage is None else execution_stages[rel1_stage]["name"]
        )

        result_name = f"result_{result_counter}"

        rel0_contains = rel_composition[current_rel0]
        rel1_contains = rel_composition[current_rel1]
        result_contains = rel0_contains.union(rel1_contains)
        rel_composition[result_name].update(result_contains)

        stage_idx = len(execution_stages)
        execution_stages.append(
            {
                "name": result_name,
                "base_relations": [base_rel0, base_rel1],
                "input_relations": [current_rel0, current_rel1],
                "on_attributes": [
                    [f"{base_rel0}_{attr}"],
                    [f"{base_rel1}_{attr}"],
                ],
                "contains": rel_composition[result_name],
            }
        )
        result_counter += 1

        # Update latest stage for all the relations in the new result
        for rel in result_contains:
            latest_stage[rel] = stage_idx

        # Create an earliest stage entry for all new pairs of relations
        # set ensures order does not matter
        for left_rel in rel0_contains:
            for right_rel in rel1_contains:
                new_pair = frozenset({left_rel, right_rel})
                if new_pair not in earliest_stage:
                    earliest_stage[new_pair] = stage_idx

        # Clean up join branches
        if current_rel0 in join_branches:
            join_branches.remove(current_rel0)
        if current_rel1 in join_branches:
            join_branches.remove(current_rel1)
        join_branches.append(result_name)

    # If there are dangling join branches join them via a cartesian product so we end with a single final result
    # Possible if analyzing a bushy join plan that produces disconnected results
    # Example with two stage bushy join plan -- stage 1: rel0 join rel1 -> result_1, stage 2: rel2 join rel3 -> result_2, need a final cartesian product to produce final output result_3
    while len(join_branches) > 1:
        rel0 = join_branches.pop(0)
        rel1 = join_branches.pop(0)

        result_name = f"result_{result_counter}"
        result_contains = rel_composition[rel0].union(rel_composition[rel1])
        rel_composition[result_name].update(result_contains)

        execution_stages.append(
            {
                "name": result_name,
                "base_relations": [rel0, rel1],
                "input_relations": [rel0, rel1],
                "on_attributes": [],
                "contains": rel_composition[result_name],
            }
        )
        result_counter += 1

        join_branches.append(result_name)

    return execution_stages


def _create_nary_execution_plan(base_plan: BasePlan) -> List[Dict[str, Any]]:
    groups = {}
    attribute_order = []

    for rel0, rel1, attr in base_plan:
        if attr not in groups:
            groups[attr] = []
            attribute_order.append(attr)

        if rel0 not in groups[attr]:
            groups[attr].append(rel0)
        if rel1 not in groups[attr]:
            groups[attr].append(rel1)

    plan_stages = []
    for attr in attribute_order:
        rels = groups[attr]
        stage = {
            "relations": rels,
            "on_attribute": attr,
        }
        plan_stages.append(stage)

    execution_stages: List[Dict[str, Any]] = []
    latest_stage: Dict[str, int] = {}
    result_counter = 1

    # used to track which relations contain which relations
    rel_composition: Dict[str, Set[str]] = defaultdict(set)
    for stage in plan_stages:
        for rel in stage["relations"]:
            rel_composition[rel].add(rel)

    # used to track join branches so we can join them together at the end if needed
    join_branches = []

    for stage in plan_stages:
        attr = stage["on_attribute"]
        base_rels = stage["relations"]

        result_name = f"result_{result_counter}"
        current_rels = []
        for rel in base_rels:
            rel = (
                execution_stages[latest_stage[rel]]["name"]
                if rel in latest_stage
                else rel
            )
            current_rels.append(rel)
            rel_composition[result_name].update(rel_composition[rel])
        on_attributes = [[f"{rel}_{attr}"] for rel in base_rels]

        stage_idx = len(execution_stages)
        execution_stages.append(
            {
                "name": result_name,
                "base_relations": base_rels,
                "input_relations": current_rels,
                "on_attributes": on_attributes,
                "contains": rel_composition[result_name],
            }
        )

        join_branches.append(result_name)

        for rel in current_rels:
            if rel in join_branches:
                join_branches.remove(rel)

        for rel in rel_composition[result_name]:
            latest_stage[rel] = stage_idx

        result_counter += 1

    if len(join_branches) > 1:
        result_name = f"result_{result_counter}"
        for rel in join_branches:
            rel_composition[result_name].update(rel_composition[rel])

        execution_stages.append(
            {
                "name": f"result_{result_counter}",
                "base_relations": join_branches,
                "input_relations": join_branches,
                "on_attributes": [],
                "contains": rel_composition[result_name],
            }
        )

    return execution_stages


def _generate_plan_permutations(
    base_plan: BasePlan, max_permutations: Optional[int] = None
) -> List[BasePlan]:
    all_perms = list(permutations(base_plan))

    if max_permutations is not None and len(all_perms) > max_permutations:
        all_perms = random.sample(all_perms, k=max_permutations)

    return [list(perm) for perm in all_perms]


def _generate_sql_from_base_plan(base_plan: BasePlan) -> str:
    if not base_plan:
        return ""

    relations = set()
    where_clauses = []

    for rel0, rel1, attr in base_plan:
        relations.add(rel0)
        relations.add(rel1)
        where_clauses.append(f"{rel0}.{attr} = {rel1}.{attr}")

    from_clause = ", ".join(sorted(list(relations)))
    where_clause = " AND\n  ".join(where_clauses)

    sql = f"SELECT\n  COUNT(*)\nFROM\n  {from_clause}\nWHERE\n  {where_clause};"

    return sql


def generate_join_plans_for_iteration(
    plangen_config: Dict[str, Any],
    datagen_config: Dict[str, Any],
    output_dir: str,
    seed: int | None = None,
) -> None:
    """Generate all join plans for iteration"""

    if seed is not None:
        logger.debug(f"\t\tSeeding Dask with random seed: {seed}")
        random.seed(seed)

    plans_output_path = os.path.join(output_dir, "plans")
    os.makedirs(plans_output_path, exist_ok=True)

    rel_configs = datagen_config.get("relations", [])

    rel_paths = {
        rel["name"]: os.path.join(output_dir, "data", rel["name"])
        for rel in rel_configs
    }

    for plan_config in plangen_config.get("base_plans", []):
        pattern = plan_config["pattern"]
        num_plans = plan_config.get("num_plans", 1)
        permutations = plan_config.get("permutations", False)

        logger.debug(f"\t\tGenerating {num_plans} instances of pattern: {pattern}")

        for i in range(num_plans):
            base_plan = []

            if pattern == "custom":
                base_plan = plan_config.get("base_plan")
                if not base_plan:
                    logger.warning(
                        "\t\t\t'pattern' was 'custom' but no 'base_plan' was provided. Skipping."
                    )
                    continue
            else:
                pattern_func = PATTERN_FUNCTIONS[pattern]
                base_plan = pattern_func(rel_configs)

            plans = [base_plan]

            if permutations is not False:
                max_permutations = None if permutations is True else permutations
                logger.debug(
                    f"\t\t\tGenerating {'all' if max_permutations is None else max_permutations} permutations"
                )

                plans = _generate_plan_permutations(base_plan, max_permutations)

            for j, plan in enumerate(plans):
                catalog = {}
                for rel in rel_configs:
                    name = rel["name"]
                    catalog[name] = {
                        "schema": [
                            {"name": attr["name"], "type": attr.get("dtype", "int64")}
                            for attr in rel.get("attributes", [])
                        ],
                        "statistics": {"cardinality": rel.get("num_rows", 0)},
                        "format": "parquet",
                        "location": rel_paths[name],
                    }

                sql_query = _generate_sql_from_base_plan(plan)
                binary_execution_plan = _create_binary_execution_plan(plan)
                nary_execution_plan = _create_nary_execution_plan(plan)

                for plan_type, exec_stages in [
                    ("binary", binary_execution_plan),
                    ("nary", nary_execution_plan),
                ]:
                    plan_name = f"{pattern}{i}_p{j}_{plan_type}"
                    output_doc = {
                        "plan_id": plan_name,
                        "catalog": catalog,
                        "query": {"base_plan": plan, "sql": sql_query},
                        "filters": [],
                        "execution_plan": {
                            "type": plan_type,
                            "stages": [],
                        },
                    }

                    for idx, stage in enumerate(exec_stages):
                        formatted_stage = {
                            "stage_id": idx,
                            "name": stage["name"],
                            "operation": "join",
                            "base_relations": stage["base_relations"],
                            "input_relations": stage["input_relations"],
                            "on_attributes": stage["on_attributes"],
                        }
                        output_doc["execution_plan"]["stages"].append(formatted_stage)

                    output_filepath = os.path.join(
                        plans_output_path, f"{plan_name}.json"
                    )
                    with open(output_filepath, "w") as f:
                        json.dump(output_doc, f, indent=4)
                    logger.debug(
                        f"\t\t\t {plan_type} join plan written to {output_filepath}"
                    )
