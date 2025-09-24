import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Set

def _select_next_attribute(
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        used_attributes_globally: Set[str],
        attributes_per_table: Dict[str, Set[str]],
) -> str:
    cols_a = {col["name"] for col in config_a.get("columns", [])}
    cols_b = {col["name"] for col in config_b.get("columns", [])}
    common_attrs = cols_a.intersection(cols_b)

    if not common_attrs:
        raise ValueError(
            f"Cannot create a join between '{config_a['name']}' and '{config_b['name']}'. "
            "No common column names found."
        )

    # Priority 1: A random, common attribute that has NOT been used in the plan yet
    fresh_attributes = list(common_attrs - used_attributes_globally)
    if fresh_attributes:
        return random.choice(fresh_attributes)

    # Priority 2: A random, common attribute that has already been joined on by either table
    reusable_candidates = attributes_per_table[config_a["name"]].union(
        attributes_per_table[config_b["name"]]
    )
    valid_reusable_attrs = list(common_attrs.intersection(reusable_candidates))
    if valid_reusable_attrs:
        return random.choice(valid_reusable_attrs)

    # Priority 3: Fallback to any random common attribute
    return random.choice(list(common_attrs))


def _get_table_config(name: str, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Retrieves a specific table configuration from a list by its name"""

    return next(t for t in configs if t["name"] == name)


def create_linear_plan(
    table_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Creates a linear join plan from a list of table configurations"""

    if len(table_configs) < 2:
        return []

    random.shuffle(table_configs)

    plan = []
    used_attributes_globally = set()
    attributes_per_table = defaultdict(set)
    for i in range(len(table_configs) - 1):
        config_a = table_configs[i]
        config_b = table_configs[i + 1]
        attribute = _select_next_attribute(
            config_a, config_b, used_attributes_globally, attributes_per_table
        )
        plan.append((config_a["name"], config_b["name"], attribute))

        used_attributes_globally.add(attribute)
        attributes_per_table[config_a["name"]].add(attribute)
        attributes_per_table[config_b["name"]].add(attribute)

    return plan


def create_star_plan(table_configs: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Creates a star join plan from a list of table configurations"""

    if len(table_configs) < 2:
        return []

    random.shuffle(table_configs)

    plan = []
    used_attributes_globally = set()
    attributes_per_table = defaultdict(set)

    hub_config = table_configs[0]
    for spoke_config in table_configs[1:]:
        attribute = _select_next_attribute(
            hub_config, spoke_config, used_attributes_globally, attributes_per_table
        )
        plan.append((hub_config["name"], spoke_config["name"], attribute))

        used_attributes_globally.add(attribute)
        attributes_per_table[hub_config["name"]].add(attribute)
        attributes_per_table[spoke_config["name"]].add(attribute)

    return plan


def create_cyclic_plan(
    table_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Creates a cyclic join plan from a list of table configurations"""

    if len(table_configs) < 2:
        return []

    plan = []
    used_attributes_globally = set()
    attributes_per_table = defaultdict(set)

    for i in range(len(table_configs) - 1):
        config_a = table_configs[i]
        config_b = table_configs[i + 1]
        attribute = _select_next_attribute(
            config_a, config_b, used_attributes_globally, attributes_per_table
        )
        plan.append((config_a["name"], config_b["name"], attribute))

        used_attributes_globally.add(attribute)
        attributes_per_table[config_a["name"]].add(attribute)
        attributes_per_table[config_b["name"]].add(attribute)

    last_config = table_configs[-1]
    first_config = table_configs[0]
    attribute = _select_next_attribute(
        last_config, first_config, used_attributes_globally, attributes_per_table
    )
    plan.append((last_config["name"], first_config["name"], attribute))

    return plan


def create_random_plan(
    table_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Creates a random join plan from a list of table configurations"""

    if len(table_configs) < 2:
        return []

    plan = []
    used_attributes_globally = set()
    attributes_per_table = defaultdict(set)

    random.shuffle(table_configs)
    connected_tables = {table_configs[0]["name"]}

    while len(connected_tables) < len(table_configs):
        table_a_name = random.choice(list(connected_tables))
        config_a = _get_table_config(table_a_name, table_configs)

        potential_partners = [t for t in table_configs if t["name"] not in connected_tables]
        if not potential_partners:
            break
        config_b = random.choice(potential_partners)

        attribute = _select_next_attribute(
            config_a, config_b, used_attributes_globally, attributes_per_table
        )
        plan.append((config_a["name"], config_b["name"], attribute))

        used_attributes_globally.add(attribute)
        attributes_per_table[config_a["name"]].add(attribute)
        attributes_per_table[config_b["name"]].add(attribute)
        connected_tables.add(config_b["name"])

    return plan
