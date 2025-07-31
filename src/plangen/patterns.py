import random
from typing import List, Tuple, Dict, Any


def _find_common_attribute(config_a: Dict[str, Any], config_b: Dict[str, Any]) -> str:
    """Finds and returns a random common column name between two table configurations"""

    cols_a = {col["name"] for col in config_a.get("columns", [])}
    cols_b = {col["name"] for col in config_b.get("columns", [])}

    common_attributes = list(cols_a.intersection(cols_b))

    if not common_attributes:
        raise ValueError(
            f"Cannot create a join between '{config_a['name']}' and '{config_b['name']}'. "
            "No common column names found."
        )

    return random.choice(common_attributes)


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
    for i in range(len(table_configs) - 1):
        config_a = table_configs[i]
        config_b = table_configs[i + 1]
        attribute = _find_common_attribute(config_a, config_b)
        plan.append((config_a["name"], config_b["name"], attribute))
    return plan


def create_star_plan(table_configs: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Creates a star join plan from a list of table configurations"""

    if len(table_configs) < 2:
        return []

    random.shuffle(table_configs)

    plan = []
    hub_config = table_configs[0]
    for spoke_config in table_configs[1:]:
        attribute = _find_common_attribute(hub_config, spoke_config)
        plan.append((hub_config["name"], spoke_config["name"], attribute))
    return plan


def create_cyclic_plan(
    table_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Creates a cyclic join plan from a list of table configurations"""

    if len(table_configs) < 2:
        return []

    plan = create_linear_plan(table_configs)

    last_config = table_configs[-1]
    first_config = table_configs[0]
    attribute = _find_common_attribute(last_config, first_config)
    plan.append((last_config["name"], first_config["name"], attribute))
    return plan


def create_random_plan(
    table_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Creates a random join plan from a list of table configurations"""

    if len(table_configs) < 2:
        return []

    plan = []
    connected_tables = {table_configs[0]["name"]}

    table_pool = list(table_configs)

    while len(connected_tables) < len(table_configs):
        table_a_name = random.choice(list(connected_tables))
        config_a = _get_table_config(table_a_name, table_configs)

        potential_partners = [
            t for t in table_pool if t["name"] not in connected_tables
        ]
        if not potential_partners:
            break

        config_b = random.choice(potential_partners)

        try:
            attribute = _find_common_attribute(config_a, config_b)
            plan.append((config_a["name"], config_b["name"], attribute))
            connected_tables.add(config_b["name"])
        except ValueError:
            continue

    return plan
