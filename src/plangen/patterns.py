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
    """Get table config by name"""

    return next(t for t in configs if t["name"] == name)


def create_linear_plan(
    table_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Create linear join plan"""

    if len(table_configs) < 2:
        return []

    # Randomize table order to avoid bias
    random.shuffle(table_configs)

    plan = []
    used_attributes_globally = set()
    attributes_per_table = defaultdict(set)
    
    # Create linear chain: T1 ⋈ T2 ⋈ T3 ⋈ ... ⋈ Tn
    for i in range(len(table_configs) - 1):
        config_a = table_configs[i]
        config_b = table_configs[i + 1]
        
        # Select join attribute and add to plan
        attribute = _select_next_attribute(
            config_a, config_b, used_attributes_globally, attributes_per_table
        )
        plan.append((config_a["name"], config_b["name"], attribute))

        # Track attribute usage for future selections
        used_attributes_globally.add(attribute)
        attributes_per_table[config_a["name"]].add(attribute)
        attributes_per_table[config_b["name"]].add(attribute)

    return plan


def create_star_plan(table_configs: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Create star join plan"""

    if len(table_configs) < 2:
        return []

    # Randomize table order to avoid bias
    random.shuffle(table_configs)

    plan = []
    used_attributes_globally = set()
    attributes_per_table = defaultdict(set)

    # Create star pattern: center ⋈ T1, center ⋈ T2, ..., center ⋈ Tn
    hub_config = table_configs[0]
    for spoke_config in table_configs[1:]:
        # Select join attribute and add to plan
        attribute = _select_next_attribute(
            hub_config, spoke_config, used_attributes_globally, attributes_per_table
        )
        plan.append((hub_config["name"], spoke_config["name"], attribute))

        # Track attribute usage for future selections
        used_attributes_globally.add(attribute)
        attributes_per_table[hub_config["name"]].add(attribute)
        attributes_per_table[spoke_config["name"]].add(attribute)

    return plan


def create_cyclic_plan(
    table_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Create cyclic join plan"""

    if len(table_configs) < 2:
        return []

    plan = []
    used_attributes_globally = set()
    attributes_per_table = defaultdict(set)

    # Create linear chain: T1 ⋈ T2 ⋈ T3 ⋈ ... ⋈ Tn
    for i in range(len(table_configs) - 1):
        config_a = table_configs[i]
        config_b = table_configs[i + 1]
        
        # Select join attribute and add to plan
        attribute = _select_next_attribute(
            config_a, config_b, used_attributes_globally, attributes_per_table
        )
        plan.append((config_a["name"], config_b["name"], attribute))

        # Track attribute usage for future selections
        used_attributes_globally.add(attribute)
        attributes_per_table[config_a["name"]].add(attribute)
        attributes_per_table[config_b["name"]].add(attribute)

    # Close the cycle: connect last table back to first (Tn ⋈ T1)
    last_config = table_configs[-1]
    first_config = table_configs[0]
    
    # Select join attribute and add to plan
    attribute = _select_next_attribute(
        last_config, first_config, used_attributes_globally, attributes_per_table
    )
    plan.append((last_config["name"], first_config["name"], attribute))

    return plan


def create_random_plan(
    table_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Create random join plan"""

    if len(table_configs) < 2:
        return []

    plan = []
    used_attributes_globally = set()
    attributes_per_table = defaultdict(set)

    # Start with random table as seed
    random.shuffle(table_configs)
    connected_tables = {table_configs[0]["name"]}

    # Randomly join remaining tables to the growing joined set
    while len(connected_tables) < len(table_configs):
        # Pick random table from already joined set
        table_a_name = random.choice(list(connected_tables))
        config_a = _get_table_config(table_a_name, table_configs)

        # Pick random table from remaining unjoined tables
        potential_partners = [t for t in table_configs if t["name"] not in connected_tables]
        if not potential_partners:
            break
        config_b = random.choice(potential_partners)

        # Select join attribute and add to plan
        attribute = _select_next_attribute(
            config_a, config_b, used_attributes_globally, attributes_per_table
        )
        plan.append((config_a["name"], config_b["name"], attribute))

        # Track attribute usage for future selections
        used_attributes_globally.add(attribute)
        attributes_per_table[config_a["name"]].add(attribute)
        attributes_per_table[config_b["name"]].add(attribute)
        connected_tables.add(config_b["name"])

    return plan
