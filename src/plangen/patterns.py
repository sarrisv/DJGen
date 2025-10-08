import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Set


def _select_next_attribute(
    config0: Dict[str, Any],
    config1: Dict[str, Any],
    used_attrs: Set[str],
    attrs_per_rel: Dict[str, Set[str]],
) -> str:
    attrs0 = {attr["name"] for attr in config0.get("attributes", [])}
    attrs1 = {attr["name"] for attr in config1.get("attributes", [])}
    common_attrs = attrs0.intersection(attrs1)

    if not common_attrs:
        raise ValueError(
            f"Cannot create a join between '{config0['name']}' and '{config1['name']}'. "
            "No common attribute names found."
        )

    # Priority 1: A random, common attribute that has NOT been used in the plan yet
    new_attrs = list(common_attrs - used_attrs)
    if new_attrs:
        return random.choice(new_attrs)

    # Priority 2: A random, common attribute that has already been joined on by either relation
    reusable_candidates = attrs_per_rel[config0["name"]].union(
        attrs_per_rel[config1["name"]]
    )
    valid_reusable_attrs = list(common_attrs.intersection(reusable_candidates))
    if valid_reusable_attrs:
        return random.choice(valid_reusable_attrs)

    # Priority 3: Fallback to any random common attribute
    return random.choice(list(common_attrs))


def _get_rel_config(name: str, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get relation config by name"""

    return next(rel for rel in configs if rel["name"] == name)


def create_linear_plan(
    rel_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Create linear join plan"""

    if len(rel_configs) < 2:
        return []

    random.shuffle(rel_configs)

    plan = []
    used_attributes_globally = set()
    attrs_per_rel = defaultdict(set)

    # Create linear chain: R1 ⋈ R2 ⋈ R3 ⋈ ... ⋈ Rn
    for i in range(len(rel_configs) - 1):
        config0 = rel_configs[i]
        config1 = rel_configs[i + 1]

        # Select join attribute and add to plan
        attribute = _select_next_attribute(
            config0, config1, used_attributes_globally, attrs_per_rel
        )
        plan.append((config0["name"], config1["name"], attribute))

        # Track attribute usage for future selections
        used_attributes_globally.add(attribute)
        attrs_per_rel[config0["name"]].add(attribute)
        attrs_per_rel[config1["name"]].add(attribute)

    return plan


def create_star_plan(rel_configs: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Create star join plan"""

    if len(rel_configs) < 2:
        return []

    random.shuffle(rel_configs)

    plan = []
    used_attributes_globally = set()
    attrs_per_rel = defaultdict(set)

    # Create star pattern: center ⋈ R1, center ⋈ R2, ..., center ⋈ Rn
    hub_config = rel_configs[0]
    for spoke_config in rel_configs[1:]:
        # Select join attribute and add to plan
        attribute = _select_next_attribute(
            hub_config, spoke_config, used_attributes_globally, attrs_per_rel
        )
        plan.append((hub_config["name"], spoke_config["name"], attribute))

        # Track attribute usage for future selections
        used_attributes_globally.add(attribute)
        attrs_per_rel[hub_config["name"]].add(attribute)
        attrs_per_rel[spoke_config["name"]].add(attribute)

    return plan


def create_cyclic_plan(
    rel_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Create cyclic join plan"""

    if len(rel_configs) < 2:
        return []

    random.shuffle(rel_configs)

    plan = []
    used_attributes_globally = set()
    attrs_per_rel = defaultdict(set)

    # Create linear chain: R1 ⋈ R2 ⋈ R3 ⋈ ... ⋈ Rn
    for i in range(len(rel_configs) - 1):
        config0 = rel_configs[i]
        config1 = rel_configs[i + 1]

        # Select join attribute and add to plan
        attribute = _select_next_attribute(
            config0, config1, used_attributes_globally, attrs_per_rel
        )
        plan.append((config0["name"], config1["name"], attribute))

        # Track attribute usage for future selections
        used_attributes_globally.add(attribute)
        attrs_per_rel[config0["name"]].add(attribute)
        attrs_per_rel[config1["name"]].add(attribute)

    # Close the cycle: connect last relation back to first (Rn ⋈ R1)
    last_config = rel_configs[-1]
    first_config = rel_configs[0]

    # Select join attribute and add to plan
    attribute = _select_next_attribute(
        last_config, first_config, used_attributes_globally, attrs_per_rel
    )
    plan.append((last_config["name"], first_config["name"], attribute))

    return plan


def create_random_plan(
    rel_configs: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Create random join plan"""

    if len(rel_configs) < 2:
        return []

    random.shuffle(rel_configs)

    plan = []
    used_attributes_globally = set()
    attrs_per_rel = defaultdict(set)
    connected_rels = {rel_configs[0]["name"]}

    # Randomly join remaining relations to the growing joined set
    while len(connected_rels) < len(rel_configs):
        # Pick random relation from already joined set
        rel0_name = random.choice(list(connected_rels))
        config0 = _get_rel_config(rel0_name, rel_configs)

        # Pick random relation from remaining unjoined relations
        potential_partners = [
            rel for rel in rel_configs if rel["name"] not in connected_rels
        ]
        if not potential_partners:
            break
        config1 = random.choice(potential_partners)

        # Select join attribute and add to plan
        attribute = _select_next_attribute(
            config0, config1, used_attributes_globally, attrs_per_rel
        )
        plan.append((config0["name"], config1["name"], attribute))

        # Track attribute usage for future selections
        used_attributes_globally.add(attribute)
        attrs_per_rel[config0["name"]].add(attribute)
        attrs_per_rel[config1["name"]].add(attribute)
        connected_rels.add(config1["name"])

    return plan
