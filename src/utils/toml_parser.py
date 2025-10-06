import collections.abc
import logging
import tomllib
from typing import Any, Dict


logger = logging.getLogger("djp")

def _get_default_config() -> Dict[str, Any]:
    """Get default project configuration"""

    return {
        "project": {
            "name": "Synthetic Join Data",
            "output_dir": "generated_data",
        },
        "iterations": [
            {
                "name": "default_iteration",
                "datagen": {
                    "enabled": False,
                    "tables": [
                        {
                            "name": "default_table",
                            "num_rows": 10000,
                            "columns": [
                                {
                                    "name": "default_col",
                                    "dtype": "int64",
                                    "distribution": {
                                        "type": "uniform",
                                    },
                                }
                            ],
                        }
                    ],
                },
                "plangen": {
                    "enabled": False,
                    "visualize": False,
                    "visualization_format": "png",
                    "base_plans": [
                        {
                            "pattern": "random",
                            "num_plans": 1,
                            "permutations": False,
                        }
                    ],
                },
                "analysis": {"enabled": False},
            }
        ],
    }


def _merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively overwrites the default configuration with the user configuration

    This enables the user to only specify the things they want to change and utilize the defaults otherwise
    """

    merged = default.copy()
    for key, user_value in user.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(user_value, collections.abc.Mapping)
        ):
            merged[key] = _merge_configs(merged[key], user_value)
        elif (
            key in merged
            and isinstance(merged[key], list)
            and len(merged[key]) > 0
            and isinstance(user_value, list)
        ):
            # Use the first item in the default list as a template for user-provided list items
            template = merged[key][0]
            merged[key] = [_merge_configs(template, item) for item in user_value]
        else:
            merged[key] = user_value
    return merged


def parse_config(config_path: str) -> Dict[str, Any]:
    """Parse TOML config and merge with defaults"""

    default_config = _get_default_config()

    try:
        with open(config_path, "rb") as f:
            user_config = tomllib.load(f)
    except FileNotFoundError:
        logger.debug(f"Warning: Config file not found at '{config_path}'. Using defaults.")
        user_config = {}
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Error decoding TOML file '{config_path}': {e}") from e

    merged_config = _merge_configs(default_config, user_config)

    user_iterations = user_config.get("iterations", [])
    final_iterations = merged_config.get("iterations", [])

    # Special handling for 'enabled' flags to allow easy toggling of sections
    for i, user_iter in enumerate(user_iterations):
        if i < len(final_iterations):
            if "datagen" in user_iter:
                # Default to enabled if the section exists in user config
                final_iterations[i]["datagen"]["enabled"] = user_iter["datagen"].get(
                    "enabled", True
                )
            if "plangen" in user_iter:
                final_iterations[i]["plangen"]["enabled"] = user_iter["plangen"].get(
                    "enabled", True
                )
            if "analysis" in user_iter:
                final_iterations[i]["analysis"]["enabled"] = user_iter["analysis"].get(
                    "enabled", True
                )

    logger.info("Configuration loaded and processed successfully.")
    return merged_config
