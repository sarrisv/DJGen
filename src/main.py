import argparse
import logging
import os
import sys
from typing import Dict, Any

from dask.distributed import Client, LocalCluster

from src.utils import parse_config
from src.datagen import generate_data_for_iteration
from src.plangen import generate_join_plans_for_iteration
from src.analysis import generate_analysis_for_iteration
from src.visualization import create_visualizations_for_plans

logger = logging.getLogger("djp")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging level"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    logger.addHandler(ch)


def run_iterations(config: Dict[str, Any]) -> None:
    """Run data generation, planning, and analysis for each iteration"""

    for iter_config in config["iterations"]:
        iter_name = iter_config["name"]
        seed = iter_config["seed"]

        logger.info(64 * "=")
        logger.info(f"ITERATION: {iter_name}")
        output_dir = os.path.join(config["project"]["output_dir"], iter_name)

        datagen_config = iter_config.get("datagen", {})
        if datagen_config.get("enabled", False):
            logger.info("\tGenerating data...")
            generate_data_for_iteration(datagen_config, output_dir, seed=seed)
        else:
            logger.debug("\tDatagen not enabled for this iteration")

        plangen_config = iter_config.get("plangen", {})
        if plangen_config.get("enabled", False):
            logger.info("\tGenerating join plans...")
            generate_join_plans_for_iteration(
                plangen_config, datagen_config, output_dir, seed=seed
            )
        else:
            logger.debug("\tPlangen not enabled for this iteration")

        analysis_config = iter_config.get("analysis", {})
        if analysis_config.get("enabled", False):
            logger.info("\tGenerating analysis...")
            generate_analysis_for_iteration(output_dir)
        else:
            logger.debug("\tAnalysis not enabled for this iteration")

        if plangen_config.get("visualize", False):
            logger.info("\tGenerating visualizations...")
            plans_dir = os.path.join(output_dir, "plans")
            visualizations_dir = os.path.join(output_dir, "visualizations")
            create_visualizations_for_plans(
                plans_dir, visualizations_dir, plangen_config["visualization_format"]
            )
        else:
            logger.debug("\tVisualization not enabled for this iteration\n")

    logger.info(64 * "=")
    logger.info("COMPLETED ALL ITERATIONS")


def main() -> None:
    """Parse command-line arguments and start data generation"""

    parser = argparse.ArgumentParser(
        prog="python -m src.main", description="Generate and analyze synthetic data"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "mode",
        choices=["run", "debug"],
        help="Execution mode. 'run' sets up a Dask cluster, 'debug' does not as it is known to interfere with the pydev debugger",
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the TOML configuration file."
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    config = parse_config(args.config_file)

    if args.mode == "run":
        with LocalCluster() as cluster, Client(cluster) as client:
            logger.debug(f"Dask dashboard: {client.dashboard_link}")
            run_iterations(config)
    else:
        run_iterations(config)


if __name__ == "__main__":
    main()
