import argparse
import logging
import os
import sys
import time
from typing import Dict, Any, List, Tuple

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


def run_iterations(config: Dict[str, Any], timing_enabled: bool = False) -> None:
    """Run data generation, planning, and analysis for each iteration"""

    total_start_time = time.perf_counter() if timing_enabled else None

    for iter_config in config["iterations"]:
        iter_name = iter_config["name"]
        seed = iter_config["seed"]

        # Track timing for this iteration
        phase_times: List[Tuple[str, float]] = []
        iteration_start_time = time.perf_counter() if timing_enabled else None
        current_time = iteration_start_time  # Track the running time point

        logger.info(64 * "=")
        logger.info(f"ITERATION: {iter_name}")
        output_dir = os.path.join(config["project"]["output_dir"], iter_name)

        datagen_config = iter_config.get("datagen", {})
        if datagen_config.get("enabled", False):
            logger.info("\tGenerating data...")
            generate_data_for_iteration(datagen_config, output_dir, seed=seed)
            if timing_enabled and current_time is not None:
                end_time = time.perf_counter()
                phase_times.append(("Data Generation", end_time - current_time))
                current_time = end_time
        else:
            logger.debug("\tDatagen not enabled for this iteration")

        plangen_config = iter_config.get("plangen", {})
        if plangen_config.get("enabled", False):
            logger.info("\tGenerating join plans...")
            generate_join_plans_for_iteration(
                plangen_config, datagen_config, output_dir, seed=seed
            )
            if timing_enabled and current_time is not None:
                end_time = time.perf_counter()
                phase_times.append(("Join Plan Generation", end_time - current_time))
                current_time = end_time
        else:
            logger.debug("\tPlangen not enabled for this iteration")

        analysis_config = iter_config.get("analysis", {})
        if analysis_config.get("enabled", False):
            logger.info("\tGenerating analysis...")
            generate_analysis_for_iteration(output_dir)
            if timing_enabled and current_time is not None:
                end_time = time.perf_counter()
                phase_times.append(("Analysis", end_time - current_time))
                current_time = end_time
        else:
            logger.debug("\tAnalysis not enabled for this iteration")

        if plangen_config.get("visualize", False):
            logger.info("\tGenerating visualizations...")
            plans_dir = os.path.join(output_dir, "plans")
            visualizations_dir = os.path.join(output_dir, "visualizations")
            create_visualizations_for_plans(
                plans_dir, visualizations_dir, plangen_config["visualization_format"]
            )
            if timing_enabled and current_time is not None:
                end_time = time.perf_counter()
                phase_times.append(("Visualization", end_time - current_time))
                current_time = end_time
        else:
            logger.debug("\tVisualization not enabled for this iteration\n")

        # Print timing summary table for this iteration
        if timing_enabled and phase_times:
            total_time = (
                current_time - iteration_start_time
                if iteration_start_time and current_time
                else 0
            )
            logger.info("\nTiming")
            for phase_name, duration in phase_times:
                logger.info(f"\t{phase_name:<25} {duration:>8.2f}s")
            logger.info(f"\t{'Total':<25} {total_time:>8.2f}s")

    logger.info(64 * "=")
    logger.info("COMPLETED ALL ITERATIONS")

    # Print overall timing if enabled
    if timing_enabled and total_start_time is not None:
        total_duration = time.perf_counter() - total_start_time
        logger.info(f"\nOverall Pipeline Time: {total_duration:.2f}s")


def main() -> None:
    """Parse command-line arguments and start data generation"""

    parser = argparse.ArgumentParser(
        prog="python -m src.main", description="Generate and analyze synthetic data"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-t",
        "--timing",
        action="store_true",
        help="Enable timing output for each pipeline phase",
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
            run_iterations(config, timing_enabled=args.timing)
    else:
        run_iterations(config, timing_enabled=args.timing)


if __name__ == "__main__":
    main()
