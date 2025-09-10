import argparse
import os

from dask.distributed import Client, LocalCluster

from src.utils import parse_config
from src.datagen import generate_data_for_iteration
from src.plangen import generate_join_plans_for_iteration
from src.analysis import generate_analysis_for_iteration
from src.visualization.generator import create_visualizations_for_analyses


def run_iterations(config):
    """Runs the data generation, planning, and analysis for each iteration in the config"""

    for iter_config in config["iterations"]:
        iter_name = iter_config["name"]
        print(64 * "=")
        print(f"ITERATION: {iter_name}")
        output_dir = os.path.join(config["project"]["output_dir"], iter_name)

        datagen_config = iter_config.get("datagen", {})
        if datagen_config.get("enabled", False):
            print("\tStarting data generation")
            generate_data_for_iteration(datagen_config, output_dir)
            print()
        else:
            print("\tDatagen not enabled for this iteration\n")

        plangen_config = iter_config.get("plangen", {})
        if plangen_config.get("enabled", False):
            print("\tStarting join plan generation")
            generate_join_plans_for_iteration(
                plangen_config, datagen_config, output_dir
            )
            print()
        else:
            print("\tPlangen not enabled for this iteration\n")

        analysis_config = iter_config.get("analysis", {})
        if analysis_config.get("enabled", False):
            print("\tStarting analysis...")
            generate_analysis_for_iteration(output_dir)
            print()
        else:
            print("\tAnalysis not enabled for this iteration\n")

        if plangen_config.get("visualize", False):
            print("\tStarting visualization...")
            analysis_dir = os.path.join(output_dir, "analysis")
            visualizations_dir = os.path.join(output_dir, "visualizations")
            visualization_format = plangen_config.get("visualization_format", "png")
            create_visualizations_for_analyses(
                analysis_dir, visualizations_dir, visualization_format
            )
            print()
        else:
            print("\tVisualization not enabled for this iteration\n")

    print(64 * "=")
    print("COMPLETED ALL ITERATIONS")


def main():
    """Parses command-line arguments and starts the data generation process"""

    parser = argparse.ArgumentParser(description="Generate and analyze synthetic data")
    parser.add_argument(
        "mode",
        choices=["run", "debug"],
        help="Execution mode. 'run' sets up a Dask cluster, 'debug' does not as it is known to interfere with the pydev debugger",
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the TOML configuration file."
    )
    args = parser.parse_args()
    config = parse_config(args.config_file)

    if args.mode == "run":
        with LocalCluster() as cluster, Client(cluster) as client:
            print(f"Dask client dashboard is available at: {client.dashboard_link}")
            run_iterations(config)
    else:  # 'debug' mode
        run_iterations(config)


if __name__ == "__main__":
    main()
