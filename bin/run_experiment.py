#!/usr/bin/env python3
import sys, os
sys.path.append(os.environ["SSO_HOME"])
from src.experiment import Experiment
#
# --------- Main ---------
if __name__ == "__main__":
    # Check if a configuration file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    try:
        # Initialize and run the experiment
        experiment = Experiment(config_file)
        experiment.run()
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


