"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/31
"""

# Standard library modules
import argparse
import importlib
import logging
from pathlib import Path

# Parsing
parser = argparse.ArgumentParser(description="Deepmd iterative program suite")
parser.add_argument("step_name", type=str, help="Step name")
parser.add_argument("phase_name", type=str, help="Phase name")
parser.add_argument("-v", "--verbose", type=int, default=0, help="verbosity, 0 (default) or 1 (debug)")
parser.add_argument("-i", "--input", type=str, default="input.json", help="name of the input file (with ext)")
parser.add_argument("-c", "--cluster", type=str, default=None, help="name of the fake cluster")

if __name__ == "__main__":
    args = parser.parse_args()

    deepmd_iterative_path = Path(__file__).parent

    if int(args.verbose) == 1:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Step/Phase name
    step_name: str = args.step_name
    phase_name: str = args.phase_name
    submodule_name: str = f"deepmd_iterative.{step_name}.{phase_name}"

    # Input
    input_fn: str = args.input

    # Using a fake cluster
    if args.cluster is not None:
        fake_cluster = args.cluster
    else:
        fake_cluster = None

    del args

    # Start
    logging.info(f"-" * 88)
    logging.info(f"-" * 88)
    logging.info(f"DEEPMD ITERATIVE PROGRAM SUITE")
    logging.info(f"Launching: {step_name.capitalize()} - {phase_name.capitalize()}")
    logging.info(f"-" * 88)
    logging.info(f"-" * 88)

    steps = ["initialization", "training", "exploration", "labeling", "test"]
    valid_phases = {}
    for step in steps:
        step_path = deepmd_iterative_path / step
        files = [f.stem for f in step_path.iterdir() if f.is_file() and f.suffix == ".py" and f.stem not in ["__init__", "utils"]]
        valid_phases[step] = files

    if step_name not in steps:
        logging.error(f"Invalid step. Valid steps are: {steps}")
        logging.error(f"Aborting...")
        exit_code = 1
        exit(exit_code)

    elif phase_name not in valid_phases.get(step_name, []):
        logging.error(f"Invalid phase for step {step_name}. Valid phases are: {valid_phases[step_name]}")
        logging.error(f"Aborting...")
        exit_code = 1
        exit(exit_code)

    # Launch the module
    else:
        try:
            submodule = importlib.import_module(submodule_name)
            exit_code = submodule.main(step_name, phase_name, deepmd_iterative_path, fake_cluster, input_fn)
            del submodule, submodule_name
        except Exception as e:
            logging.error(f"{e}")
            exit_code = 1

    del deepmd_iterative_path, fake_cluster, input_fn

    # Exit
    logging.info(f"-" * 88)
    logging.info(f"-" * 88)
    if exit_code == 0:
        logging.info(f"{step_name.capitalize()} - {phase_name.capitalize()} finished")
    else:
        logging.error(f"{step_name.capitalize()} - {phase_name.capitalize()} encountered an error")
    logging.info(f"-" * 88)
    logging.info(f"-" * 88)

    del exit_code, step_name, phase_name
