"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/07/14
"""

# Standard library modules
import argparse
import importlib
import logging
import logging.config
from pathlib import Path

# Local imports
from arcann_training.common.logging import setup_logging

# Parsing
parser = argparse.ArgumentParser(description="Deepmd iterative program suite")
parser.add_argument("step_name", type=str, help="Step name")
parser.add_argument("phase_name", type=str, help="Phase name")
parser.add_argument("-v", "--verbose", type=int, default=0, help="verbosity, 0 (default) or 1 (debug)")
parser.add_argument("-i", "--input", type=str, default="input.json", help="name of the input file (with ext)")
parser.add_argument("-c", "--cluster", type=str, default=None, help="name of the fake cluster")

if __name__ == "__main__":
    args = parser.parse_args()

    deepmd_iterative_path: Path = Path(__file__).parent

    # Setup logging
    logging_config = setup_logging(args.verbose)
    logging.config.dictConfig(logging_config)
    arcann_logger = logging.getLogger("ArcaNN")
    del logging_config

    # Step/Phase name
    step_name: str = args.step_name
    phase_name: str = args.phase_name
    submodule_name: str = f"arcann_training.{step_name}.{phase_name}"

    # Input
    input_fn: str = args.input

    # Using a fake cluster
    if args.cluster is not None:
        fake_cluster = args.cluster
    else:
        fake_cluster = None

    del args

    # Start
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"ARCANN TRAINING PROGRAM SUITE")
    arcann_logger.info(f"Launching: {step_name.capitalize()} - {phase_name.capitalize()}")
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"-" * 88)

    steps = ["initialization", "training", "exploration", "labeling", "test"]
    valid_phases = {}
    for step in steps:
        step_path = deepmd_iterative_path / step
        files = [f.stem for f in step_path.iterdir() if f.is_file() and f.suffix == ".py" and f.stem not in ["__init__", "utils"]]
        valid_phases[step] = files

    if step_name not in steps:
        arcann_logger.error(f"Invalid step. Valid steps are: {steps}")
        arcann_logger.error(f"Aborting...")
        exit_code = 1
        exit(exit_code)

    elif phase_name not in valid_phases.get(step_name, []):
        arcann_logger.error(f"Invalid phase for step {step_name}. Valid phases are: {valid_phases[step_name]}")
        arcann_logger.error(f"Aborting...")
        exit_code = 1
        exit(exit_code)

    # Launch the module
    else:
        try:
            submodule = importlib.import_module(submodule_name)
            exit_code = submodule.main(step_name, phase_name, deepmd_iterative_path, fake_cluster, input_fn)
            del submodule, submodule_name
        except Exception as e:
            exit_code = 1

    del deepmd_iterative_path, fake_cluster, input_fn

    # Exit
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"-" * 88)
    if exit_code == 0:
        arcann_logger.info(f"{step_name.capitalize()} - {phase_name.capitalize()} finished")
    else:
        arcann_logger.error(f"{step_name.capitalize()} - {phase_name.capitalize()} encountered an error")
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"-" * 88)

    del exit_code, step_name, phase_name
