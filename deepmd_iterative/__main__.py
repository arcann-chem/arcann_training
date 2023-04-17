"""
Created: 2023/01/01
Last modified: 2023/04/17
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
parser.add_argument(
    "-v",
    "--verbose",
    type=int,
    default=0,
    help="verbosity, 0 (default) or 1 (debug)",
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="input.json",
    help="name of the input file (with ext)",
)

parser.add_argument(
    "-c",
    "--cluster",
    type=str,
    default=None,
    help="name of the fake cluster",
)

if __name__ == "__main__":
    args = parser.parse_args()

    deepmd_iterative_path = Path(__file__).parent

    if int(args.verbose) == 1:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
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

    del args, parser

    # Start
    logging.info(f"-" * 88)
    logging.info(f"-" * 88)
    logging.info(f"DEEPMD ITERATIVE PROGRAM SUITE")
    logging.info(f"Launching: {step_name.capitalize()} - {phase_name.capitalize()}")
    logging.info(f"-" * 88)
    logging.info(f"-" * 88)

    # Launch the module
    try:
        submodule = importlib.import_module(submodule_name)
        exit_code = submodule.main(
            step_name, phase_name, deepmd_iterative_path, fake_cluster, input_fn
        )
        del submodule, submodule_name
    except (ModuleNotFoundError) as e:
        exit_code = 1
        logging.error(f"Step/Phase: '{submodule_name.split('.')[-2]} / {submodule_name.split('.')[-1]}' are not a valid combination.")
        logging.error(f"Aborting...")
        logging.error(f"{e}")
    except Exception as e:
        exit_code = 1

    del deepmd_iterative_path, fake_cluster, input_fn

    # Exit
    logging.info(f"-" * 88)
    logging.info(f"-" * 88)
    if exit_code == 0:
        logging.info(f"{step_name.capitalize()} - {phase_name.capitalize()} finished")
    else:
        logging.error(
            f"{step_name.capitalize()} - {phase_name.capitalize()} encountered an error"
        )
    logging.info(f"-" * 88)
    logging.info(f"-" * 88)

    del exit_code, step_name, phase_name
