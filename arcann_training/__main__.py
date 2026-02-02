"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2026 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2026/01/31
"""

# Standard library modules
import argparse
import importlib
import logging
import logging.config
import sys
from pathlib import Path

# Local imports
from arcann_training.common.logging import setup_logging


# Parsing
def _discover_steps(base_path: Path):
    steps = ["initialization", "training", "exploration", "labeling", "test"]
    valid_phases = {}
    for step in steps:
        step_path = base_path / step
        files = [
            f.stem
            for f in step_path.iterdir()
            if f.is_file() and f.suffix == ".py" and f.stem not in ["__init__", "utils"]
        ]
        valid_phases[step] = sorted(files)
    return steps, valid_phases


def _build_parser(base_path: Path):
    steps, valid_phases = _discover_steps(base_path)

    parser = argparse.ArgumentParser(description="Deepmd iterative program suite")
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="list available steps and exit",
    )
    parser.add_argument(
        "--list-phases",
        nargs="?",
        const="__all__",
        metavar="STEP",
        help="list phases for a step (or all steps) and exit",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    common.add_argument(
        "-i",
        "--input",
        type=str,
        default="input.json",
        help="name of the input file (with ext)",
    )
    common.add_argument(
        "-c", "--cluster", type=str, default=None, help="name of the fake cluster"
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and show the selected step/phase without running",
    )

    subparsers = parser.add_subparsers(dest="step_name", required=False)
    for step in steps:
        step_parser = subparsers.add_parser(step, help=f"{step} step")
        phase_subparsers = step_parser.add_subparsers(dest="phase_name", required=True)
        for phase in valid_phases.get(step, []):
            phase_subparsers.add_parser(phase, parents=[common], help=f"{phase} phase")

    return parser, steps, valid_phases


def main(argv=None) -> int:
    deepmd_iterative_path: Path = Path(__file__).parent
    parser, steps, valid_phases = _build_parser(deepmd_iterative_path)
    args = parser.parse_args(argv)

    if args.list_steps:
        print("\n".join(steps))
        return 0

    if args.list_phases is not None:
        if args.list_phases == "__all__":
            for step in steps:
                phases = ", ".join(valid_phases.get(step, []))
                print(f"{step}: {phases}")
            return 0
        if args.list_phases not in steps:
            parser.error(f"Invalid step '{args.list_phases}'. Valid steps are: {steps}")
        phases = ", ".join(valid_phases.get(args.list_phases, []))
        print(f"{args.list_phases}: {phases}")
        return 0

    if args.step_name is None or args.phase_name is None:
        parser.print_help()
        return 2

    # Setup logging
    verbose_level = 1 if args.verbose else 0
    logging_config = setup_logging(verbose_level)
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

    # Start
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"ARCANN TRAINING PROGRAM SUITE")
    arcann_logger.info(
        f"Launching: {step_name.capitalize()} - {phase_name.capitalize()}"
    )
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"-" * 88)

    if args.dry_run:
        arcann_logger.info(
            f"Dry run: {step_name.capitalize()} - {phase_name.capitalize()} (input: {input_fn})"
        )
        return 0
    del args

    # Launch the module
    try:
        submodule = importlib.import_module(submodule_name)
        exit_code = submodule.main(
            step_name, phase_name, deepmd_iterative_path, fake_cluster, input_fn
        )
        del submodule, submodule_name
    except Exception:
        arcann_logger.exception("Unhandled error while running the step.")
        exit_code = 1

    del deepmd_iterative_path, fake_cluster, input_fn

    # Exit
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"-" * 88)
    if exit_code == 0:
        arcann_logger.info(
            f"{step_name.capitalize()} - {phase_name.capitalize()} finished"
        )
    else:
        arcann_logger.error(
            f"{step_name.capitalize()} - {phase_name.capitalize()} encountered an error"
        )
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"-" * 88)

    del step_name, phase_name
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
