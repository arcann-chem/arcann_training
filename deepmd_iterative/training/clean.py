"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/31
"""
# Standard library modules
import logging
import sys
from pathlib import Path

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.filesystem import remove_files_matching_glob
from deepmd_iterative.common.json import load_json_file


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_config_filename: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}."
    )
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Get control path and load the main config (JSON) and the training config (JSON)
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))
    training_config = load_json_file(
        (control_path / f"training_{padded_curr_iter}.json")
    )

    # Check if we can continue
    if not training_config["is_frozen"]:
        logging.error(f"Lock found. Execute first: training check_freeze.")
        logging.error(f"Aborting...")
        return 1
    logging.critical(f"This the cleaning step for the training.")
    logging.critical(f"It should be run after training update_iter.")
    continuing = input(
        f"Do you want to continue?\n['Y' for yes, anything else to abort]\n"
    )
    if continuing == "Y":
        del continuing
    else:
        logging.error(f"Aborting...")
        return 1

    # Delete
    logging.info(f"Deleting DP-Freeze related error files...")
    remove_files_matching_glob(current_path, "**/graph*freeze.out")
    logging.info(f"Deleting DP-Compress related error files...")
    remove_files_matching_glob(current_path, "**/graph*compress.out")
    logging.info(f"Deleting DP-Train related error files...")
    remove_files_matching_glob(current_path, "**/training.out")
    logging.info(f"Deleting SLURM launch files...")
    remove_files_matching_glob(current_path, "**/*.sh")

    # End
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del user_config_filename
    del main_config, training_config
    del curr_iter, padded_curr_iter

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "clean",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_config_filename=sys.argv[3],
        )
    else:
        pass
