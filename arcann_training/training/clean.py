"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15
"""

# Standard library modules
import logging
import sys
from pathlib import Path

# Local imports
from arcann_training.common.check import validate_step_folder
from arcann_training.common.filesystem import (
    remove_files_matching_glob,
    remove_tree,
    remove_all_symlink,
)
from arcann_training.common.json import load_json_file


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_config_filename: str = "input.json",
):
    # Get the logger
    arcann_logger = logging.getLogger("ArcaNN")

    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    arcann_logger.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}."
    )
    arcann_logger.debug(f"Current path :{current_path}")
    arcann_logger.debug(f"Training path: {training_path}")
    arcann_logger.debug(f"Program path: {deepmd_iterative_path}")
    arcann_logger.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Get control path and load the main config (JSON) and the training config (JSON)
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    training_config = load_json_file(
        (control_path / f"training_{padded_curr_iter}.json")
    )

    # Check if we can continue
    if not training_config["is_incremented"]:
        arcann_logger.error(f"Lock found. Please execute 'training increment' first.")
        arcann_logger.error(f"Aborting...")
        return 1
    arcann_logger.critical(f"This is the cleaning step for training step.")
    arcann_logger.critical(f"It should be run after training increment phase.")
    arcann_logger.critical(f"This is will delete:")
    arcann_logger.critical(
        f"symbolic links, 'job_*.sh', 'training.out', 'graph*freeze.out', 'graph*compress.out', 'checkpoint.*', 'input_v2_compat.json', 'DeepMD_*'"
    )
    arcann_logger.critical(f"'model-compression' folders")
    arcann_logger.critical(
        f"'*.pb' models files (they are saved in the '{current_path.parent / 'NNP'}' root folder)"
    )
    arcann_logger.critical(
        f"'data' folder (it is saved in the '{current_path.parent / 'data'}' root folder)"
    )
    arcann_logger.critical(f"in the folder: '{current_path}' and all subdirectories.")
    continuing = input(
        f"Do you want to continue? [Enter 'Y' for yes, or any other key to abort]: "
    )
    if continuing == "Y":
        del continuing
    else:
        arcann_logger.error(f"Aborting...")
        return 0

    # Delete
    arcann_logger.info("Deleting symbolic links...")
    remove_all_symlink(current_path)
    arcann_logger.info("Deleting job files...")
    remove_files_matching_glob(current_path, "**/job_*.sh")
    arcann_logger.info(f"Deleting training unwanted output file..")
    remove_files_matching_glob(current_path, "**/training.out")
    arcann_logger.info(f"Deleting freezing unwanted output files...")
    remove_files_matching_glob(current_path, "**/graph*freeze.out")
    arcann_logger.info(f"Deleting compressing unwanted output file...")
    remove_files_matching_glob(current_path, "**/graph*compress.out")
    arcann_logger.info(f"Deleting extra model.ckpt...")
    remove_files_matching_glob(current_path, "**/model.ckpt-*")
    arcann_logger.info(f"Deleting models files files...")
    remove_files_matching_glob(current_path, "**/*.pb")
    arcann_logger.info(f"Deleting extra training files...")
    remove_files_matching_glob(current_path, "**/checkpoint")
    remove_files_matching_glob(current_path, "**/input_v2_compat.json")
    arcann_logger.info(f"Deleting job error files...")
    remove_files_matching_glob(current_path, "**/DeepMD_*")
    arcann_logger.info(f"Deleting the data folder ...")
    if (current_path / "data").is_dir():
        remove_tree(current_path / "data")
    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"
        if (local_path / "model-compression").is_dir():
            arcann_logger.info("Deleting the temp model-compression folder...")
            remove_tree(local_path / "model-compression")
    arcann_logger.info(f"Cleaning done!")

    # End
    arcann_logger.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del user_config_filename
    del main_json, training_config
    del curr_iter, padded_curr_iter

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
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
