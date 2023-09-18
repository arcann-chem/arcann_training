"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/09/18
"""
# Standard library modules
import logging
import sys
from pathlib import Path

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.filesystem import (
    remove_files_matching_glob,
    remove_tree,
    remove_all_symlink,
)
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
    main_json = load_json_file((control_path / "config.json"))
    training_config = load_json_file(
        (control_path / f"training_{padded_curr_iter}.json")
    )

    # Check if we can continue
    if not training_config["is_incremented"]:
        logging.error(f"Lock found. Please execute 'training increment' first.")
        logging.error(f"Aborting...")
        return 1
    logging.critical(f"This is the cleaning step for training step.")
    logging.critical(f"It should be run after training increment phase.")
    logging.critical(f"This is will delete:")
    logging.critical(
        f"symbolic links, 'job_*.sh', 'training.out', 'graph*freeze.out', 'graph*compress.out', 'checkpoint.*', 'input_v2_compat.json', 'DeepMD_*'"
    )
    logging.critical(f"'model-compression' folders")
    logging.critical(
        f"'*.pb' models files (they are saved in the '{current_path.parent / 'NNP'}' root folder)"
    )
    logging.critical(
        f"'data' folder (it is saved in the '{current_path.parent / 'data'}' root folder)"
    )
    logging.critical(f"in the folder: '{current_path}' and all subdirectories.")
    continuing = input(
        f"Do you want to continue? [Enter 'Y' for yes, or any other key to abort]: "
    )
    if continuing == "Y":
        del continuing
    else:
        logging.error(f"Aborting...")
        return 0

    # Delete
    logging.info("Deleting symbolic links...")
    remove_all_symlink(current_path)
    logging.info("Deleting job files...")
    remove_files_matching_glob(current_path, "**/job_*.sh")
    logging.info(f"Deleting training unwanted output file..")
    remove_files_matching_glob(current_path, "**/training.out")
    logging.info(f"Deleting freezing unwanted output files...")
    remove_files_matching_glob(current_path, "**/graph*freeze.out")
    logging.info(f"Deleting compressing unwanted output file...")
    remove_files_matching_glob(current_path, "**/graph*compress.out")
    logging.info(f"Deleting extra model.ckpt...")
    remove_files_matching_glob(current_path, "**/model.ckpt-*")
    logging.info(f"Deleting models files files...")
    remove_files_matching_glob(current_path, "**/*.pb")
    logging.info(f"Deleting extra training files...")
    remove_files_matching_glob(current_path, "checkpoint.*")
    remove_files_matching_glob(current_path, "input_v2_compat.json")
    logging.info(f"Deleting job error files...")
    remove_files_matching_glob(current_path, "**/DeepMD_*")
    logging.info(f"Deleting the data folder ...")
    if (current_path / "data").is_dir():
        remove_tree(current_path / "data")
    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"
        if (local_path / "model-compression").is_dir():
            logging.info("Deleting the temp model-compression folder...")
            remove_tree(local_path / "model-compression")
    logging.info(f"Cleaning done!")

    # End
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del user_config_filename
    del main_json, training_config
    del curr_iter, padded_curr_iter

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
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
