"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/09/25
"""
# Standard library modules
import logging
import sys
from pathlib import Path

# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.common.filesystem import check_directory, check_file_existence
from deepmd_iterative.common.json import (
    backup_and_overwrite_json_file,
    load_default_json_file,
    load_json_file,
    write_json_file,
)
from deepmd_iterative.initialization.utils import generate_main_json


# Main function
def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path,
    fake_machine=None,
    user_input_json_filename: str = "input.json",
):
    # Get the current path and set the training path as the current path
    current_path = Path(".").resolve()
    training_path = current_path

    # Log the step and phase of the program
    logging.info(f"Step: {current_step.capitalize()}.")
    logging.debug(f"Current path: {current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Load the default input JSON
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "assets" / "default_config.json"
    )[current_step]
    default_input_json_present = bool(default_input_json)
    logging.debug(f"default_input_json: {default_input_json}")
    logging.debug(f"default_input_json_present: {default_input_json_present}")

    # Load the user input JSON
    user_input_json = load_json_file((current_path / user_input_json_filename))
    user_input_json_present = bool(user_input_json)
    logging.debug(f"user_input_json: {user_input_json}")
    logging.debug(f"user_input_json_present: {user_input_json_present}")

    # Check if a data folder is present in the training path
    check_directory(
        (training_path / "data"),
        error_msg=f"No data folder found in: {training_path}",
    )

    # Generate the main JSON, the merged input JSON and the padded current iteration
    main_json, merged_input_json = generate_main_json(
        user_input_json, default_input_json
    )
    padded_curr_iter = str(main_json["current_iteration"]).zfill(3)
    logging.debug(f"main_json: {main_json}")
    logging.debug(f"merged_input_json : {merged_input_json }")
    logging.debug(f"padded_curr_iter : {padded_curr_iter}")

    # Create the control directory
    control_path = training_path / "control"
    control_path.mkdir(exist_ok=True)
    check_directory(control_path)

    # Create the initial training directory
    (training_path / f"{padded_curr_iter}-training").mkdir(exist_ok=True)
    check_directory((training_path / f"{padded_curr_iter}-training"))

    # Check if data exists, get init_* datasets and extract number of atoms and cell dimensions
    initial_datasets_paths = [_ for _ in (training_path / "data").glob("init_*")]
    if len(initial_datasets_paths) == 0:
        logging.error(f"No initial datasets found.")
        logging.error(f"Aborting...")
        return 1
    logging.debug(f"initial_datasets_paths: {initial_datasets_paths}")

    # Create and set the initial datasets JSON
    initial_datasets_json = {}
    for initial_dataset_path in initial_datasets_paths:
        check_file_existence(initial_dataset_path / "type.raw")
        initial_dataset_set_path = initial_dataset_path / "set.000"
        for data_type in ["box", "coord", "energy", "force"]:
            check_file_existence(initial_dataset_set_path / (data_type + ".npy"))
        del data_type
        initial_datasets_json[initial_dataset_path.name] = np.load(
            initial_dataset_set_path / "box.npy"
        ).shape[0]
    logging.debug(f"initial_datasets_json: {initial_datasets_json}")
    del initial_dataset_path, initial_datasets_paths, initial_dataset_set_path

    # Populate
    main_json["initial_datasets"] = [_ for _ in initial_datasets_json.keys()]

    # DEBUG: Print the JSON files
    logging.debug(f"main_json: {main_json}")
    logging.debug(f"initial_datasets_json: {initial_datasets_json}")
    logging.debug(f"user_input_json: {user_input_json}")
    logging.debug(f"merged_input_json: {merged_input_json}")

    # Dump the JSON files (main, initial datasets and merged input)
    logging.info(f"-" * 88)
    write_json_file(main_json, (control_path / "config.json"))
    write_json_file(initial_datasets_json, (control_path / "initial_datasets.json"))
    backup_and_overwrite_json_file(
        merged_input_json, (current_path / user_input_json_filename)
    )

    # End
    logging.info(f"-" * 88)
    logging.info(f"Step: {current_step.capitalize()} is a success!")

    # Cleaning
    del current_path, control_path, training_path
    del (
        default_input_json,
        default_input_json_present,
        user_input_json,
        user_input_json_present,
        user_input_json_filename,
    )
    del padded_curr_iter
    del main_json, initial_datasets_json, merged_input_json

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


# Standalone part
if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "initialization",
            "start",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
