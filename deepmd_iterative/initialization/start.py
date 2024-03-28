"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/28
"""

# Standard library modules
import logging
import sys
from pathlib import Path

# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.common.filesystem import check_directory, check_file_existence
from deepmd_iterative.common.json import backup_and_overwrite_json_file, load_default_json_file, load_json_file, write_json_file
from deepmd_iterative.initialization.utils import generate_main_json, check_properties_file, check_dptrain_properties, check_lmp_properties, check_typeraw_properties
from deepmd_iterative.common.utils import natural_sort_key


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
    user_files_path = current_path / "user_files"

    # Log the step and phase of the program
    logging.info(f"Step: {current_step.capitalize()}.")
    logging.debug(f"Current path: {current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Load the default input JSON
    default_input_json = load_default_json_file(deepmd_iterative_path / "assets" / "default_config.json")[current_step]
    default_input_json_present = bool(default_input_json)
    if default_input_json_present and not (current_path / "default_input.json").is_file():
        write_json_file(default_input_json, (current_path / "default_input.json"), read_only=True)

    logging.debug(f"default_input_json: {default_input_json}")
    logging.debug(f"default_input_json_present: {default_input_json_present}")

    # Load the user input JSON
    user_input_json = load_json_file((current_path / user_input_json_filename), abort_on_error=False)
    user_input_json_present = bool(user_input_json)
    logging.debug(f"user_input_json: {user_input_json}")
    logging.debug(f"user_input_json_present: {user_input_json_present}")

    # Check if a data folder is present in the training path
    check_directory((training_path / "data"), error_msg=f"No data folder found in: {training_path}")

    # Check the properties file
    properties_dict = check_properties_file(user_files_path / "properties.txt")

    # Auto-populate the systems_auto
    if "system_auto" not in user_input_json:
        list_of_lmp = [file.stem for file in user_files_path.glob("*.lmp")]
        if not list_of_lmp:
            logging.error(f"No lmp found in {user_files_path}")
            logging.error(f"Aborting...")
            return 1
        else:
            user_input_json["systems_auto"] = list_of_lmp.sort()
            logging.info(f"Auto-populated 'systems_auto' with: {user_input_json['systems_auto']}")
    else:
        for system_auto in user_input_json["systems_auto"]:
            if not (user_files_path / f"{system_auto}.lmp").is_file():
                logging.error(f"File not found: {user_files_path / f'{system_auto}.lmp'} but requested as system")
                logging.error(f"Aborting...")
                return 1
    logging.debug(f"user_input_json: {user_input_json}")

    # Generate the main JSON, the merged input JSON and the padded current iteration
    main_json, merged_input_json, padded_curr_iter = generate_main_json(user_input_json, default_input_json)
    logging.debug(f"main_json: {main_json}")
    logging.debug(f"merged_input_json : {merged_input_json }")
    logging.debug(f"padded_curr_iter : {padded_curr_iter}")

    # Add the properties dictionary to the main JSON
    main_json["properties"] = properties_dict

    # Check the lmp against the properties
    for system_auto in main_json["systems_auto"]:
        check_lmp_properties(user_files_path / f"{system_auto}.lmp", main_json["properties"])

    # Check the dptrain against the properties
    check_dptrain_properties(user_files_path, main_json["properties"])

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
        # Check the type.raw file against the properties
        check_typeraw_properties(initial_dataset_path / "type.raw", main_json["properties"])
        initial_dataset_set_path = initial_dataset_path / "set.000"
        for data_type in ["box", "coord", "energy", "force"]:
            check_file_existence(initial_dataset_set_path / (data_type + ".npy"))
        del data_type
        initial_datasets_json[initial_dataset_path.name] = np.load(initial_dataset_set_path / "box.npy").shape[0]
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
    write_json_file(main_json, (control_path / "config.json"), read_only=True)
    write_json_file(initial_datasets_json, (control_path / "initial_datasets.json"), read_only=True)
    backup_and_overwrite_json_file(merged_input_json, (current_path / "used_input.json"), read_only=True)

    # End
    logging.info(f"-" * 88)
    logging.info(f"Step: {current_step.capitalize()} is a success!")

    # Cleaning
    del current_path, control_path, training_path
    del default_input_json, default_input_json_present, user_input_json, user_input_json_present, user_input_json_filename
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
