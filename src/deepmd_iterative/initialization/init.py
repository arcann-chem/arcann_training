from pathlib import Path
import logging
import sys

# Non-standard library imports
import numpy as np

# deepmd_iterative imports
from deepmd_iterative.common.file import check_directory, check_file_existence
from deepmd_iterative.common.json import (
    backup_and_overwrite_json_file,
    load_default_json_file,
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.json_parameters import set_main_config


# Main function
def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path,
    fake_machine=None,
    user_config_filename: str = "config.json",
):
    # Get the current path and set the training path as the current path
    current_path = Path(".").resolve()
    training_path = current_path

    # Log the step and phase of the program
    logging.info(f"Step: {current_step.capitalize()}")
    logging.debug(f"Current path: {current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Load the default config (JSON)
    default_config = load_default_json_file(deepmd_iterative_path / "data" / "default_config.json")[current_step]
    default_config_present = bool(default_config)
    logging.debug(f"default_config: {default_config}")
    logging.debug(f"default_config_present: {default_config_present}")

    # Load the user config (JSON)
    user_config = load_json_file((current_path / user_config_filename))
    user_config_present = bool(user_config)
    logging.debug(f"user_config: {user_config}")
    logging.debug(f"user_config_present: {user_config_present}")

    # Check if a "data" folder is present in the training path
    check_directory(
        (training_path / "data"),
        error_msg=f"No data folder found in: {training_path}",
    )

    # Create the config JSON file (and set everything)
    main_config, current_config, padded_curr_iter = set_main_config(user_config, default_config)
    logging.debug(f"main_config: {main_config}")
    logging.debug(f"current_config : {current_config }")
    logging.debug(f"padded_curr_iter : {padded_curr_iter}")

    # Create the control directory (where JSON files are)
    control_path = training_path / "control"
    control_path.mkdir(exist_ok=True)
    check_directory(control_path)

    # Create the initial training directory
    (training_path / f"{padded_curr_iter}-training").mkdir(exist_ok=True)
    check_directory((training_path / f"{padded_curr_iter}-training"))

    # Check if data exists, get init_* datasets and extract number of atoms and cell dimensions
    initial_datasets_path = [_ for _ in (training_path / "data").glob("init_*")]
    if len(initial_datasets_path) == 0:
        logging.error("No initial data sets found.")
        logging.error("Aborting...")
        return 1
    logging.debug(f"initial_datasets_path: {initial_datasets_path}")

    # Create the initial datasets JSON
    initial_datasets_json = {}
    for it_initial_datasets_path in initial_datasets_path:
        check_file_existence(it_initial_datasets_path / "type.raw")
        it_initial_datasets_set_path = it_initial_datasets_path / "set.000"
        for it_npy in ["box", "coord", "energy", "force"]:
            check_file_existence(it_initial_datasets_set_path / (it_npy + ".npy"))
        del it_npy
        initial_datasets_json[it_initial_datasets_path.name] = np.load(
            str(it_initial_datasets_set_path / "box.npy")
        ).shape[0]
    logging.debug(f"initial_datasets_json: {initial_datasets_json}")
    del it_initial_datasets_path, it_initial_datasets_set_path
    del initial_datasets_path

    # Populate
    main_config["initial_datasets"] = [zzz for zzz in initial_datasets_json.keys()]

    # DEBUG: Print the dicts
    logging.debug(f"main_config: {main_config}")
    logging.debug(f"initial_datasets_json: {initial_datasets_json}")
    logging.debug(f"user_config: {user_config}")
    logging.debug(f"current_config: {current_config}")

    # Dump the dicts
    logging.info(f"-" * 88)
    write_json_file(main_config, (control_path / "config.json"))
    write_json_file(initial_datasets_json, (control_path / "initial_datasets.json"))
    backup_and_overwrite_json_file(current_config, (current_path / user_config_filename))

    # Delete
    del current_path, control_path, training_path
    del default_config, default_config_present, user_config, user_config_present, user_config_filename
    del main_config, initial_datasets_json, current_config

    logging.info(f"-" * 88)
    logging.info(f"Step: {current_step.capitalize()} is a success!")

    return 0


# Standalone part
if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "initialization",
            "init",
            Path(sys.argv[1]),
            fake_machine = sys.argv[2],
            user_config_filename = sys.argv[3],
        )
    else:
        pass
