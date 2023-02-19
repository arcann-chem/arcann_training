from pathlib import Path
import logging
import sys
import copy

# ### Non-standard imports
import numpy as np

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    backup_and_overwrite_json_file,
    load_default_json_file,
)
from deepmd_iterative.common.file import check_directory, check_file_existence
from deepmd_iterative.common.generate_config import generate_config_json

def main(
    step_name: str,
    phase_name: str,
    deepmd_iterative_path,
    fake_machine=None,
    input_fn: str="input.json",
):

    current_path = Path(".").resolve()
    training_path = current_path

    logging.info(f"Step: {step_name.capitalize()}")
    logging.debug(f"Current path: {current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # ### Get default inputs json
    default_present = False
    default_input_json = load_default_json_file(deepmd_iterative_path / "data" / "inputs.json")
    if bool(default_input_json):
        default_present = True

    # ### Get input json (user one)
    input_json = load_json_file((current_path / input_fn))
    new_input_json = copy.deepcopy(input_json)

    # ### Check if the input provided is correct
    if step_name not in input_json["step_name"]:
        logging.error(f"Wrong input: {input_json['step_name']}")
        logging.error("Aborting...")
        return 1

    # ### Check if we are in the correct dir
    check_directory(
        (training_path / "data"),
        error_msg=f"No data folder found in: {training_path}",
    )

    # ### Create the config.json (and set everything)
    config_json, current_iteration_zfill = generate_config_json(
        input_json,
        new_input_json,
        default_input_json,
        step_name,
        default_present
    )

    # ### Create the control directory
    control_path = training_path / "control"
    control_path.mkdir(exist_ok=True)
    check_directory(control_path)

    # ### Create the initial training directory
    (training_path / f"{current_iteration_zfill}-training").mkdir(exist_ok=True)
    check_directory((training_path / f"{current_iteration_zfill}-training"))

    # ### Check if data exists, get init_* datasets and extract number of atoms and cell dimensions
    initial_datasets_path = [_ for _ in (training_path / "data").glob("init_*")]
    if len(initial_datasets_path) == 0:
        logging.error("No initial data sets found.")
        logging.error("Aborting...")
        return 1

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

    del it_initial_datasets_path, it_initial_datasets_set_path
    del initial_datasets_path

    config_json["initial_datasets"] = [zzz for zzz in initial_datasets_json.keys()]

    logging.debug(config_json)
    logging.debug(initial_datasets_json)

    # ### Dump the dicts
    logging.info(f"-" * 88)
    write_json_file(config_json, (control_path / "config.json"))
    write_json_file(initial_datasets_json, (control_path / "initial_datasets.json"))
    backup_and_overwrite_json_file(new_input_json, (current_path / input_fn))

    del control_path
    del input_json, default_input_json, default_present, new_input_json
    del config_json, initial_datasets_json
    del training_path, current_path

    logging.info(f"-" * 88)
    logging.info(f"Step: {step_name.capitalize()} is a success !")

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "initialization",
            "init",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass