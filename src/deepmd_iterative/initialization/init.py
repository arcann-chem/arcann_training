from pathlib import Path
import logging
import sys
import copy

# ### Non-standard imports
import numpy as np

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    json_read,
    json_dump,
    json_dump_bak,
    read_default_input_json,
    read_key_input_json,
)
from deepmd_iterative.common.files import check_dir, check_file


def main(
    step_name,
    phase_name,
    deepmd_iterative_apath,
    fake_cluster=None,
    input_fn="input.json",
):

    current_apath = Path(".").resolve()
    training_iterative_apath = current_apath

    logging.info(f"Step: {step_name.capitalize()}")
    logging.debug(f"Current path: {current_apath}")
    logging.debug(f"Training path: {training_iterative_apath}")
    logging.debug(f"Program path: {deepmd_iterative_apath}")
    logging.info(f"-" * 88)

    # ### Get default inputs json
    default_present = False
    default_input_json = read_default_input_json(
        deepmd_iterative_apath / "data" / "inputs.json"
    )
    if bool(default_input_json):
        default_present = True

    # ### Get input json (user one)
    input_json = json_read((current_apath / input_fn), True, True)
    new_input_json = copy.deepcopy(input_json)

    # ### Check if the input provided is correct
    if step_name not in input_json["step_name"]:
        logging.error(f"Wrong input: {input_json['step_name']}")
        logging.error("Aborting...")
        return 1

    # ### Check if we are in the correct dir
    check_dir(
        (training_iterative_apath / "data"),
        True,
        error_msg=f"No data folder found in: {training_iterative_apath}",
    )

    # ### Create the config.json (and set everything)
    config_json = {"system": read_key_input_json(
        input_json,
        new_input_json,
        "system",
        default_input_json,
        step_name,
        default_present,
    ), "nb_nnp": read_key_input_json(
        input_json,
        new_input_json,
        "nb_nnp",
        default_input_json,
        step_name,
        default_present,
    ), "current_iteration": 0}
    current_iteration_zfill = str(config_json["current_iteration"]).zfill(3)
    config_json["subsys_nr"] = {}
    for it0_subsys_nr, it_subsys_nr in enumerate(
        read_key_input_json(
            input_json,
            new_input_json,
            "subsys_nr",
            default_input_json,
            step_name,
            default_present,
        )
    ):
        config_json["subsys_nr"][it_subsys_nr] = {}
    del it0_subsys_nr, it_subsys_nr

    # ### Create the control directory
    control_apath = training_iterative_apath / "control"
    control_apath.mkdir(exist_ok=True)
    check_dir(control_apath, True)

    # ### Create the initial training directory
    (
        training_iterative_apath
        / f"{current_iteration_zfill}-training"
    ).mkdir(exist_ok=True)
    check_dir(
        (
            training_iterative_apath
            / f"{current_iteration_zfill}-training"
        ),
        True,
    )

    # ### Check if data exists, get init_* datasets and extract number of atoms and cell dimensions
    initial_datasets_apath = [
        _ for _ in (training_iterative_apath / "data").glob("init_*")
    ]
    if len(initial_datasets_apath) == 0:
        logging.error("No initial data sets found.")
        logging.error("Aborting...")
        return 1

    initial_datasets_json = {}
    for it_initial_datasets_apath in initial_datasets_apath:
        check_file(it_initial_datasets_apath / "type.raw", True, True)
        it_initial_datasets_set_apath = it_initial_datasets_apath / "set.000"
        for it_npy in ['box", "coord", "energy", "force']:
            check_file(it_initial_datasets_set_apath / (it_npy + ".npy"), True, True)
        del it_npy
        initial_datasets_json[it_initial_datasets_apath.name] = np.load(
            str(it_initial_datasets_set_apath / "box.npy")
        ).shape[0]

    del it_initial_datasets_apath, it_initial_datasets_set_apath
    del initial_datasets_apath

    config_json["initial_datasets"] = [zzz for zzz in initial_datasets_json.keys()]

    logging.debug(config_json)
    logging.debug(initial_datasets_json)

    # ### Dump the dicts
    logging.info(f"-" * 88)
    json_dump(config_json, (control_apath / "config.json"), True)
    json_dump(initial_datasets_json, (control_apath / "initial_datasets.json"), True)
    json_dump_bak(new_input_json, (current_apath / input_fn), True)

    del control_apath
    del input_json, default_input_json, default_present, new_input_json
    del config_json, initial_datasets_json
    del training_iterative_apath, current_apath

    logging.info(f"-" * 88)
    logging.info(f"Step: {step_name.capitalize()} is a success !")

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "initialization",
            "init",
            Path(sys.argv[1]),
            fake_cluster=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
