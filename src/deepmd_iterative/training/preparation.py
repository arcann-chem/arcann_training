from pathlib import Path
import logging
import sys
import copy
import subprocess
import random

# Non-standard library imports
import numpy as np

# deepmd_iterative imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.file import (
    check_directory,
    check_file_existence,
    file_to_list_of_strings,
    write_list_of_strings_to_file,
)
from deepmd_iterative.common.json import (
    backup_and_overwrite_json_file,
    load_default_json_file,
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.json_parameters import (
    get_machine_keyword,
    set_training_json,
)
from deepmd_iterative.common.list import replace_substring_in_list_of_strings
from deepmd_iterative.common.machine import get_machine_spec_for_step
from deepmd_iterative.common.slurm import replace_in_slurm_file_general
from deepmd_iterative.common.training import (
    calculate_decay_rate,
    calculate_decay_steps,
    check_initial_datasets,
)

def main(
    step_name: str,
    phase_name: str,
    deepmd_iterative_path: Path,
    fake_machine = None,
    input_fn: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()}")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if correct folder
    validate_step_folder(step_name)

    # Get iteration
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Load the master input JSON file for the program
    default_present = False
    default_json = load_default_json_file(deepmd_iterative_path / "data" / "input_defaults.json")[step_name]
    if bool(default_json):
        default_present = True
    logging.debug(f"default_json: {default_json}")
    logging.debug(f"default_present: {default_present}")

    # Load the user input JSON file
    if (current_path / input_fn).is_file():
        input_json = load_json_file((current_path / input_fn))
        input_present = True
    else:
        input_json = {}
        input_present = False
    logging.debug(f"input_json: {input_json}")
    logging.debug(f"input_present: {input_present}")

    # Make a deepcopy
    new_input_json = copy.deepcopy(input_json)

    # Get control path and config_json
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))

    # Get extra needed paths
    jobs_path = deepmd_iterative_path / "data" / "jobs" / step_name

    # Load the previous training_json
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        prevtraining_json = load_json_file((control_path / f"training_{padded_prev_iter}.json"))
    else:
        prevtraining_json = {}

    # Get the machine keyword (input override previous training override default_json)
    # And update the new input
    user_machine_keyword = get_machine_keyword(input_json, prevtraining_json, default_json)
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    new_input_json["user_machine_keyword"] = user_machine_keyword
    logging.debug(f"new_input_json: {new_input_json}")
    # Set it to None if bool, because: get_machine_spec_for_step needs None
    user_machine_keyword = (
        None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")

    # From the keyword (or default), get the machine spec (or for the fake one)
    (
        machine,
        machine_spec,
        machine_walltime_format,
        machine_launch_command,
    ) = get_machine_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "training",
        fake_machine,
        user_machine_keyword,
    )
    logging.debug(f"machine: {machine}")
    logging.debug(f"machine_spec: {machine_spec}")
    logging.debug(f"machine_walltime_format: {machine_walltime_format}")
    logging.debug(f"machine_launch_command: {machine_launch_command}")

    if fake_machine is not None:
        logging.info(f"Pretending to be on: {fake_machine}.")
    else:
        logging.info(f"We are on: {machine}.")
    del fake_machine

    # Checked if the iter > 0, if there load the past labeling and see if it's extracted
    if curr_iter > 0:
        labeling_json = load_json_file(
            (control_path / f"labeling_{padded_curr_iter}.json")
        )
        if not labeling_json["is_extracted"]:
            logging.error("Lock found. Run/Check first: labeling extract.")
            logging.error("Aborting...")
            return 1

    # Create the training JSON file (and set everything)
    # Priority: input > previous > default
    training_json, new_input_json = set_training_json(
        control_path,
        padded_curr_iter,
        input_json,
        prevtraining_json,
        default_json,
        new_input_json,
    )
    logging.debug(f"training_json: {training_json}")
    logging.debug(f"new_input_json: {new_input_json}")

    # Set additional machine-related parameters in the training JSON file (not need in the input)
    training_json = {
        **training_json,
        "machine": machine,
        "project_name": machine_spec["project_name"],
        "allocation_name": machine_spec["allocation_name"],
        "arch_name": machine_spec["arch_name"],
        "arch_type": machine_spec["arch_type"],
        "launch_command": machine_launch_command,
    }

    # Check if the training job file exists
    check_file_existence(
        jobs_path / f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh",
        error_msg=f"No SLURM file present for {step_name.capitalize()} / {phase_name.capitalize()} on this machine.",
    )
    slurm_file_master = file_to_list_of_strings(
        jobs_path / f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh"
    )
    del jobs_path
    logging.debug(f"slurm_file_master: {slurm_file_master[0:5]}, {slurm_file_master[-5:-1]}")

    # TODO: Maybe as function / parameters file for later
    # Check DeePMD version
    if training_json["deepmd_model_version"] not in [2.0, 2.1]:
        logging.critical(
            f"Invalid deepmd model version (2.0 or 2.1): {training_json['deepmd_model_version']}."
        )
        logging.critical("Aborting...")
        return 1

    # Check DeePMD descriptor type
    if training_json["deepmd_model_type_descriptor"] not in ["se_e2_a"]:
        logging.critical(
            f"Invalid deepmd type descriptor (se_e2_a): {training_json['deepmd_model_type_descriptor']}."
        )
        logging.critical("Aborting...")
        return 1

    # Check mismatch between machine/arch_name/arch and DeePMD
    if training_json["deepmd_model_version"] < 2.0:
        logging.critical("Only version >= 2.0 on Jean Zay!")
        logging.critical("Aborting...")
        return 1
    if (
        training_json["deepmd_model_version"] < 2.1
        and training_json["arch_name"] == "a100"
    ):
        logging.critical("Only version >= 2.1 on Jean Zay A100!")
        logging.critical("Aborting...")
        return 1


    # Check if the default input json file exists
    input_file_fpath = (
        training_path
        / "files"
        / (
            f"dptrain_{training_json['deepmd_model_version']}_{training_json['deepmd_model_type_descriptor']}.json"
        )
    ).resolve()
    training_input_json = load_json_file(input_file_fpath)
    config_json["type_map"] = {}
    config_json["type_map"] = training_input_json["model"]["type_map"]
    del input_file_fpath
    logging.debug(f"training_input_json: {training_input_json}")
    logging.debug(f"config_json: {config_json}")

    # Check the initial sets json file
    datasets_initial_json = check_initial_datasets(training_path)
    logging.debug(f"datasets_initial_json: {datasets_initial_json}")

    # Let us find what is in data
    data_path = training_path / "data"
    check_directory(data_path)
    subsys_name = []

    # This is building the datasets (roughly 200 lines)
    # TODO later
    datasets_extra = []
    datasets_validation = []
    for it_data_folders in data_path.iterdir():
        if it_data_folders.is_dir():
            # Escape initial/extra sets, because initial get added first and extra as last, and also escape init_
            # not in initial_json (in case of removal)
            if (
                it_data_folders.name not in datasets_initial_json.keys()
                and "extra_" != it_data_folders.name[:6]
                and "init_" != it_data_folders.name[:5]
            ):
                # Escape test sets
                if "test_" != it_data_folders.name[:5]:
                    # Escape if set iter is superior as iter, it is only for reprocessing old stuff.
                    try:
                        if (
                            int(it_data_folders.name.rsplit("_", 1)[-1])
                            <= curr_iter
                        ):
                            subsys_name.append(it_data_folders.name.rsplit("_", 1)[0])
                    # TODO: Better except clause
                    except:
                        pass
                else:
                    datasets_validation.append(it_data_folders.name)
            # Get the extra sets !
            elif "extra_" == it_data_folders.name[:6]:
                datasets_extra.append(it_data_folders.name)
    del it_data_folders

    del datasets_validation

    # Training sets list construction
    datasets_training = []
    datasets_training_json = []
    # Initial
    nb_initial = 0
    if training_json["use_initial_datasets"]:
        for it_datasets_initial_json in datasets_initial_json.keys():
            if (data_path / it_datasets_initial_json).is_dir():
                datasets_training.append(
                    f"{(Path(data_path.parts[-1]) / it_datasets_initial_json / '_')}"[
                        :-1
                    ]
                )
                datasets_training_json.append(it_datasets_initial_json)
                nb_initial = (
                    nb_initial + datasets_initial_json[it_datasets_initial_json]
                )
        del it_datasets_initial_json
    del datasets_initial_json

    # Non-Reactive (aka subsys_nr in the initialization first) && all the others are REACTIVE !
    # Total and what is added just for this iteration
    nb_added_nr = 0
    nb_added_r = 0
    nb_added_nr_iter = 0
    nb_added_r_iter = 0

    # This trick remove duplicates from list via set
    subsys_name = list(set(subsys_name))
    subsys_name = [i for i in subsys_name if i not in config_json["subsys_nr"]]
    subsys_name = [
        i
        for i in subsys_name
        if i not in [zzz + "-disturbed" for zzz in config_json["subsys_nr"]]
    ]
    subsys_name = sorted(subsys_name)
    config_json["subsys_r"] = subsys_name
    del subsys_name

    if curr_iter > 0:
        for it_iteration in np.arange(1, curr_iter + 1):
            it_iteration_zfill = str(it_iteration).zfill(3)
            try:
                for system_it in config_json["subsys_nr"]:
                    if (data_path / f"{system_it}_{it_iteration_zfill}").is_dir():
                        datasets_training.append(
                            f"{(Path(data_path.parts[-1]) / (system_it+'_'+it_iteration_zfill) / '_')}"[
                                :-1
                            ]
                        )
                        datasets_training_json.append(
                            f"{system_it}_{it_iteration_zfill}"
                        )
                        nb_added_nr = (
                            nb_added_nr
                            + np.load(
                                str(
                                    data_path
                                    / f"{system_it}_{it_iteration_zfill}"
                                    / "set.000"
                                    / "box.npy"
                                )
                            ).shape[0]
                        )
                        if it_iteration == curr_iter:
                            nb_added_nr_iter = (
                                nb_added_nr_iter
                                + np.load(
                                    str(
                                        data_path
                                        / f"{system_it}_{it_iteration_zfill}"
                                        / "set.000"
                                        / "box.npy"
                                    )
                                ).shape[0]
                            )
                del system_it
            except (KeyError, NameError):
                pass
            try:
                for system_it in [
                    zzz + "-disturbed" for zzz in config_json["subsys_nr"]
                ]:
                    if (data_path / f"{system_it}_{it_iteration_zfill}").is_dir():
                        datasets_training.append(
                            f"{(Path(data_path.parts[-1]) / (system_it+'_'+it_iteration_zfill) / '_')}"[
                                :-1
                            ]
                        )
                        datasets_training_json.append(
                            f"{system_it}_{it_iteration_zfill}"
                        )
                        nb_added_nr = (
                            nb_added_nr
                            + np.load(
                                str(
                                    data_path
                                    / f"{system_it}_{it_iteration_zfill}"
                                    / "set.000"
                                    / "box.npy"
                                )
                            ).shape[0]
                        )
                        if it_iteration == curr_iter:
                            nb_added_nr_iter = (
                                nb_added_nr_iter
                                + np.load(
                                    str(
                                        data_path
                                        / f"{system_it}_{it_iteration_zfill}"
                                        / "set.000"
                                        / "box.npy"
                                    )
                                ).shape[0]
                            )
                del system_it
            except (KeyError, NameError):
                pass
            try:
                for system_it in config_json["subsys_r"]:
                    if (data_path / f"{system_it}_{it_iteration_zfill}").is_dir():
                        datasets_training.append(
                            f"{(Path(data_path.parts[-1]) / (system_it+'_'+it_iteration_zfill) / '_')}"[
                                :-1
                            ]
                        )
                        datasets_training_json.append(
                            f"{system_it}_{it_iteration_zfill}"
                        )
                        nb_added_nr = (
                            nb_added_nr
                            + np.load(
                                str(
                                    data_path
                                    / f"{system_it}_{it_iteration_zfill}"
                                    / "set.000"
                                    / "box.npy"
                                )
                            ).shape[0]
                        )
                        if it_iteration == curr_iter:
                            nb_added_nr_iter = (
                                nb_added_nr_iter
                                + np.load(
                                    str(
                                        data_path
                                        / f"{system_it}_{it_iteration_zfill}"
                                        / "set.000"
                                        / "box.npy"
                                    )
                                ).shape[0]
                            )
                del system_it
            except (KeyError, NameError):
                pass
        del it_iteration, it_iteration_zfill

    # Finally the extra sets !
    nb_extra = 0
    if training_json["use_extra_datasets"]:
        config_json["datasets_extra"] = datasets_extra
        del datasets_extra
        for it_datasets_extra in config_json["datasets_extra"]:
            datasets_training.append(
                f"{(Path(data_path.parts[-1]) / it_datasets_extra / '_')}"[:-1]
            )
            datasets_training_json.append(it_datasets_extra)
            nb_extra = (
                nb_extra
                + np.load(
                    str(data_path / it_datasets_extra / "set.000" / "box.npy")
                ).shape[0]
            )
        del it_datasets_extra
    else:
        del datasets_extra

    # Total
    nb_trained = nb_initial + nb_added_nr + nb_added_r + nb_extra
    logging.debug(f"nb_trained: {nb_trained} = {nb_initial} + {nb_added_nr} + {nb_added_r} + {nb_extra}")
    logging.debug(f"datasets_training: {datasets_training}")

    # Update the inputs with the sets
    training_input_json["training"]["training_data"]["systems"] = datasets_training

    # Update the training JSON
    training_json = {
        **training_json,
        "training_data": datasets_training_json,
        "nb_trained": nb_trained,
        "nb_initial": nb_initial,
        "nb_added_nr": nb_added_nr,
        "nb_added_r": nb_added_r,
        "nb_added_nr_iter": nb_added_nr_iter,
        "nb_added_r_iter": nb_added_r_iter,
        "nb_extra": nb_extra,
    }
    logging.debug(f"training_json: {training_json}")

    del datasets_training_json
    del nb_trained, nb_initial, nb_extra
    del nb_added_nr, nb_added_r, nb_added_nr_iter, nb_added_r_iter

    # Here calculate the parameters

    # decay_steps it auto-recalculated as funcion of nb_trained
    logging.debug(f"training_json - decay_steps: {training_json['decay_steps']}")
    logging.debug(f"new_input_json - decay_steps: {new_input_json['decay_steps']}")
    if not training_json["decay_steps_fixed"]:
        decay_steps = calculate_decay_steps(training_json["nb_trained"], training_json["decay_steps"])
        logging.debug(f"Recalculating decay_steps")
        # Update the training JSON and the new input JSON:
        training_json["decay_steps"] = decay_steps
        new_input_json["decay_steps"] = decay_steps
    else:
        decay_steps = training_json["decay_steps"]
    logging.debug(f"decay_steps: {decay_steps}")
    logging.debug(f"training_json - decay_steps: {training_json['decay_steps']}")
    logging.debug(f"new_input_json - decay_steps: {new_input_json['decay_steps']}")

    # numb_steps and decay_rate
    logging.debug(f"training_json - numb_steps / decay_rate: {training_json['numb_steps']} / {training_json['decay_rate']}")
    logging.debug(f"new_input_json - numb_steps / decay_rate: {new_input_json['numb_steps']} / {new_input_json['decay_rate']}")
    numb_steps = training_json["numb_steps"]
    decay_rate_new = calculate_decay_rate(
        numb_steps,
        training_json["start_lr"],
        training_json["stop_lr"],
        training_json["decay_steps"],
    )
    while decay_rate_new < training_json["decay_rate"]:
        numb_steps = numb_steps + 10000
        decay_rate_new = calculate_decay_rate(
            numb_steps,
            training_json["start_lr"],
            training_json["stop_lr"],
            training_json["decay_steps"],
        )
    # Update the training JSON and the new input JSON:
    training_json["numb_steps"] = int(numb_steps)
    training_json["decay_rate"] = decay_rate_new
    new_input_json["numb_steps"] = int(numb_steps)
    new_input_json["decay_rate"] = decay_rate_new
    logging.debug(f"numb_steps: {numb_steps}")
    logging.debug(f"decay_rate: {decay_rate_new}")
    logging.debug(f"training_json - numb_steps / decay_rate: {training_json['numb_steps']} / {training_json['decay_rate']}")
    logging.debug(f"new_input_json - numb_steps / decay_rate: {new_input_json['numb_steps']} / {new_input_json['decay_rate']}")

    del decay_steps, numb_steps, decay_rate_new

    training_input_json["training"]["numb_steps"] = training_json["numb_steps"]
    training_input_json["learning_rate"]["decay_steps"] = training_json["decay_steps"]
    training_input_json["learning_rate"]["stop_lr"] = training_json["stop_lr"]

    # Set frozen/compressed bool !
    training_json = {
        **training_json,
        "is_locked": True,
        "is_launched": False,
        "is_checked": False,
        "is_frozen": False,
        "is_compressed": False,
    }

    # Rsync data to local data
    localdata_path = Path(".").resolve() / "data"
    localdata_path.mkdir(exist_ok=True)
    for it_datasets_training in datasets_training:
        subprocess.run(
            [
                "rsync",
                "-a",
                f"{training_path}/{it_datasets_training.rsplit('/', 1)[0]}",
                str(localdata_path),
            ]
        )
    del it_datasets_training, localdata_path, datasets_training

    # Change some inside output
    training_input_json["training"]["disp_file"] = "lcurve.out"
    training_input_json["training"]["save_ckpt"] = "model.ckpt"

    logging.debug(f"training_json: {training_json}")
    logging.debug(f"input_json: {input_json}")
    logging.debug(f"new_input_json: {new_input_json}")
    logging.debug(f"default_json: {default_json}")
    logging.debug(f"prevtraining_json: {prevtraining_json}")

    # Create the inputs/jobfiles for each NNP with random SEED

    # Walltime
    if "s_per_step" in input_json and input_json['s_per_step'] > 0:
        walltime_approx_s = int(np.ceil((training_json['numb_steps'] * input_json['s_per_step'])))
        logging.debug(f"s_per_step: {input_json['s_per_step']}")
    elif "s_per_step" in prevtraining_json:
        walltime_approx_s = int(np.ceil((training_json['numb_steps'] * (prevtraining_json['s_per_step'] * 1.50))))
        logging.debug(f"s_per_step: {prevtraining_json['s_per_step']}")
    else:
        walltime_approx_s = int(np.ceil((training_json["numb_steps"] * default_json['s_per_step'])))
        logging.debug(f"s_per_step: {default_json['s_per_step']}")
    # Set it to the input as -1, so the user knows it can be used but use auto
    new_input_json['s_per_step'] = -1
    logging.debug(f"walltime_approx_s: {walltime_approx_s}")

    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_path = Path(".").resolve() / str(it_nnp)
        local_path.mkdir(exist_ok=True)
        check_directory(local_path)

        random.seed()
        random_0_1000 = random.randrange(0, 1000)
        if training_json["deepmd_model_type_descriptor"] == "se_ar":
            training_input_json["model"]["descriptor"]["a"]["seed"] = int(
                str(it_nnp) + str(random_0_1000) + padded_curr_iter
            )
            training_input_json["model"]["descriptor"]["r"]["seed"] = int(
                str(it_nnp) + str(random_0_1000) + padded_curr_iter
            )
        else:
            training_input_json["model"]["descriptor"]["seed"] = int(
                str(it_nnp) + str(random_0_1000) + padded_curr_iter
            )

        training_input_json["model"]["fitting_net"]["seed"] = int(
            str(it_nnp) + str(random_0_1000) + padded_curr_iter
        )

        training_input_json["training"]["seed"] = int(
            str(it_nnp) + str(random_0_1000) + padded_curr_iter
        )

        training_input_json_fpath = Path(str(it_nnp) + "/training.json").resolve()

        write_json_file(training_input_json, training_input_json_fpath, False)

        slurm_file = replace_in_slurm_file_general(
            slurm_file_master,
            machine_spec,
            walltime_approx_s,
            machine_walltime_format,
            training_json["job_email"],
        )

        slurm_file = replace_substring_in_list_of_strings(
            slurm_file, "_R_DEEPMD_VERSION_", str(training_json["deepmd_model_version"])
        )
        write_list_of_strings_to_file(
            local_path / f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh",
            slurm_file,
        )
        del slurm_file, local_path, training_input_json_fpath, random_0_1000

    del it_nnp, walltime_approx_s, training_input_json

    # Dump the dicts
    logging.info(f"-" * 88)
    write_json_file(config_json, (control_path / "config.json"))
    write_json_file(
        training_json, (control_path / f"training_{padded_curr_iter}.json")
    )
    backup_and_overwrite_json_file(new_input_json, (current_path / input_fn))

    logging.info(f"-" * 88)
    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a success!")

    # Cleaning
    del control_path
    del data_path
    del input_json, default_json, default_present, new_input_json
    del config_json
    del curr_iter, padded_curr_iter
    del training_json
    del machine, machine_spec, machine_walltime_format, machine_launch_command
    del slurm_file_master
    del training_path, current_path

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "preparation",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
