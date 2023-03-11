from pathlib import Path
import logging
import sys
import copy
import random
import subprocess

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
    set_training_config,
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
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine = None,
    user_config_filename: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if correct folder
    validate_step_folder(current_step)

    # Get iteration
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Load the default config (JSON)
    default_config = load_default_json_file(deepmd_iterative_path / "data" / "default_config.json")[current_step]
    default_config_present = bool(default_config)
    logging.debug(f"default_config: {default_config}")
    logging.debug(f"default_config_present: {default_config_present}")

    # Load the user config (JSON)
    if (current_path / user_config_filename).is_file():
        user_config = load_json_file((current_path / user_config_filename))
    else:
        user_config = {}
    user_config_present = bool(user_config)
    logging.debug(f"user_config: {user_config}")
    logging.debug(f"user_config_present: {user_config_present}")

    # Make a deepcopy
    current_config = copy.deepcopy(user_config)

    # Get control path and load the main config (JSON)
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))

    # Get extra needed paths
    jobs_path = deepmd_iterative_path / "data" / "jobs" / current_step

    # Load the previous training config (JSON)
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_training_config = load_json_file((control_path / f"training_{padded_prev_iter}.json"))
    else:
        previous_training_config = {}

    # Get the machine keyword (input override previous training override default_config)
    # And update the new input
    user_machine_keyword = get_machine_keyword(user_config, previous_training_config, default_config)
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    current_config["user_machine_keyword"] = user_machine_keyword
    logging.debug(f"current_config: {current_config}")
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
        labeling_config = load_json_file(
            (control_path / f"labeling_{padded_curr_iter}.json")
        )
        if not labeling_config["is_extracted"]:
            logging.error("Lock found. Run/Check first: labeling extract.")
            logging.error("Aborting...")
            return 1
    else:
        labeling_config = {}

    # Create the training JSON file (and set everything)
    # Priority: input > previous > default
    training_config, current_config = set_training_config(
        user_config,
        previous_training_config,
        default_config,
        current_config,
    )
    logging.debug(f"training_config: {training_config}")
    logging.debug(f"current_config: {current_config}")

    # Set additional machine-related parameters in the training JSON file (not need in the input)
    training_config = {
        **training_config,
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
        error_msg=f"No SLURM file present for {current_step.capitalize()} / {current_phase.capitalize()} on this machine.",
    )
    slurm_file_master = file_to_list_of_strings(
        jobs_path / f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh"
    )
    del jobs_path
    logging.debug(f"slurm_file_master: {slurm_file_master[0:5]}, {slurm_file_master[-5:-1]}")

    # TODO: Maybe as function / parameters file for later
    # Check DeePMD version
    if training_config["deepmd_model_version"] not in [2.0, 2.1]:
        logging.critical(
            f"Invalid deepmd model version (2.0 or 2.1): {training_config['deepmd_model_version']}."
        )
        logging.critical("Aborting...")
        return 1

    # Check DeePMD descriptor type
    if training_config["deepmd_model_type_descriptor"] not in ["se_e2_a"]:
        logging.critical(
            f"Invalid deepmd type descriptor (se_e2_a): {training_config['deepmd_model_type_descriptor']}."
        )
        logging.critical("Aborting...")
        return 1

    # Check mismatch between machine/arch_name/arch and DeePMD
    if training_config["deepmd_model_version"] < 2.0:
        logging.critical("Only version >= 2.0 on Jean Zay!")
        logging.critical("Aborting...")
        return 1
    if (
        training_config["deepmd_model_version"] < 2.1
        and training_config["arch_name"] == "a100"
    ):
        logging.critical("Only version >= 2.1 on Jean Zay A100!")
        logging.critical("Aborting...")
        return 1


    # Check if the default input json file exists
    input_file_fpath = (
        training_path
        / "files"
        / (
            f"dptrain_{training_config['deepmd_model_version']}_{training_config['deepmd_model_type_descriptor']}.json"
        )
    ).resolve()
    dp_train_input = load_json_file(input_file_fpath)
    main_config["type_map"] = {}
    main_config["type_map"] = dp_train_input["model"]["type_map"]
    del input_file_fpath
    logging.debug(f"dp_train_input: {dp_train_input}")
    logging.debug(f"main_config: {main_config}")

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
    if training_config["use_initial_datasets"]:
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
    subsys_name = [i for i in subsys_name if i not in main_config["subsys_nr"]]
    subsys_name = [
        i
        for i in subsys_name
        if i not in [zzz + "-disturbed" for zzz in main_config["subsys_nr"]]
    ]
    subsys_name = sorted(subsys_name)
    main_config["subsys_r"] = subsys_name
    del subsys_name

    # TODO Function
    if curr_iter > 0:
        for it_iteration in np.arange(1, curr_iter + 1):
            it_iteration_zfill = str(it_iteration).zfill(3)
            try:
                for system_it in main_config["subsys_nr"]:
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
                    zzz + "-disturbed" for zzz in main_config["subsys_nr"]
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
                for system_it in main_config["subsys_r"]:
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
    if training_config["use_extra_datasets"]:
        main_config["datasets_extra"] = datasets_extra
        del datasets_extra
        for it_datasets_extra in main_config["datasets_extra"]:
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
    dp_train_input["training"]["training_data"]["systems"] = datasets_training

    # Update the training JSON
    training_config = {
        **training_config,
        "training_data": datasets_training_json,
        "nb_trained": nb_trained,
        "nb_initial": nb_initial,
        "nb_added_nr": nb_added_nr,
        "nb_added_r": nb_added_r,
        "nb_added_nr_iter": nb_added_nr_iter,
        "nb_added_r_iter": nb_added_r_iter,
        "nb_extra": nb_extra,
    }
    logging.debug(f"training_config: {training_config}")

    del datasets_training_json
    del nb_trained, nb_initial, nb_extra
    del nb_added_nr, nb_added_r, nb_added_nr_iter, nb_added_r_iter

    # Here calculate the parameters

    # decay_steps it auto-recalculated as funcion of nb_trained
    logging.debug(f"training_config - decay_steps: {training_config['decay_steps']}")
    logging.debug(f"current_config - decay_steps: {current_config['decay_steps']}")
    if not training_config["decay_steps_fixed"]:
        decay_steps = calculate_decay_steps(training_config["nb_trained"], training_config["decay_steps"])
        logging.debug(f"Recalculating decay_steps")
        # Update the training JSON and the new input JSON:
        training_config["decay_steps"] = decay_steps
        current_config["decay_steps"] = decay_steps
    else:
        decay_steps = training_config["decay_steps"]
    logging.debug(f"decay_steps: {decay_steps}")
    logging.debug(f"training_config - decay_steps: {training_config['decay_steps']}")
    logging.debug(f"current_config - decay_steps: {current_config['decay_steps']}")

    # numb_steps and decay_rate
    logging.debug(f"training_config - numb_steps / decay_rate: {training_config['numb_steps']} / {training_config['decay_rate']}")
    logging.debug(f"current_config - numb_steps / decay_rate: {current_config['numb_steps']} / {current_config['decay_rate']}")
    numb_steps = training_config["numb_steps"]
    decay_rate_new = calculate_decay_rate(
        numb_steps,
        training_config["start_lr"],
        training_config["stop_lr"],
        training_config["decay_steps"],
    )
    while decay_rate_new < training_config["decay_rate"]:
        numb_steps = numb_steps + 10000
        decay_rate_new = calculate_decay_rate(
            numb_steps,
            training_config["start_lr"],
            training_config["stop_lr"],
            training_config["decay_steps"],
        )
    # Update the training JSON and the new input JSON:
    training_config["numb_steps"] = int(numb_steps)
    training_config["decay_rate"] = decay_rate_new
    current_config["numb_steps"] = int(numb_steps)
    current_config["decay_rate"] = decay_rate_new
    logging.debug(f"numb_steps: {numb_steps}")
    logging.debug(f"decay_rate: {decay_rate_new}")
    logging.debug(f"training_config - numb_steps / decay_rate: {training_config['numb_steps']} / {training_config['decay_rate']}")
    logging.debug(f"current_config - numb_steps / decay_rate: {current_config['numb_steps']} / {current_config['decay_rate']}")

    del decay_steps, numb_steps, decay_rate_new

    dp_train_input["training"]["numb_steps"] = training_config["numb_steps"]
    dp_train_input["learning_rate"]["decay_steps"] = training_config["decay_steps"]
    dp_train_input["learning_rate"]["stop_lr"] = training_config["stop_lr"]

    # Set frozen/compressed bool !
    training_config = {
        **training_config,
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
    dp_train_input["training"]["disp_file"] = "lcurve.out"
    dp_train_input["training"]["save_ckpt"] = "model.ckpt"

    logging.debug(f"training_config: {training_config}")
    logging.debug(f"user_config: {user_config}")
    logging.debug(f"current_config: {current_config}")
    logging.debug(f"default_config: {default_config}")
    logging.debug(f"previous_training_config: {previous_training_config}")

    # Create the inputs/jobfiles for each NNP with random SEED

    # Walltime
    if "s_per_step" in user_config and user_config['s_per_step'] > 0:
        walltime_approx_s = int(np.ceil((training_config['numb_steps'] * user_config['s_per_step'])))
        logging.debug(f"s_per_step: {user_config['s_per_step']}")
    elif "s_per_step" in previous_training_config:
        walltime_approx_s = int(np.ceil((training_config['numb_steps'] * (previous_training_config['s_per_step'] * 1.50))))
        logging.debug(f"s_per_step: {previous_training_config['s_per_step']}")
    else:
        walltime_approx_s = int(np.ceil((training_config["numb_steps"] * default_config['s_per_step'])))
        logging.debug(f"s_per_step: {default_config['s_per_step']}")
    # Set it to the input as -1, so the user knows it can be used but use auto
    current_config['s_per_step'] = -1
    logging.debug(f"walltime_approx_s: {walltime_approx_s}")

    for it_nnp in range(1, main_config["nb_nnp"] + 1):
        local_path = Path(".").resolve() / str(it_nnp)
        local_path.mkdir(exist_ok=True)
        check_directory(local_path)

        random.seed()
        random_0_1000 = random.randrange(0, 1000)
        if training_config["deepmd_model_type_descriptor"] == "se_ar":
            dp_train_input["model"]["descriptor"]["a"]["seed"] = int(
                str(it_nnp) + str(random_0_1000) + padded_curr_iter
            )
            dp_train_input["model"]["descriptor"]["r"]["seed"] = int(
                str(it_nnp) + str(random_0_1000) + padded_curr_iter
            )
        else:
            dp_train_input["model"]["descriptor"]["seed"] = int(
                str(it_nnp) + str(random_0_1000) + padded_curr_iter
            )

        dp_train_input["model"]["fitting_net"]["seed"] = int(
            str(it_nnp) + str(random_0_1000) + padded_curr_iter
        )

        dp_train_input["training"]["seed"] = int(
            str(it_nnp) + str(random_0_1000) + padded_curr_iter
        )

        dp_train_input_file = Path(str(it_nnp) + "/training.json").resolve()

        write_json_file(dp_train_input, dp_train_input_file, False)

        slurm_file = replace_in_slurm_file_general(
            slurm_file_master,
            machine_spec,
            walltime_approx_s,
            machine_walltime_format,
            training_config["job_email"],
        )

        slurm_file = replace_substring_in_list_of_strings(
            slurm_file, "_R_DEEPMD_VERSION_", str(training_config["deepmd_model_version"])
        )
        write_list_of_strings_to_file(
            local_path / f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh",
            slurm_file,
        )
        del slurm_file, local_path, dp_train_input_file, random_0_1000

    del it_nnp, walltime_approx_s, dp_train_input

    # Dump the dicts
    logging.info(f"-" * 88)
    write_json_file(main_config, (control_path / "config.json"))
    write_json_file(
        training_config, (control_path / f"training_{padded_curr_iter}.json")
    )
    backup_and_overwrite_json_file(current_config, (current_path / user_config_filename))

    logging.info(f"-" * 88)
    logging.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")

    # Cleaning
    del current_path, control_path, training_path, data_path
    del default_config, default_config_present, user_config, user_config_present, user_config_filename
    del main_config, current_config, training_config, previous_training_config, labeling_config
    del curr_iter, padded_curr_iter
    del machine, machine_spec, machine_walltime_format, machine_launch_command
    del slurm_file_master

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "preparation",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_config_filename=sys.argv[3],
        )
    else:
        pass
