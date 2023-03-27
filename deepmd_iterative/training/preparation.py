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
from deepmd_iterative.common.filesystem import (
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
from deepmd_iterative.training.utils import (
    calculate_decay_rate,
    calculate_decay_steps,
    check_initial_datasets,
    validate_deepmd_config,
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

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)
    
    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)
    logging.debug(f"curr_iter, padded_curr_iter: {curr_iter}, {padded_curr_iter}")

    # Load the default config (JSON)
    default_config = load_default_json_file(deepmd_iterative_path / "assets" / "default_config.json")[current_step]
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
    jobs_path = deepmd_iterative_path / "assets" / "jobs" / current_step

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

    # Check if we can continue
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
    master_job_file = file_to_list_of_strings(
        jobs_path / f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh"
    )
    del jobs_path
    logging.debug(f"master_job_file : {master_job_file [0:5]}, {master_job_file [-5:-1]}")

    # Check DeePMD version
    validate_deepmd_config(training_config)

    # Check if the default input json file exists
    dp_train_input_path = (
        training_path
        / "files"
        / (
            f"dptrain_{training_config['deepmd_model_version']}_{training_config['deepmd_model_type_descriptor']}.json"
        )
    ).resolve()
    dp_train_input = load_json_file(dp_train_input_path)
    main_config["type_map"] = {}
    main_config["type_map"] = dp_train_input["model"]["type_map"]
    del dp_train_input_path
    logging.debug(f"dp_train_input: {dp_train_input}")
    logging.debug(f"main_config: {main_config}")

    # Check the initial sets json file
    initial_datasets_info = check_initial_datasets(training_path)
    logging.debug(f"initial_datasets_info: {initial_datasets_info}")

    # Let us find what is in data
    data_path = training_path / "data"
    check_directory(data_path)

    # This is building the datasets (roughly 200 lines)
    # TODO later
    subsystems = []
    extra_datasets = []
    validation_datasets = []
    for data_dir in data_path.iterdir():
        if data_dir.is_dir():
            # Escape initial/extra sets, because initial get added first and extra as last, and also escape init_
            # not in initial_json (in case of removal)
            if (
                data_dir.name not in initial_datasets_info.keys()
                and "extra_" != data_dir.name[:6]
                and "init_" != data_dir.name[:5]
            ):
                # Escape test sets
                if "test_" != data_dir.name[:5]:
                    # Escape if set iter is superior as iter, it is only for reprocessing old stuff.
                    try:
                        if (
                            int(data_dir.name.rsplit("_", 1)[-1])
                            <= curr_iter
                        ):
                            subsystems.append(data_dir.name.rsplit("_", 1)[0])
                    # TODO: Better except clause
                    except:
                        pass
                else:
                    validation_datasets.append(data_dir.name)
            # Get the extra sets !
            elif "extra_" == data_dir.name[:6]:
                extra_datasets.append(data_dir.name)
    del data_dir

    # Training sets list construction
    dp_train_input_datasets = []
    training_datasets = []

    # Initial
    initial_count = 0
    if training_config["use_initial_datasets"]:
        for it_datasets_initial_json in initial_datasets_info.keys():
            if (data_path / it_datasets_initial_json).is_dir():
                dp_train_input_datasets.append(
                    f"{(Path(data_path.parts[-1]) / it_datasets_initial_json / '_')}"[
                        :-1
                    ]
                )
                training_datasets.append(it_datasets_initial_json)
                initial_count += initial_datasets_info[it_datasets_initial_json]

        del it_datasets_initial_json
    del initial_datasets_info

    # This trick remove duplicates from list via set
    subsystems = list(set(subsystems))
    subsystems = [i for i in subsystems if i not in main_config["subsys_nr"]]
    subsystems = [
        i
        for i in subsystems
        if i not in [zzz + "-disturbed" for zzz in main_config["subsys_nr"]]
    ]
    subsystems = sorted(subsystems)
    main_config["subsys_r"] = subsystems
    del subsystems

    # TODO As function
    # Non-Reactive (aka subsys_nr in the initialization first) && all the others are REACTIVE !
    # Total and what is added just for this iteration
    added_nr_count = 0
    added_r_count = 0
    added_nr_iter_count = 0
    added_r_iter_count = 0

    if curr_iter > 0:
        for iteration in np.arange(1, curr_iter + 1):
            padded_iteration = str(iteration).zfill(3)
            try:
                for subsys_nr in main_config["subsys_nr"]:
                    if (data_path / f"{subsys_nr}_{padded_iteration}").is_dir():
                        dp_train_input_datasets.append(
                            f"{(Path(data_path.parts[-1]) / (subsys_nr+'_'+padded_iteration) / '_')}"[:-1]
                        )
                        training_datasets.append(
                            f"{subsys_nr}_{padded_iteration}"
                        )
                        added_nr_count += np.load(
                                    data_path
                                    / f"{subsys_nr}_{padded_iteration}"
                                    / "set.000"
                                    / "box.npy"
                            ).shape[0]
                        if iteration == curr_iter:
                            added_nr_iter_count += np.load(
                                        data_path
                                        / f"{subsys_nr}_{padded_iteration}"
                                        / "set.000"
                                        / "box.npy"
                                ).shape[0]
                del subsys_nr
            except (KeyError, NameError):
                pass
            try:
                for subsys_disturbed in [
                    zzz + "-disturbed" for zzz in main_config["subsys_nr"]
                ]:
                    if (data_path / f"{subsys_disturbed}_{padded_iteration}").is_dir():
                        dp_train_input_datasets.append(
                            f"{(Path(data_path.parts[-1]) / (subsys_disturbed+'_'+padded_iteration) / '_')}"[
                                :-1
                            ]
                        )
                        training_datasets.append(
                            f"{subsys_disturbed}_{padded_iteration}"
                        )
                        added_nr_count += np.load(
                                    data_path
                                    / f"{subsys_disturbed}_{padded_iteration}"
                                    / "set.000"
                                    / "box.npy"
                            ).shape[0]
                        if iteration == curr_iter:
                            added_nr_iter_count += np.load(
                                        data_path
                                        / f"{subsys_disturbed}_{padded_iteration}"
                                        / "set.000"
                                        / "box.npy"
                                ).shape[0]
                del subsys_disturbed
            except (KeyError, NameError):
                pass
            try:
                for subsys_r in main_config["subsys_r"]:
                    if (data_path / f"{subsys_r}_{padded_iteration}").is_dir():
                        dp_train_input_datasets.append(
                            f"{(Path(data_path.parts[-1]) / (subsys_r+'_'+padded_iteration) / '_')}"[
                                :-1
                            ]
                        )
                        training_datasets.append(
                            f"{subsys_r}_{padded_iteration}"
                        )
                        added_nr_count = (
                            added_nr_count
                            + np.load(
                                    data_path
                                    / f"{subsys_r}_{padded_iteration}"
                                    / "set.000"
                                    / "box.npy"
                            ).shape[0]
                        )
                        if iteration == curr_iter:
                            added_nr_iter_count += np.load(
                                        data_path
                                        / f"{subsys_r}_{padded_iteration}"
                                        / "set.000"
                                        / "box.npy"
                                ).shape[0]
                del subsys_r
            except (KeyError, NameError):
                pass
        del iteration, padded_iteration
    # TODO End of As function

    # Finally the extra sets !
    extra_count = 0
    if training_config["use_extra_datasets"]:
        main_config["extra_datasets"] = extra_datasets
        del extra_datasets
        for extra_dataset in main_config["extra_datasets"]:
            dp_train_input_datasets.append(f"{(Path(data_path.parts[-1]) / extra_dataset / '_')}"[:-1])
            training_datasets.append(extra_dataset)
            extra_count += np.load(data_path / extra_dataset / "set.000" / "box.npy").shape[0]
        del extra_dataset
    else:
        del extra_datasets

    # Total
    trained_count = initial_count + added_nr_count + added_r_count + extra_count
    logging.debug(f"trained_count: {trained_count} = {initial_count} + {added_nr_count} + {added_r_count} + {extra_count}")
    logging.debug(f"dp_train_input_datasets: {dp_train_input_datasets}")

    # Update the inputs with the sets
    dp_train_input["training"]["training_data"]["systems"] = dp_train_input_datasets

    # Update the training JSON
    training_config = {
        **training_config,
        "training_datasets": training_datasets,
        "trained_count": trained_count,
        "initial_count": initial_count,
        "added_nr_count": added_nr_count,
        "added_r_count": added_r_count,
        "added_nr_iter_count": added_nr_iter_count,
        "added_r_iter_count": added_r_iter_count,
        "extra_count": extra_count,
    }
    logging.debug(f"training_config: {training_config}")

    del training_datasets
    del trained_count, initial_count, extra_count
    del added_nr_count, added_r_count, added_nr_iter_count, added_r_iter_count

    # Here calculate the parameters
    # decay_steps it auto-recalculated as funcion of trained_count
    logging.debug(f"training_config - decay_steps: {training_config['decay_steps']}")
    logging.debug(f"current_config - decay_steps: {current_config['decay_steps']}")
    if not training_config["decay_steps_fixed"]:
        decay_steps = calculate_decay_steps(training_config["trained_count"], training_config["decay_steps"])
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
    for dp_train_input_dataset in dp_train_input_datasets:
        subprocess.run(
            [
                "rsync",
                "-a",
                f"{training_path / (dp_train_input_dataset.rsplit('/', 1)[0])}",
                f"{localdata_path}",
            ]
        )
    del dp_train_input_dataset, localdata_path, dp_train_input_datasets

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

    for nnp in range(1, main_config["nnp_count"] + 1):
        local_path = Path(".").resolve() / f"{nnp}"
        local_path.mkdir(exist_ok=True)
        check_directory(local_path)

        random.seed()
        random_0_1000 = random.randrange(0, 1000)

        if training_config["deepmd_model_type_descriptor"] == "se_ar":
            dp_train_input["model"]["descriptor"]["a"]["seed"] = int(f"{nnp}{random_0_1000}{padded_curr_iter}")
            dp_train_input["model"]["descriptor"]["r"]["seed"] = int(f"{nnp}{random_0_1000}{padded_curr_iter}")
        else:
            dp_train_input["model"]["descriptor"]["seed"] = int(f"{nnp}{random_0_1000}{padded_curr_iter}")
        dp_train_input["model"]["fitting_net"]["seed"] = int(f"{nnp}{random_0_1000}{padded_curr_iter}")
        dp_train_input["training"]["seed"] = int(f"{nnp}{random_0_1000}{padded_curr_iter}")

        dp_train_input_file = (Path(f"{nnp}") / "training.json").resolve()

        write_json_file(dp_train_input, dp_train_input_file, False)

        job_file = replace_in_slurm_file_general(
            master_job_file ,
            machine_spec,
            walltime_approx_s,
            machine_walltime_format,
            training_config["job_email"],
        )

        job_file = replace_substring_in_list_of_strings(job_file, "_R_DEEPMD_VERSION_", f"{training_config['deepmd_model_version']}")
        write_list_of_strings_to_file(
            local_path / f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh",
            job_file,
        )
        del job_file, local_path, dp_train_input_file, random_0_1000

    del nnp, walltime_approx_s, dp_train_input

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
    del master_job_file

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "preparation",
            Path(sys.argv[1]),
            fake_machine = sys.argv[2],
            user_config_filename = sys.argv[3],
        )
    else:
        pass
