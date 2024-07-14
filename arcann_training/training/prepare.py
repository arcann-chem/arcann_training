"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/07/14
"""

# Standard library modules
import logging
import sys
from pathlib import Path
from copy import deepcopy
import random
import subprocess

# Non-standard library imports
import numpy as np

# Local imports
from arcann_training.common.check import validate_step_folder
from arcann_training.common.filesystem import check_directory
from arcann_training.common.json import backup_and_overwrite_json_file, get_key_in_dict, load_default_json_file, load_json_file, write_json_file, replace_values_by_key_name
from arcann_training.common.list import replace_substring_in_string_list, string_list_to_textfile, textfile_to_string_list
from arcann_training.common.machine import get_machine_keyword, get_machine_spec_for_step
from arcann_training.common.slurm import replace_in_slurm_file_general
from arcann_training.training.utils import calculate_decay_rate, calculate_decay_steps, check_initial_datasets, validate_deepmd_config, generate_training_json


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_input_json_filename: str = "input.json",
):
    # Get the logger
    arcann_logger = logging.getLogger("ArcaNN")

    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    arcann_logger.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}.")
    arcann_logger.debug(f"Current path :{current_path}")
    arcann_logger.debug(f"Training path: {training_path}")
    arcann_logger.debug(f"Program path: {deepmd_iterative_path}")
    arcann_logger.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)
    arcann_logger.debug(f"curr_iter, padded_curr_iter: {curr_iter}, {padded_curr_iter}")

    # Load the default input JSON
    default_input_json = load_default_json_file(deepmd_iterative_path / "assets" / "default_config.json")[current_step]
    default_input_json_present = bool(default_input_json)
    if default_input_json_present and not (current_path / "default_input.json").is_file():
        write_json_file(default_input_json, (current_path / "default_input.json"), read_only=True)
    arcann_logger.debug(f"default_input_json: {default_input_json}")
    arcann_logger.debug(f"default_input_json_present: {default_input_json_present}")

    # Load the user input JSON
    if (current_path / user_input_json_filename).is_file():
        user_input_json = load_json_file((current_path / user_input_json_filename))
    else:
        user_input_json = {}
    user_input_json_present = bool(user_input_json)
    arcann_logger.debug(f"user_input_json: {user_input_json}")
    arcann_logger.debug(f"user_input_json_present: {user_input_json_present}")

    # Make a deepcopy of it to create the used input JSON
    current_input_json = deepcopy(user_input_json)

    # Get control path and load the main JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))

    # Load the previous training JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_training_json = load_json_file((control_path / f"training_{padded_prev_iter}.json"))
        del prev_iter, padded_prev_iter
    else:
        previous_training_json = {}

    # Get the machine keyword (Priority: user > previous > default)
    # And update the merged input JSON
    user_machine_keyword = get_machine_keyword(current_input_json, previous_training_json, default_input_json, "train")
    # Set it to None if bool, because: get_machine_spec_for_step needs None
    user_machine_keyword = None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    arcann_logger.debug(f"user_machine_keyword: {user_machine_keyword}")

    # From the keyword (or default), get the machine spec (or for the fake one)
    (
        machine,
        machine_walltime_format,
        machine_job_scheduler,
        machine_launch_command,
        machine_max_jobs,
        machine_max_array_size,
        user_machine_keyword,
        machine_spec,
    ) = get_machine_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "training",
        fake_machine,
        user_machine_keyword,
    )
    arcann_logger.debug(f"machine: {machine}")
    arcann_logger.debug(f"machine_walltime_format: {machine_walltime_format}")
    arcann_logger.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    arcann_logger.debug(f"machine_launch_command: {machine_launch_command}")
    arcann_logger.debug(f"machine_max_jobs: {machine_max_jobs}")
    arcann_logger.debug(f"machine_max_array_size: {machine_max_array_size}")
    arcann_logger.debug(f"user_machine_keyword: {user_machine_keyword}")
    arcann_logger.debug(f"machine_spec: {machine_spec}")

    current_input_json["user_machine_keyword_train"] = user_machine_keyword
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    if fake_machine is not None:
        arcann_logger.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        arcann_logger.info(f"Machine identified: '{machine}'.")
    del fake_machine

    # Check if we can continue
    if curr_iter > 0:
        labeling_json = load_json_file((control_path / f"labeling_{padded_curr_iter}.json"))
        if not labeling_json["is_extracted"]:
            arcann_logger.error(f"Lock found. Please execute 'labeling extract' first.")
            arcann_logger.error(f"Aborting...")
            return 1
        # exploration_json = load_json_file((control_path / f"exploration_{padded_curr_iter}.json"))
    else:
        # exploration_json = {}
        labeling_json = {}

    if "deepmd_model_version" not in user_input_json:
        dptrain_list = []
        for file in (current_path.parent / "user_files").iterdir():
            if file.suffix != ".json":
                continue
            if "dptrain" not in file.stem:
                continue
            dptrain_list.append(file)
        arcann_logger.debug(f"dptrain_list: {dptrain_list}")
        del file

        if not dptrain_list:
            arcann_logger.error(f"No dptrain_DEEPMDVERSION.json files found in {(current_path.parent / 'user_files')}")
            arcann_logger.error(f"Aborting...")
            return 1

        dptrain_max_version = 0
        for dptrain in dptrain_list:
            dptrain_max_version = max(dptrain_max_version, float(dptrain.stem.split("_")[-1]))
        del dptrain

        arcann_logger.debug(f"dptrain_max_version: {dptrain_max_version}")
        current_input_json["deepmd_model_version"] = dptrain_max_version
        del dptrain_list, dptrain_max_version

    # Generate/update both the training JSON and the merged input JSON
    # Priority: user/current > previous > default
    training_json, current_input_json = generate_training_json(current_input_json, previous_training_json, default_input_json)
    arcann_logger.info(f"Using DeePMD version: {current_input_json['deepmd_model_version']}")
    arcann_logger.debug(f"training_json: {training_json}")
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    # Check if the job file exists
    job_file_name = f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh"
    if (current_path.parent / "user_files" / job_file_name).is_file():
        master_job_file = textfile_to_string_list(current_path.parent / "user_files" / job_file_name)
    else:
        arcann_logger.error(f"No JOB file provided for '{current_step.capitalize()} / {current_phase.capitalize()}' for this machine.")
        arcann_logger.error(f"Aborting...")
        return 1

    arcann_logger.debug(f"master_job_file: {master_job_file[0:5]}, {master_job_file[-5:-1]}")
    current_input_json["job_email"] = get_key_in_dict("job_email", user_input_json, previous_training_json, default_input_json)
    del job_file_name

    # Check DeePMD version
    validate_deepmd_config(training_json)

    # Check if the default input json file exists
    dp_train_input_path = (training_path / "user_files" / f"dptrain_{training_json['deepmd_model_version']}.json").resolve()

    dp_train_input = load_json_file(dp_train_input_path)
    if "type_map" not in main_json:
        type_map = []
        for element in main_json["properties"]:
            type_map.append(main_json["properties"][element]["symbol"])
        main_json["type_map"] = type_map

    # Make sure they are the same
    if dp_train_input["model"]["type_map"] != main_json["type_map"]:
        arcann_logger.error(f"Type map in {dp_train_input_path} does not match the one in config.json.")
        arcann_logger.error(f"Aborting...")
        return 1

    # main_json["type_map"] = {}
    # main_json["type_map"] = dp_train_input["model"]["type_map"]
    del dp_train_input_path
    arcann_logger.debug(f"dp_train_input: {dp_train_input}")
    arcann_logger.debug(f"main_json: {main_json}")

    # Check the initial sets json file
    initial_datasets_info = check_initial_datasets(training_path)
    arcann_logger.debug(f"initial_datasets_info: {initial_datasets_info}")

    # Let us find what is in data
    data_path = training_path / "data"
    check_directory(data_path)

    # This is building the datasets (roughly 200 lines)
    # TODO later
    systems = []
    extra_datasets = []
    validation_datasets = []
    for data_dir in data_path.iterdir():
        if data_dir.is_dir():
            # Escape initial/extra sets, because initial get added first and extra as last, and also escape init_
            # not in initial_json (in case of removal)
            if data_dir.name not in initial_datasets_info.keys() and "extra_" != data_dir.name[:6] and "init_" != data_dir.name[:5]:
                # Escape test sets
                if "test_" != data_dir.name[:5]:
                    # Escape if set iter is superior as iter, it is only for reprocessing old stuff
                    try:
                        if int(data_dir.name.rsplit("_", 1)[-1]) <= curr_iter:
                            systems.append(data_dir.name.rsplit("_", 1)[0])
                    # TODO Better except clause
                    except:
                        pass
                else:
                    validation_datasets.append(data_dir.name)
            # Get the extra sets !
            elif "extra_" == data_dir.name[:6]:
                extra_datasets.append(data_dir.name)
    del data_dir

    # TODO Implement validation dataset
    del validation_datasets

    # Training sets list construction
    dp_train_input_datasets = []
    training_datasets = []

    # Initial
    initial_count = 0
    if training_json["use_initial_datasets"]:
        for it_datasets_initial_json in initial_datasets_info.keys():
            if (data_path / it_datasets_initial_json).is_dir():
                dp_train_input_datasets.append(f"{(Path(data_path.parts[-1]) / it_datasets_initial_json / '_')}"[:-1])
                training_datasets.append(it_datasets_initial_json)
                initial_count += initial_datasets_info[it_datasets_initial_json]

        del it_datasets_initial_json
    del initial_datasets_info

    # This trick remove duplicates from list via set
    systems = list(set(systems))
    systems = [i for i in systems if i not in main_json["systems_auto"]]
    systems = [i for i in systems if i not in [zzz + "-disturbed" for zzz in main_json["systems_auto"]]]
    systems = sorted(systems)
    main_json["systems_adhoc"] = systems
    del systems

    # TODO As function
    # Automatic Systems (aka systems_auto in the initialization first) && all the others are not automated !
    # Total and what is added just for this iteration
    added_auto_count = 0
    added_adhoc_count = 0
    added_auto_iter_count = 0
    added_adhoc_iter_count = 0

    if curr_iter > 0:
        for iteration in np.arange(1, curr_iter + 1):
            padded_iteration = str(iteration).zfill(3)
            try:
                for system_auto in main_json["systems_auto"]:
                    if (data_path / f"{system_auto}_{padded_iteration}").is_dir():
                        dp_train_input_datasets.append(f"{(Path(data_path.parts[-1]) / (system_auto+'_'+padded_iteration) / '_')}"[:-1])
                        training_datasets.append(f"{system_auto}_{padded_iteration}")
                        added_auto_count += np.load(data_path / f"{system_auto}_{padded_iteration}" / "set.000" / "box.npy").shape[0]
                        if iteration == curr_iter:
                            added_auto_iter_count += np.load(data_path / f"{system_auto}_{padded_iteration}" / "set.000" / "box.npy").shape[0]
                del system_auto
            except (KeyError, NameError):
                pass
            try:
                for system_auto_disturbed in [zzz + "-disturbed" for zzz in main_json["systems_auto"]]:
                    if (data_path / f"{system_auto_disturbed}_{padded_iteration}").is_dir():
                        dp_train_input_datasets.append(f"{(Path(data_path.parts[-1]) / (system_auto_disturbed+'_'+padded_iteration) / '_')}"[:-1])
                        training_datasets.append(f"{system_auto_disturbed}_{padded_iteration}")
                        added_auto_count += np.load(data_path / f"{system_auto_disturbed}_{padded_iteration}" / "set.000" / "box.npy").shape[0]
                        if iteration == curr_iter:
                            added_auto_iter_count += np.load(data_path / f"{system_auto_disturbed}_{padded_iteration}" / "set.000" / "box.npy").shape[0]
                del system_auto_disturbed
            except (KeyError, NameError):
                pass
            try:
                for system_adhoc in main_json["systems_adhoc"]:
                    if (data_path / f"{system_adhoc}_{padded_iteration}").is_dir():
                        dp_train_input_datasets.append(f"{(Path(data_path.parts[-1]) / (system_adhoc+'_'+padded_iteration) / '_')}"[:-1])
                        training_datasets.append(f"{system_adhoc}_{padded_iteration}")
                        added_auto_count = added_auto_count + np.load(data_path / f"{system_adhoc}_{padded_iteration}" / "set.000" / "box.npy").shape[0]
                        if iteration == curr_iter:
                            added_auto_iter_count += np.load(data_path / f"{system_adhoc}_{padded_iteration}" / "set.000" / "box.npy").shape[0]
                del system_adhoc
            except (KeyError, NameError):
                pass
        del iteration, padded_iteration
    # TODO End of As function

    # Finally the extra sets !
    extra_count = 0
    if training_json["use_extra_datasets"]:
        main_json["extra_datasets"] = extra_datasets
        del extra_datasets
        for extra_dataset in main_json["extra_datasets"]:
            dp_train_input_datasets.append(f"{(Path(data_path.parts[-1]) / extra_dataset / '_')}"[:-1])
            training_datasets.append(extra_dataset)
            extra_count += np.load(data_path / extra_dataset / "set.000" / "box.npy").shape[0]
        del extra_dataset
    else:
        del extra_datasets

    # Total
    trained_count = initial_count + added_auto_count + added_adhoc_count + extra_count
    arcann_logger.debug(f"trained_count: {trained_count} = {initial_count} + {added_auto_count} + {added_adhoc_count} + {extra_count}")
    arcann_logger.debug(f"dp_train_input_datasets: {dp_train_input_datasets}")

    # Update the inputs with the sets
    dp_train_input["training"]["training_data"]["systems"] = dp_train_input_datasets

    # Update the training JSON
    training_json = {
        **training_json,
        "training_datasets": training_datasets,
        "trained_count": trained_count,
        "initial_count": initial_count,
        "added_auto_count": added_auto_count,
        "added_adhoc_count": added_adhoc_count,
        "added_auto_iter_count": added_auto_iter_count,
        "added_adhoc_iter_count": added_adhoc_iter_count,
        "extra_count": extra_count,
    }
    arcann_logger.debug(f"training_json: {training_json}")

    del training_datasets
    del trained_count, initial_count, extra_count
    del added_auto_count, added_adhoc_count, added_auto_iter_count, added_adhoc_iter_count

    # Here calculate the parameters
    # decay_steps it auto-recalculated as funcion of trained_count
    arcann_logger.debug(f"training_json - decay_steps: {training_json['decay_steps']}")
    arcann_logger.debug(f"current_input_json - decay_steps: {current_input_json['decay_steps']}")
    if not training_json["decay_steps_fixed"]:
        decay_steps = calculate_decay_steps(training_json["trained_count"], training_json["decay_steps"])
        arcann_logger.debug(f"Recalculating decay_steps")
        # Update the training JSON and the merged input JSON
        training_json["decay_steps"] = decay_steps
        current_input_json["decay_steps"] = decay_steps
    else:
        decay_steps = training_json["decay_steps"]
    arcann_logger.debug(f"decay_steps: {decay_steps}")
    arcann_logger.debug(f"training_json - decay_steps: {training_json['decay_steps']}")
    arcann_logger.debug(f"current_input_json - decay_steps: {current_input_json['decay_steps']}")

    # numb_steps and decay_rate
    arcann_logger.debug(f"training_json - numb_steps / decay_rate: {training_json['numb_steps']} / {training_json['decay_rate']}")
    arcann_logger.debug(f"current_input_json - numb_steps / decay_rate: {current_input_json['numb_steps']} / {current_input_json['decay_rate']}")
    numb_steps = training_json["numb_steps"]
    decay_rate_new = calculate_decay_rate(numb_steps, training_json["start_lr"], training_json["stop_lr"], training_json["decay_steps"])
    while decay_rate_new < training_json["decay_rate"]:
        numb_steps = numb_steps + 10000
        decay_rate_new = calculate_decay_rate(numb_steps, training_json["start_lr"], training_json["stop_lr"], training_json["decay_steps"])
    # Update the training JSON and the merged input JSON
    training_json["numb_steps"] = int(numb_steps)
    training_json["decay_rate"] = decay_rate_new
    current_input_json["numb_steps"] = int(numb_steps)
    current_input_json["decay_rate"] = decay_rate_new
    arcann_logger.debug(f"numb_steps: {numb_steps}")
    arcann_logger.debug(f"decay_rate: {decay_rate_new}")
    arcann_logger.debug(f"training_json - numb_steps / decay_rate: {training_json['numb_steps']} / {training_json['decay_rate']}")
    arcann_logger.debug(f"current_input_json - numb_steps / decay_rate: {current_input_json['numb_steps']} / {current_input_json['decay_rate']}")

    del decay_steps, numb_steps, decay_rate_new

    dp_train_input["training"]["numb_steps"] = training_json["numb_steps"]
    dp_train_input["learning_rate"]["decay_steps"] = training_json["decay_steps"]
    dp_train_input["learning_rate"]["stop_lr"] = training_json["stop_lr"]

    # Set booleans in the training JSON
    training_json = {
        **training_json,
        "is_prepared": True,
        "is_launched": False,
        "is_checked": False,
        "is_freeze_launched": False,
        "is_frozen": False,
        "is_compress_launched": False,
        "is_compressed": False,
        "is_incremented": False,
    }

    # Rsync data to local data
    localdata_path = current_path / "data"
    localdata_path.mkdir(exist_ok=True)
    for dp_train_input_dataset in dp_train_input_datasets:
        subprocess.run(["rsync", "-a", f"{training_path / (dp_train_input_dataset.rsplit('/', 1)[0])}", f"{localdata_path}"])
    del dp_train_input_dataset, localdata_path, dp_train_input_datasets

    # Change some inside output
    dp_train_input["training"]["disp_file"] = "lcurve.out"
    dp_train_input["training"]["save_ckpt"] = "model.ckpt"

    arcann_logger.debug(f"training_json: {training_json}")
    arcann_logger.debug(f"user_input_json: {user_input_json}")
    arcann_logger.debug(f"current_input_json: {current_input_json}")
    arcann_logger.debug(f"default_input_json: {default_input_json}")
    arcann_logger.debug(f"previous_training_json: {previous_training_json}")

    # Create the inputs/jobfiles for each NNP with random SEED

    # Walltime
    if "job_walltime_train_h" in user_input_json and user_input_json["job_walltime_train_h"] > 0:
        walltime_approx_s = int(user_input_json["job_walltime_train_h"] * 3600)
        mean_s_per_step = walltime_approx_s / training_json["numb_steps"]
        arcann_logger.debug(f"job_walltime_train_h: {user_input_json['job_walltime_train_h']}")
    elif "mean_s_per_step" in user_input_json and user_input_json["mean_s_per_step"] > 0:
        walltime_approx_s = int(np.ceil((training_json["numb_steps"] * user_input_json["mean_s_per_step"])))
        mean_s_per_step = walltime_approx_s / training_json["numb_steps"]
        arcann_logger.debug(f"mean_s_per_step: {user_input_json['mean_s_per_step']}")
    else:
        if curr_iter == 0:
            # This is rounded up to the next hour
            walltime_approx_s = int(np.ceil(training_json["numb_steps"] * default_input_json["mean_s_per_step"] / 3600) * 3600)
            mean_s_per_step = walltime_approx_s / training_json["numb_steps"]
        else:
            walltime_approx_s = int(np.ceil(training_json["numb_steps"] * previous_training_json["mean_s_per_step"] * 1.5 / 3600) * 3600)
            mean_s_per_step = walltime_approx_s / training_json["numb_steps"]

    current_input_json["job_walltime_train_h"] = float(walltime_approx_s / 3600)
    current_input_json["mean_s_per_step"] = mean_s_per_step
    training_json["job_walltime_train_h"] = float(walltime_approx_s / 3600)
    training_json["mean_s_per_step"] = mean_s_per_step
    arcann_logger.debug(f"walltime_approx_s: {walltime_approx_s}")
    arcann_logger.debug(f"mean_s_per_step: {mean_s_per_step}")

    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"
        local_path.mkdir(exist_ok=True)
        check_directory(local_path)

        random.seed()
        random_0_1000 = random.randrange(0, 1000)

        replace_values_by_key_name(dp_train_input, "seed", int(f"{nnp}{random_0_1000}{padded_curr_iter}"))

        dp_train_input_file = (Path(f"{nnp}") / "training.json").resolve()

        write_json_file(dp_train_input, dp_train_input_file, enable_logging=False, read_only=True)

        job_file = replace_in_slurm_file_general(master_job_file, machine_spec, walltime_approx_s, machine_walltime_format, training_json["job_email"])

        # Replace the inputs/variables in the job file
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_VERSION_", f"{training_json['deepmd_model_version']}")
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_INPUT_FILE_", "training.json")
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_LOG_FILE_", "training.log")
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_OUTPUT_FILE_", "training.out")

        string_list_to_textfile(local_path / f"job_deepmd_train_{machine_spec['arch_type']}_{machine}.sh", job_file, read_only=True)
        del job_file, local_path, dp_train_input_file, random_0_1000

    del nnp, walltime_approx_s, dp_train_input, mean_s_per_step

    # Dump the JSON files (main, training and current input)
    arcann_logger.info(f"-" * 88)
    write_json_file(main_json, (control_path / "config.json"), read_only=True)
    write_json_file(training_json, (control_path / f"training_{padded_curr_iter}.json"), read_only=True)
    backup_and_overwrite_json_file(current_input_json, (current_path / "used_input.json"), read_only=True)

    # End
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")

    # Cleaning
    del current_path, control_path, training_path, data_path
    del default_input_json, default_input_json_present, user_input_json, user_input_json_present, user_input_json_filename
    del main_json, current_input_json, training_json, previous_training_json, labeling_json
    del user_machine_keyword
    del curr_iter, padded_curr_iter
    del machine, machine_spec, machine_walltime_format, machine_job_scheduler, machine_launch_command, machine_max_jobs, machine_max_array_size
    del master_job_file

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "prepare",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
