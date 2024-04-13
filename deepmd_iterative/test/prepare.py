"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/04/13
"""

# Standard library modules
import logging
import sys
import subprocess
from pathlib import Path

# Non-standard library imports
import numpy as np

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.json import backup_and_overwrite_json_file, get_key_in_dict, load_default_json_file, load_json_file, write_json_file
from deepmd_iterative.common.list import replace_substring_in_string_list, string_list_to_textfile, textfile_to_string_list
from deepmd_iterative.common.machine import get_machine_keyword, get_machine_spec_for_step
from deepmd_iterative.common.slurm import replace_in_slurm_file_general
from deepmd_iterative.common.filesystem import check_directory


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_input_json_filename: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}.")
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

    # Load the default input JSON
    default_input_json = load_default_json_file(deepmd_iterative_path / "assets" / "default_config.json")[current_step]
    default_input_json_present = bool(default_input_json)
    if default_input_json_present and not (current_path / "default_input.json").is_file():
        write_json_file(default_input_json, (current_path / "default_input.json"))
    logging.debug(f"default_input_json: {default_input_json}")
    logging.debug(f"default_input_json_present: {default_input_json_present}")

    # Load the user input JSON
    if (current_path / user_input_json_filename).is_file():
        user_input_json = load_json_file((current_path / user_input_json_filename))
    else:
        user_input_json = {}
    user_input_json_present = bool(user_input_json)
    logging.debug(f"user_input_json: {user_input_json}")
    logging.debug(f"user_input_json_present: {user_input_json_present}")

    # Create a empty (None/Null) current input JSON
    current_input_json = {}
    for key in default_input_json:
        current_input_json[key] = None
    logging.debug(f"current_input_json: {current_input_json}")

    # Create a empty (None/Null) testing JSON
    testing_json = {}
    for key in default_input_json:
        testing_json[key] = None
    logging.debug(f"testing_json: {testing_json}")

    # Get control path, load the main JSON and the training JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    training_json = load_json_file((control_path / f"training_{padded_curr_iter}.json"))

    # Load the previous testing JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_testing_json = load_json_file((control_path / f"testing_{padded_prev_iter}.json"))
    else:
        previous_testing_json = {}

    # If the user input JSON is present, update the current input JSON with it (Priority: user > default)
    for key in ["user_machine_keyword_test", "job_email", "job_walltime_h", "deepmd_model_version", "is_compressed"]:
        if user_input_json_present and key in user_input_json:
            current_input_json[key] = user_input_json[key]
        else:
            current_input_json[key] = default_input_json[key]

    # If the user input JSON is not present, update the current input JSON with the training JSON (Priority: training > default)
    if not user_input_json_present and "deepmd_model_version" not in user_input_json:
        current_input_json["deepmd_model_version"] = training_json["deepmd_model_version"]
    if not user_input_json_present and "is_compressed" not in user_input_json:
        current_input_json["is_compressed"] = training_json["is_compressed"]

    # Get the machine keyword (Priority: user > previous > default)
    # And update the current input JSON
    user_machine_keyword = get_machine_keyword(current_input_json, previous_testing_json, default_input_json, "test")
    # Set it to None if bool, because: get_machine_spec_for_step needs None
    user_machine_keyword = None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")

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
        "test",
        fake_machine,
        user_machine_keyword,
    )
    logging.debug(f"machine: {machine}")
    logging.debug(f"machine_walltime_format: {machine_walltime_format}")
    logging.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    logging.debug(f"machine_launch_command: {machine_launch_command}")
    logging.debug(f"machine_max_jobs: {machine_max_jobs}")
    logging.debug(f"machine_max_array_size: {machine_max_array_size}")
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    logging.debug(f"machine_spec: {machine_spec}")

    # Update the current input JSON
    current_input_json["user_machine_keyword_test"] = user_machine_keyword
    logging.debug(f"current_input_json: {current_input_json}")

    # Log the machine
    if fake_machine is not None:
        logging.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        logging.info(f"Machine identified: '{machine}'.")
    del fake_machine

    # Check if we can continue
    if not training_json["is_incremented"]:
        logging.error(f"Lock found. Please execute 'training increment' first.")
        logging.error(f"Aborting...")
        return 1

    # Check if the job file is present
    job_file_name = f"job_deepmd_test_{machine_spec['arch_type']}_{machine}.sh"
    if (current_path.parent / "user_files" / job_file_name).is_file():
        master_job_file = textfile_to_string_list(current_path.parent / "user_files" / job_file_name)
        logging.debug(f"master_job_file: {master_job_file[0:5]}, {master_job_file[-5:-1]}")
    else:
        logging.error(f"No JOB file provided for '{current_step.capitalize()} / {current_phase.capitalize()}' for this machine.")
        logging.error(f"Aborting...")
        return 1
    del job_file_name

    # Get the email (Priority: user > previous > default)
    current_input_json["job_email"] = get_key_in_dict("job_email", user_input_json, previous_testing_json, default_input_json)

    # Calculate the walltime (Priority: user > previous > default)
    if "job_walltime_h" in user_input_json and user_input_json["job_walltime_h"] > 0:
        walltime_approx_s = int(user_input_json["job_walltime_h"] * 3600)
        logging.debug(f"job_walltime_h: {user_input_json['job_walltime_h']}")
    else:
        if curr_iter == 0:
            walltime_approx_s = int(default_input_json["job_walltime_h"] * 3600)
        else:
            walltime_approx_s = int(previous_testing_json["job_walltime_h"] * 3600)
    logging.debug(f"walltime_approx_s: {walltime_approx_s}")

    # "deepmd_model_version" and "is_compressed" should not be carried over from the previous testing JSON but from the training JSON
    # because the user might have changed the model version or the compression status in the training JSON

    # Update the current input JSON
    current_input_json["job_walltime_h"] = walltime_approx_s / 3600

    # Update the testing JSON with the current input JSON
    for key in current_input_json:
        testing_json[key] = current_input_json[key]

    # Get the list of the NNP files
    if current_input_json["is_compressed"]:
        nnp_list = [f"graph_{nnp}_{padded_curr_iter}_compressed.pb" for nnp in range(1, main_json["nnp_count"] + 1)]
    else:
        nnp_list = [f"graph_{nnp}_{padded_curr_iter}.pb" for nnp in range(1, main_json["nnp_count"] + 1)]
    logging.debug(f"nnp_list: {nnp_list}")

    # Check if the NNP files are present
    if not all([(training_path / "NNP" / nnp).is_file() for nnp in nnp_list]):
        logging.error(f"NNP file(s) not found.")
        logging.error(f"Aborting...")
        return 1

    # Check if the data folder is present
    if not (training_path / "data").is_dir():
        logging.error(f"Data folder not found.")
        logging.error(f"Aborting...")
        return 1

    # Prepare the testing, create the folders and the job files, and update the testing JSON
    for idx_nnp, nnp in enumerate(nnp_list):
        idx_nnp = idx_nnp + 1
        local_path = current_path / f"{idx_nnp}"
        local_path.mkdir(exist_ok=True)
        check_directory(local_path)

        # Prepare the job file and save it
        job_file = replace_in_slurm_file_general(master_job_file, machine_spec, walltime_approx_s, machine_walltime_format, current_input_json["job_email"])

        # Replace the inputs/variables in the job file
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_VERSION_", f"{training_json['deepmd_model_version']}")
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_MODEL_FILE_", f"{nnp}")

        string_list_to_textfile(local_path / f"job_deepmd_test_{machine_spec['arch_type']}_{machine}.sh", job_file, read_only=True)

        # Create the symbolic links for the NNP files and the data folder
        subprocess.call(["ln", "-nsf", str((training_path / "NNP" / nnp)), str(local_path)])
        subprocess.call(["ln", "-nsf", str((current_path / "data")), str(local_path)])

        # Update the testing JSON
        testing_json[f"{nnp.replace('.pb', '')}"] = {}

    # Set the flags in the testing JSON
    testing_json = {
        **testing_json,
        "is_prepared": True,
        "is_launched": False,
        "is_checked": False,
    }

    logging.info(f"-" * 88)
    # Dump the testing JSON and the current input JSON
    write_json_file(testing_json, (control_path / f"testing_{padded_curr_iter}.json"), read_only=True)
    backup_and_overwrite_json_file(current_input_json, (current_path / "used_input.json"), read_only=True)

    # End
    logging.info(f"-" * 88)
    logging.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")

    # Cleaning
    del current_path, control_path, training_path
    del default_input_json, default_input_json_present, user_input_json, user_input_json_present, user_input_json_filename
    del main_json, current_input_json
    del curr_iter, padded_curr_iter
    del machine, machine_walltime_format, machine_job_scheduler, machine_launch_command, user_machine_keyword, machine_spec
    del master_job_file

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "test",
            "prepare",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
