"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/14
"""

# Standard library modules
import logging
import sys
from pathlib import Path
import subprocess

# Local imports
from arcann_training.common.check import validate_step_folder
from arcann_training.common.filesystem import change_directory, check_file_existence
from arcann_training.common.json import load_json_file, write_json_file, load_default_json_file, backup_and_overwrite_json_file
from arcann_training.common.list import replace_substring_in_string_list, string_list_to_textfile, textfile_to_string_list
from arcann_training.common.machine import get_machine_keyword, get_machine_spec_for_step
from arcann_training.common.slurm import replace_in_slurm_file_general


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

    # Load the default input JSON
    default_input_json = load_default_json_file(deepmd_iterative_path / "assets" / "default_config.json")[current_step]
    default_input_json_present = bool(default_input_json)
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

    # If the used input JSON is present, load it
    if (current_path / "used_input.json").is_file():
        current_input_json = load_json_file((current_path / "used_input.json"))
    else:
        arcann_logger.warning(f"No used_input.json found. Starting with empty one.")
        arcann_logger.warning(f"You should avoid this by not deleting the used_input.json file.")
        current_input_json = {}
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    # Get control path, load the main JSON and the training JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    training_json = load_json_file((control_path / f"training_{padded_curr_iter}.json"))

    # Load the previous training JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_training_json = load_json_file((control_path / f"training_{padded_prev_iter}.json"))
    else:
        previous_training_json = {}

    # If the user input JSON is present, update the user_machine_keyword_freeze and job_email if present
    for key in ["user_machine_keyword_compress", "job_email"]:
        if user_input_json_present and key in user_input_json:
            current_input_json[key] = user_input_json[key]
        elif key in previous_training_json:
            current_input_json[key] = previous_training_json[key]
        else:
            current_input_json[key] = default_input_json[key]

    # Check if we can continue
    if training_json["is_compress_launched"]:
        arcann_logger.critical(f"Already launched...")
        continuing = input(f"Do you want to continue?\n['Y' for yes, anything else to abort]\n")
        if continuing == "Y":
            del continuing
        else:
            arcann_logger.error(f"Aborting...")
            return 0
    if not training_json["is_frozen"]:
        arcann_logger.error(f"Lock found. Please execute 'training check_freeze' first.")
        arcann_logger.error(f"Aborting...")
        return 1

    # Get the machine keyword (Priority: user > previous > default)
    # And update the current input JSON
    user_machine_keyword = get_machine_keyword(current_input_json, training_json, default_input_json, "compress")
    arcann_logger.debug(f"user_machine_keyword: {user_machine_keyword}")
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
        "compressing",
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

    current_input_json["user_machine_keyword_compress"] = user_machine_keyword
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    if fake_machine is not None:
        arcann_logger.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        arcann_logger.info(f"Machine identified: '{machine}'.")
    del fake_machine

    training_json["user_machine_keyword_compress"] = user_machine_keyword

    # Check if the job file exists
    job_file_name = f"job_deepmd_compress_{machine_spec['arch_type']}_{machine}.sh"
    if (current_path.parent / "user_files" / job_file_name).is_file():
        master_job_file = textfile_to_string_list(current_path.parent / "user_files" / job_file_name)
    else:
        arcann_logger.error(f"No JOB file provided for '{current_step.capitalize()} / {current_phase.capitalize()}' for this machine.")
        arcann_logger.error(f"Aborting...")
        return 1

    arcann_logger.debug(f"master_job_file: {master_job_file[0:5]}, {master_job_file[-5:-1]}")
    del job_file_name

    # Prep and launch DP Compress
    completed_count = 0
    walltime_approx_s = 3900
    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"

        check_file_existence(local_path / "model.ckpt.index")

        job_file = replace_in_slurm_file_general(master_job_file, machine_spec, walltime_approx_s, machine_walltime_format, current_input_json["job_email"])
        # Replace the inputs/variables in the job file
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_VERSION_", f"{training_json['deepmd_model_version']}")
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_MODEL_FILE_", f"graph_{nnp}_{padded_curr_iter}.pb")
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_COMPRESSED_MODEL_FILE_", f"graph_{nnp}_{padded_curr_iter}_compressed.pb")
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_LOG_FILE_", f"graph_{nnp}_{padded_curr_iter}_compress.log")
        job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_OUTPUT_FILE_", f"graph_{nnp}_{padded_curr_iter}_compress.out")

        string_list_to_textfile(local_path / f"job_deepmd_compress_{machine_spec['arch_type']}_{machine}.sh", job_file, read_only=True)
        del job_file

        with (local_path / "checkpoint").open("w") as f:
            f.write('model_checkpoint_path: "model.ckpt"\n')
            f.write('all_model_checkpoint_paths: "model.ckpt"\n')
        del f
        if (local_path / f"job_deepmd_compress_{machine_spec['arch_type']}_{machine}.sh").is_file():
            change_directory(local_path)
            try:
                subprocess.run([machine_launch_command, f"./job_deepmd_compress_{machine_spec['arch_type']}_{machine}.sh"])
                arcann_logger.info(f"DP Compress - '{nnp}' launched.")
                completed_count += 1
            except FileNotFoundError:
                arcann_logger.critical(f"DP Compress - '{nnp}' NOT launched - '{machine_launch_command}' not found.")
            change_directory(local_path.parent)
        else:
            arcann_logger.critical(f"DP Compress - '{nnp}' NOT launched - No job file.")
        del local_path

    del nnp, master_job_file

    arcann_logger.info(f"-" * 88)
    # Update the boolean in the training JSON
    if completed_count == main_json["nnp_count"]:
        training_json["is_compress_launched"] = True

    # Dump the JSON files (main, training and current input)
    write_json_file(main_json, (control_path / "config.json"), read_only=True)
    write_json_file(training_json, (control_path / f"training_{padded_curr_iter}.json"), read_only=True)
    backup_and_overwrite_json_file(current_input_json, (current_path / "used_input.json"), read_only=True)

    # End
    arcann_logger.info(f"-" * 88)
    if completed_count == main_json["nnp_count"]:
        arcann_logger.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")
    else:
        arcann_logger.critical(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is semi-success!")
        arcann_logger.critical(f"Some jobs did not launch correctly.")
        arcann_logger.critical(f"Please launch manually before continuing to the next step.")
    del completed_count

    # Cleaning
    del current_path, control_path, training_path
    del default_input_json, default_input_json_present, user_input_json, user_input_json_present, user_input_json_filename
    del user_machine_keyword, walltime_approx_s
    del main_json, current_input_json, training_json
    del curr_iter, padded_curr_iter
    del machine, machine_spec, machine_walltime_format, machine_launch_command, machine_job_scheduler

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "compress",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
