"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/29
"""

# Standard library modules
import copy
import logging
import subprocess
import sys
from pathlib import Path

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.json import backup_and_overwrite_json_file, load_default_json_file, load_json_file, write_json_file
from deepmd_iterative.common.machine import assert_same_machine, get_machine_keyword, get_machine_spec_for_step


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

    # Load the input JSON is present, load it
    current_input_json = load_json_file((current_path / "used_input.json"), abort_on_error=True)

    # Get control path, load the main JSON and the exploration JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    exploration_json = load_json_file((control_path / f"exploration_{padded_curr_iter}.json"))

    user_machine_keyword = current_input_json["user_machine_keyword_exp"]
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
        "exploration",
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

    if fake_machine is not None:
        logging.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        logging.info(f"Machine identified: '{machine}'.")
    del fake_machine

    # Check prep/launch
    assert_same_machine(user_machine_keyword, exploration_json, "exp")

    # Check if we can continue
    if exploration_json["is_launched"]:
        logging.critical(f"Already launched...")
        continuing = input(f"Do you want to continue?\n['Y' for yes, anything else to abort]\n")
        if continuing == "Y":
            del continuing
        else:
            logging.error(f"Aborting...")
            return 1
    if not exploration_json["is_locked"]:
        logging.error(f"Lock found. Execute first: exploration preparation.")
        logging.error(f"Aborting...")
        return 1

    # Launch the jobs
    exploration_types = []
    for system_auto in exploration_json["systems_auto"]:
        exploration_types.append(exploration_json["systems_auto"][system_auto]["exploration_type"])
    exploration_types = list(set(exploration_types))

    completed_count = 0
    for exploration_type in exploration_types:
        job_name = f"job-array_{exploration_type}-deepmd_explore_{machine_spec['arch_type']}_{machine}.sh"
        if (current_path / job_name).is_file():
            subprocess.run([machine_launch_command, f"./{job_name}"])
            logging.info(f"Exploration - Array LAMMPS launched.")
            completed_count += 1

    logging.info(f"-" * 88)

    if completed_count == len(exploration_types):
        exploration_json["is_launched"] = True

    # Dump the JSON files (exploration JSON and merged input JSON)
    write_json_file(exploration_json, (control_path / f"exploration_{padded_curr_iter}.json"))

    # End
    logging.info(f"-" * 88)

    if completed_count == len(exploration_types):
        logging.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")
    else:
        logging.critical(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is semi-success!")
        logging.critical(f"Some jobs did not launch correctly.")
        logging.critical(f"Please launch manually before continuing to the next step.")
        logging.critical(f"Replace the key 'is_launched' to 'True' in the 'exploration_{padded_curr_iter}.json'.")
    del completed_count

    # Cleaning
    del current_path, control_path, training_path
    del user_input_json_filename
    del main_json, current_input_json
    del curr_iter, padded_curr_iter
    del machine, machine_spec, machine_walltime_format, machine_launch_command, machine_job_scheduler, machine_max_jobs, machine_max_array_size

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "launch",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
