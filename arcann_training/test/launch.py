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
import subprocess
import sys
from pathlib import Path

# Local imports
from arcann_training.common.check import validate_step_folder
from arcann_training.common.filesystem import change_directory
from arcann_training.common.json import load_json_file, write_json_file
from arcann_training.common.machine import (
    assert_same_machine,
    get_machine_spec_for_step,
)


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
    arcann_logger.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}."
    )
    arcann_logger.debug(f"Current path :{current_path}")
    arcann_logger.debug(f"Training path: {training_path}")
    arcann_logger.debug(f"Program path: {deepmd_iterative_path}")
    arcann_logger.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # If the used input JSON is present, load it
    current_input_json = load_json_file(
        (current_path / "used_input.json"), abort_on_error=True
    )
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    # Get control path and load the main JSON and the training JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    testing_json = load_json_file((control_path / f"testing_{padded_curr_iter}.json"))

    user_machine_keyword = current_input_json["user_machine_keyword_test"]
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
    arcann_logger.debug(f"machine: {machine}")
    arcann_logger.debug(f"machine_walltime_format: {machine_walltime_format}")
    arcann_logger.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    arcann_logger.debug(f"machine_launch_command: {machine_launch_command}")
    arcann_logger.debug(f"machine_max_jobs: {machine_max_jobs}")
    arcann_logger.debug(f"machine_max_array_size: {machine_max_array_size}")
    arcann_logger.debug(f"user_machine_keyword: {user_machine_keyword}")
    arcann_logger.debug(f"machine_spec: {machine_spec}")

    if fake_machine is not None:
        arcann_logger.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        arcann_logger.info(f"Machine identified: '{machine}'.")
    del fake_machine

    # Check prep/launch
    assert_same_machine(user_machine_keyword, testing_json, "test")

    # Check if we can continue
    if testing_json["is_launched"]:
        arcann_logger.critical(f"Already launched...")
        continuing = input(
            f"Do you want to continue?\n['Y' for yes, anything else to abort]\n"
        )
        if continuing == "Y":
            del continuing
        else:
            arcann_logger.error(f"Aborting...")
            return 0
    if not testing_json["is_prepared"]:
        arcann_logger.error(f"Lock found. Please execute 'test prepare' first.")
        arcann_logger.error(f"Aborting...")
        return 1

    # Launch the jobs
    completed_count = 0
    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"
        if (
            local_path / f"job_deepmd_test_{machine_spec['arch_type']}_{machine}.sh"
        ).is_file():
            change_directory(local_path)
            try:
                subprocess.run(
                    [
                        machine_launch_command,
                        f"./job_deepmd_test_{machine_spec['arch_type']}_{machine}.sh",
                    ]
                )
                arcann_logger.info(f"DP Test - '{nnp}' launched.")
                completed_count += 1
            except FileNotFoundError:
                arcann_logger.critical(
                    f"DP Test - '{nnp}' NOT launched - '{machine_launch_command}' not found."
                )
            change_directory(local_path.parent)
        else:
            arcann_logger.critical(f"DP Test - '{nnp}' NOT launched - No job file.")
        del local_path
    del nnp

    arcann_logger.info(f"-" * 88)
    # Update the flag in the training JSON
    if completed_count == main_json["nnp_count"]:
        testing_json["is_launched"] = True

    # Dump the training JSON
    write_json_file(
        testing_json,
        (control_path / f"testing_{padded_curr_iter}.json"),
        read_only=True,
    )

    # End
    arcann_logger.info(f"-" * 88)
    if completed_count == main_json["nnp_count"]:
        arcann_logger.info(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
        )
    else:
        arcann_logger.critical(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is semi-success!"
        )
        arcann_logger.critical(f"Some jobs did not launch correctly.")
        arcann_logger.critical(
            f"Please launch manually before continuing to the next step."
        )
        arcann_logger.critical(
            f"Replace the key 'is_launched' to 'True' in the 'testing_{padded_curr_iter}.json'."
        )
    del completed_count

    # Cleaning
    del current_path, control_path, training_path
    del user_input_json_filename
    del user_machine_keyword
    del main_json, current_input_json, testing_json
    del curr_iter, padded_curr_iter
    del (
        machine,
        machine_spec,
        machine_walltime_format,
        machine_launch_command,
        machine_job_scheduler,
        machine_max_jobs,
        machine_max_array_size,
    )

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "test",
            "launch",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
