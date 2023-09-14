"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/09/15
"""
# Standard library modules
import copy
import logging
import subprocess
import sys
from pathlib import Path

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.filesystem import (
    change_directory,
    check_file_existence,
)
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    load_default_json_file,
    backup_and_overwrite_json_file,
)
from deepmd_iterative.common.list import (
    replace_substring_in_string_list,
    string_list_to_textfile,
    textfile_to_string_list,
)
from deepmd_iterative.common.machine import (
    get_machine_keyword,
    get_machine_spec_for_step,
)
from deepmd_iterative.common.slurm import replace_in_slurm_file_general


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
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}."
    )
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Load the default input JSON
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "assets" / "default_config.json"
    )[current_step]
    default_input_json_present = bool(default_input_json)
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

    # Make a deepcopy of it to create the current input JSON
    merged_input_json = copy.deepcopy(user_input_json)

    # Get control path, load the main JSON and the training JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    training_json = load_json_file((control_path / f"training_{padded_curr_iter}.json"))

    # Check if we can continue
    if not training_json["is_frozen"]:
        logging.error(f"Lock found. Execute first: training check_freeze.")
        logging.error(f"Aborting...")
        return 1

    # Get the machine keyword (Priority: user > previous > default)
    # And update the current input JSON
    user_machine_keyword = get_machine_keyword(
        user_input_json, training_json, default_input_json, "compress"
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    # Set it to None if bool, because: get_machine_spec_for_step needs None
    user_machine_keyword = (
        None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")

    # From the keyword (or default), get the machine spec (or for the fake one)
    (
        machine,
        machine_walltime_format,
        machine_job_scheduler,
        machine_launch_command,
        user_machine_keyword,
        machine_spec,
    ) = get_machine_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "compressing",
        fake_machine,
        user_machine_keyword,
    )
    logging.debug(f"machine: {machine}")
    logging.debug(f"machine_walltime_format: {machine_walltime_format}")
    logging.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    logging.debug(f"machine_launch_command: {machine_launch_command}")
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    logging.debug(f"machine_spec: {machine_spec}")

    merged_input_json["user_machine_keyword_compress"] = user_machine_keyword
    logging.debug(f"merged_input_json: {merged_input_json}")

    if fake_machine is not None:
        logging.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        logging.info(f"Machine identified: '{machine}'.")
    del fake_machine

    training_json["user_machine_keyword_compress"] = user_machine_keyword

    # Check if the job file exists
    job_file_name = f"job_deepmd_compress_{machine_spec['arch_type']}_{machine}.sh"
    if (current_path.parent / "user_files" / job_file_name).is_file():
        master_job_file = textfile_to_string_list(
            current_path.parent / "user_files" / job_file_name
        )
    else:
        logging.error(
            f"No JOB file provided for '{current_step.capitalize()} / {current_phase.capitalize()}' for this machine."
        )
        logging.error(f"Aborting...")
        return 1

    logging.debug(f"master_job_file: {master_job_file[0:5]}, {master_job_file[-5:-1]}")
    del job_file_name

    # Prep and launch DP Compress
    completed_count = 0
    walltime_approx_s = 7200
    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"

        check_file_existence(local_path / "model.ckpt.index")

        job_file = replace_in_slurm_file_general(
            master_job_file,
            machine_spec,
            walltime_approx_s,
            machine_walltime_format,
            merged_input_json["job_email"],
        )

        job_file = replace_substring_in_string_list(
            job_file, "_R_DEEPMD_VERSION_", f"{training_json['deepmd_model_version']}"
        )
        job_file = replace_substring_in_string_list(
            job_file,
            "_R_DEEPMD_MODEL_",
            f"graph_{nnp}_{padded_curr_iter}",
        )

        string_list_to_textfile(
            local_path
            / f"job_deepmd_compress_{machine_spec['arch_type']}_{machine}.sh",
            job_file,
        )
        del job_file

        with (local_path / "checkpoint").open("w") as f:
            f.write('model_checkpoint_path: "model.ckpt"\n')
            f.write('all_model_checkpoint_paths: "model.ckpt"\n')
        del f
        if (
            local_path / f"job_deepmd_compress_{machine_spec['arch_type']}_{machine}.sh"
        ).is_file():
            change_directory(local_path)
            try:
                # subprocess.run(
                #     [
                #         machine_launch_command,
                #         f"./job_deepmd_compress_{machine_spec['arch_type']}_{machine}.sh",
                #     ]
                # )
                logging.info(f"DP Compress - '{nnp}' launched.")
                completed_count += 1
            except FileNotFoundError:
                logging.critical(
                    f"DP Compress - '{nnp}' NOT launched - '{training_json['launch_command']}' not found."
                )
            change_directory(local_path.parent)
        else:
            logging.critical(f"DP Compress - '{nnp}' NOT launched - No job file.")
        del local_path

    del nnp, master_job_file

    # Dump the JSON files (main, training and merged input)
    logging.info(f"-" * 88)
    write_json_file(main_json, (control_path / "config.json"))
    write_json_file(training_json, (control_path / f"training_{padded_curr_iter}.json"))
    backup_and_overwrite_json_file(
        merged_input_json, (current_path / user_input_json_filename)
    )

    # End
    logging.info(f"-" * 88)
    if completed_count == main_json["nnp_count"]:
        logging.info(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
        )
    else:
        logging.critical(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is semi-success!"
        )
        logging.critical(f"Some jobs did not launch correctly.")
        logging.critical(f"Please launch manually before continuing to the next step.")
    del completed_count

    # Cleaning
    del current_path, control_path, training_path
    del (
        default_input_json,
        default_input_json_present,
        user_input_json,
        user_input_json_present,
        user_input_json_filename,
    )
    del walltime_approx_s, user_machine_keyword
    del main_json, merged_input_json, training_json
    del curr_iter, padded_curr_iter
    del machine, machine_spec, machine_walltime_format, machine_launch_command, machine_job_scheduler

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
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
