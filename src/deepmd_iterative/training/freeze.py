from pathlib import Path
import logging
import sys
import copy
import subprocess

# deepmd_iterative imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.file import (
    change_directory,
    check_file_existence,
    file_to_list_of_strings,
    write_list_of_strings_to_file,
)
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    load_default_json_file,
    backup_and_overwrite_json_file,
)
from deepmd_iterative.common.json_parameters import (
    get_key_in_dict,
    get_machine_keyword,
)
from deepmd_iterative.common.list import replace_substring_in_list_of_strings
from deepmd_iterative.common.machine import get_machine_spec_for_step
from deepmd_iterative.common.slurm import replace_in_slurm_file_general


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

    # Get control path and load the main config (JSON) and the training config (JSON)
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))
    training_config = load_json_file(
        (control_path / f"training_{padded_curr_iter}.json")
    )

    # Checks
    if not training_config["is_checked"]:
        logging.error(f"Lock found. Execute first: training check.")
        logging.error(f"Aborting...")
        return 1

    # Get extra needed paths
    jobs_path = deepmd_iterative_path / "data" / "jobs" / "training"

    # Get the machine keyword (input override previous training override default_config)
    # And update the new input
    user_machine_keyword = get_machine_keyword(user_config, training_config, default_config)
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

    check_file_existence(
        jobs_path / f"job_deepmd_freeze_{machine_spec['arch_type']}_{machine}.sh",
        error_msg=f"No SLURM file present for {current_step.capitalize()} / {current_phase.capitalize()} on this machine.",
    )
    master_job_file = file_to_list_of_strings(
        jobs_path / f"job_deepmd_freeze_{machine_spec['arch_type']}_{machine}.sh"
    )
    current_config["job_email"] = get_key_in_dict("job_email", user_config, training_config, default_config)
    del jobs_path

    # Prep and launch DP Freeze
    completed_count = 0
    walltime_approx_s = 7200
    for nnp in range(1, main_config["nnp_count"] + 1):
        local_path = Path(".").resolve() / f"{nnp}"

        check_file_existence(local_path / "model.ckpt.index")

        job_file = replace_in_slurm_file_general(
            master_job_file,
            machine_spec,
            walltime_approx_s,
            machine_walltime_format,
            current_config["job_email"],
        )

        job_file = replace_substring_in_list_of_strings(job_file, "_R_DEEPMD_VERSION_", f"{training_config['deepmd_model_version']}")
        job_file = replace_substring_in_list_of_strings(job_file, "_R_DEEPMD_MODEL_", f"graph_{nnp}_{padded_curr_iter}",)

        write_list_of_strings_to_file(
            local_path / f"job_deepmd_freeze_{machine_spec['arch_type']}_{machine}.sh",
            job_file,
        )
        del job_file

        with (local_path / "checkpoint").open("w") as f:
            f.write('model_checkpoint_path: "model.ckpt"\n')
            f.write('all_model_checkpoint_paths: "model.ckpt"\n')
        del f
        if (local_path / f"job_deepmd_freeze_{machine_spec['arch_type']}_{machine}.sh").is_file():
            change_directory(local_path)
            try:
                subprocess.run(
                    [
                        machine_launch_command,
                        f"./job_deepmd_freeze_{machine_spec['arch_type']}_{machine}.sh",
                    ]
                )
                logging.info(f"DP Freeze - {nnp} launched.")
                completed_count += 1
            except FileNotFoundError:
                logging.critical(
                    f"DP Freeze - {nnp} NOT launched - {training_config['launch_command']} not found."
                )
            change_directory(local_path.parent)
        else:
            logging.critical(f"DP Freeze - {nnp} NOT launched - No job file.")
        del local_path

    del nnp, master_job_file

    # Dump the dicts
    logging.info(f"-" * 88)
    write_json_file(main_config, (control_path / "config.json"))
    write_json_file(
        training_config, (control_path / f"training_{padded_curr_iter}.json")
    )
    backup_and_overwrite_json_file(current_config, (current_path / user_config_filename))
    logging.info(f"-" * 88)
    if completed_count == main_config["nnp_count"]:
        pass
    else:
        logging.critical(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is semi-success!"
        )
        logging.critical(f"Some SLURM jobs did not launch correctly.")
        logging.critical(f"Please launch manually before continuing to the next step.")
    del completed_count

    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del default_config, default_config_present, user_config, user_config_present, user_config_filename
    del main_config, current_config, training_config
    del curr_iter, padded_curr_iter
    del machine, machine_spec, machine_walltime_format, machine_launch_command

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "freeze",
            Path(sys.argv[1]),
            fake_machine = sys.argv[2],
            user_config_filename = sys.argv[3],
        )
    else:
        pass
