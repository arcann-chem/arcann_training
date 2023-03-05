from pathlib import Path
import logging
import sys
import copy
import subprocess

# deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    backup_and_overwrite_json_file,
    load_default_json_file,
    read_key_input_json,
)
from deepmd_iterative.common.machine import get_machine_spec_for_step
from deepmd_iterative.common.file import (
    check_file_existence,
    file_to_list_of_strings,
    change_directory,
    write_list_of_strings_to_file,
)
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.slurm import replace_in_slurm_file_general
from deepmd_iterative.common.list import replace_substring_in_list_of_strings


def main(
    step_name,
    phase_name,
    deepmd_iterative_path,
    fake_machine=None,
    input_fn="input.json",
):
    current_path = Path(".").resolve()
    training_path = current_path.parent

    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()}")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if correct folder
    validate_step_folder(step_name)

    # Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)

    # Get default inputs json
    default_present = False
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "data" / "inputs.json"
    )
    if bool(default_input_json):
        default_present = True

    # Get input json (user one)
    if (current_path / input_fn).is_file():
        input_json = load_json_file((current_path / input_fn))
    else:
        input_json = {}
    new_input_json = copy.deepcopy(input_json)

    # Get control path and config_json
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))
    training_json = load_json_file(
        (control_path / f"training_{current_iteration_zfill}.json")
    )
    jobs_path = deepmd_iterative_path / "data" / "jobs" / "training"

    # Get user machine keyword
    user_machine_keyword = read_key_input_json(
        input_json,
        new_input_json,
        "user_machine_keyword",
        default_input_json,
        step_name,
        default_present,
    )
    user_machine_keyword = (
        None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    )

    # Read machine spec
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
    if fake_machine is not None:
        logging.info(f"Pretending to be on: {fake_machine}")
    else:
        logging.info(f"We are on: {machine}")
    del fake_machine

    # Checks
    if not training_json["is_checked"]:
        logging.error(f"Lock found. Execute first: training check")
        logging.error(f"Aborting...")
        return 1

    check_file_existence(
        jobs_path / f"job_deepmd_freeze_{machine_spec['arch_type']}_{machine}.sh",
        error_msg=f"No SLURM file present for {step_name.capitalize()} / {phase_name.capitalize()} on this machine.",
    )
    slurm_file_master = file_to_list_of_strings(
        jobs_path / f"job_deepmd_freeze_{machine_spec['arch_type']}_{machine}.sh"
    )
    del jobs_path

    # Prep and launch DP Freeze
    completed_count = 0
    walltime_approx_s = 7200
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_path = Path(".").resolve() / str(it_nnp)

        check_file_existence(local_path / "model.ckpt.index")

        job_email = read_key_input_json(
            input_json,
            new_input_json,
            "job_email",
            default_input_json,
            step_name,
            default_present,
        )
        slurm_file = replace_in_slurm_file_general(
            slurm_file_master,
            machine_spec,
            walltime_approx_s,
            machine_walltime_format,
            job_email,
        )

        slurm_file = replace_substring_in_list_of_strings(
            slurm_file, "_R_DEEPMD_VERSION_", str(training_json["deepmd_model_version"])
        )
        slurm_file = replace_substring_in_list_of_strings(
            slurm_file,
            "_R_DEEPMD_MODEL_",
            "graph_" + str(it_nnp) + "_" + current_iteration_zfill,
        )

        write_list_of_strings_to_file(
            local_path / f"job_deepmd_freeze_{machine_spec['arch_type']}_{machine}.sh",
            slurm_file,
        )
        del slurm_file

        with (local_path / "checkpoint").open("w") as f:
            f.write('model_checkpoint_path: "model.ckpt"\n')
            f.write('all_model_checkpoint_paths: "model.ckpt"\n')
        del f
        if (
            local_path
            / ("job_deepmd_freeze_" + machine_spec["arch_type"] + "_" + machine + ".sh")
        ).is_file():
            change_directory(local_path)
            try:
                subprocess.run(
                    [
                        machine_launch_command,
                        f"./job_deepmd_freeze_{machine_spec['arch_type']}_{machine}.sh",
                    ]
                )
                logging.info(f"DP Freeze - {it_nnp} launched")
                completed_count += 1
            except FileNotFoundError:
                logging.critical(
                    f"DP Freeze - {it_nnp} NOT launched - {training_json['launch_command']} not found"
                )
            change_directory(local_path.parent)
        else:
            logging.critical(f"DP Freeze - {it_nnp} NOT launched - No job file")
        del local_path

    del it_nnp, slurm_file_master

    # Dump the dicts
    logging.info(f"-" * 88)
    write_json_file(config_json, (control_path / "config.json"))
    write_json_file(
        training_json, (control_path / f"training_{current_iteration_zfill}.json")
    )
    backup_and_overwrite_json_file(new_input_json, (current_path / input_fn))
    logging.info(f"-" * 88)
    if completed_count == config_json["nb_nnp"]:
        pass
    else:
        logging.critical(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is semi-succes !"
        )
        logging.critical(f"Some SLURM jobs did not launch correctly")
        logging.critical(f"Please launch manually before continuing to the next step")
    del completed_count

    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
    )

    # Cleaning
    del control_path
    del config_json
    del current_iteration, current_iteration_zfill
    del training_json
    del training_path, current_path

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "freeze",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
