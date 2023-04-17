"""
Created: 2023/01/01
Last modified: 2023/04/17
"""
from pathlib import Path
import logging
import sys
import copy
import subprocess

# deepmd_iterative imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.filesystem import (
    change_directory,
)
from deepmd_iterative.common.json import (
    backup_and_overwrite_json_file,
    load_default_json_file,
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.json_parameters import get_machine_keyword

from deepmd_iterative.common.machine import (
    assert_same_machine,
    get_machine_spec_for_step,
)


def main(
    step_name: str,
    phase_name: str,
    deepmd_iterative_path: Path,
    fake_machine = None,
    input_fn: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()}")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(step_name)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Load the master input JSON file for the program
    default_present = False
    default_json = load_default_json_file(deepmd_iterative_path / "assets" / "default_config.json")[step_name]
    if bool(default_json):
        default_present = True
    logging.debug(f"default_json: {default_json}")
    logging.debug(f"default_present: {default_present}")

    # Load the user input JSON file
    if (current_path / input_fn).is_file():
        input_json = load_json_file((current_path / input_fn))
        input_present = True
    else:
        input_json = {}
        input_present = False
    logging.debug(f"input_json: {input_json}")
    logging.debug(f"input_present: {input_present}")

    # Make a deepcopy
    new_input_json = copy.deepcopy(input_json)

    # Get control path, config JSON and exploration JSON
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))
    exploration_json = load_json_file(
        (control_path / f"exploration_{padded_curr_iter}.json")
    )

    # Get the machine keyword (input override training override default_json)
    # And update the new input
    user_machine_keyword = get_machine_keyword(input_json, exploration_json, default_json)
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    new_input_json["user_machine_keyword"] = user_machine_keyword
    logging.debug(f"new_input_json: {new_input_json}")
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

    # Check prep/launch
    assert_same_machine(machine, exploration_json)

    # Checks
    if exploration_json["is_launched"]:
        logging.critical(f"Already launched.")
        continuing = input(
            f"Should it be run again? (Y for Yes, anything else to abort)"
        )
        if continuing == "Y":
            del continuing
        else:
            logging.error(f"Aborting...")
            return 1
    if not exploration_json["is_locked"]:
        logging.error(f"Lock found. Execute first: training preparation.")
        logging.error(f"Aborting...")
        return 1

    # Launch the jobs
    completed_count = 0
    for it0_subsys_nr, it_subsys_nr in enumerate(config_json["subsys_nr"]):
        for it_nnp in range(1, config_json["nnp_count"] + 1):
            for it_number in range(1, exploration_json["traj_count"] + 1):

                local_path = (
                    Path(".").resolve()
                    / str(it_subsys_nr)
                    / str(it_nnp)
                    / (str(it_number).zfill(5))
                )

                if (
                    local_path
                    / f"job_deepmd_{exploration_json['exploration_type']}_{exploration_json['arch_type']}_{machine}.sh"
                ).is_file():
                    change_directory(local_path)
                    try:
                        subprocess.run(
                            [
                                exploration_json["launch_command"],
                                f"./job_deepmd_{exploration_json['exploration_type']}_{exploration_json['arch_type']}_{machine}.sh",
                            ]
                        )
                        logging.info(
                            f"Exploration - {it_subsys_nr} {it_nnp} {it_number} launched."
                        )
                        completed_count += 1
                    except FileNotFoundError:
                        logging.critical(
                            f"Exploration - {it_subsys_nr} {it_nnp} {it_number} NOT launched - {exploration_json['launch_command']} not found."
                        )
                    change_directory(local_path.parent.parent.parent)
                else:
                    logging.critical(
                        f"Exploration - {it_subsys_nr} {it_nnp} {it_number} NOT launched - No job file."
                    )
                del local_path
            del it_number
        del it_nnp
    del it0_subsys_nr, it_subsys_nr

    if completed_count == (
        len(exploration_json["subsys_nr"])
        * exploration_json["nnp_count"]
        * exploration_json["traj_count"]
    ):
        exploration_json["is_launched"] = True

    write_json_file(
        exploration_json, (control_path / f"exploration_{padded_curr_iter}.json")
    )
    backup_and_overwrite_json_file(new_input_json, (current_path / input_fn))

    logging.info(f"-" * 88)
    if completed_count == (
        len(exploration_json["subsys_nr"])
        * exploration_json["nnp_count"]
        * exploration_json["traj_count"]
    ):
        logging.info(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a success!"
        )
    else:
        logging.critical(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is semi-success!"
        )
        logging.critical(f"Some SLURM jobs did not launch correctly.")
        logging.critical(f"Please launch manually before continuing to the next step.")
        logging.critical(
            f'Replace the key "is_launched" to True in the training_{padded_curr_iter}.json.'
        )
    del completed_count

    # Cleaning
    del control_path
    del input_json, default_json, default_present, new_input_json
    del config_json
    del curr_iter, padded_curr_iter
    del exploration_json
    del machine
    del training_path, current_path

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "launch",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
