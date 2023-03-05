from pathlib import Path
import logging
import sys
import copy
import subprocess

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    load_default_json_file,
    read_key_input_json,
)
from deepmd_iterative.common.file import (
    change_directory,
)

from deepmd_iterative.common.machine import (
    get_machine_spec_for_step,
    assert_same_machine,
)
from deepmd_iterative.common.check import validate_step_folder


def main(
    step_name: str,
    phase_name: str,
    deepmd_iterative_path,
    fake_machine=None,
    input_fn: str = "input.json",
):
    current_path = Path(".").resolve()
    training_path = current_path.parent

    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()}")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # ### Check if correct folder
    validate_step_folder(step_name)

    # ### Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)

    # ### Get default inputs json
    default_present = False
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "data" / "inputs.json"
    )
    if bool(default_input_json):
        default_present = True

    # ### Get input json (user one)
    if (current_path / input_fn).is_file():
        input_json = load_json_file((current_path / input_fn))
    else:
        input_json = {}
    new_input_json = copy.deepcopy(input_json)

    # ### Get control path and config_json
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))
    exploration_json = load_json_file(
        (control_path / f"exploration_{current_iteration_zfill}.json")
    )

    # ### Get machine info
    user_spec = read_key_input_json(
        input_json,
        new_input_json,
        "user_spec",
        default_input_json,
        step_name,
        default_present,
    )
    user_spec = None if isinstance(user_spec, bool) else user_spec

    # ### Read machine info
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
        user_spec,
        True,
    )
    if fake_machine is not None:
        logging.info(f"Pretending to be on {fake_machine}")
    else:
        logging.info(f"machine is {machine}")
    del fake_machine

    # ### Check prep/launch
    assert_same_machine(machine, exploration_json)

    # ### Checks
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
        logging.error(f"Lock found. Execute first: training preparation")
        logging.error(f"Aborting...")
        return 1

    # ### Launch the jobs
    completed_count = 0
    for it0_subsys_nr, it_subsys_nr in enumerate(config_json["subsys_nr"]):
        for it_nnp in range(1, config_json["nb_nnp"] + 1):
            for it_number in range(1, exploration_json["nb_traj"] + 1):

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
                            f"Exploration - {it_subsys_nr} {it_nnp} {it_number} launched"
                        )
                        completed_count += 1
                    except FileNotFoundError:
                        logging.critical(
                            f"Exploration - {it_subsys_nr} {it_nnp} {it_number} NOT launched - {exploration_json['launch_command']} not found"
                        )
                    change_directory(local_path.parent.parent.parent)
                else:
                    logging.critical(
                        f"Exploration - {it_subsys_nr} {it_nnp} {it_number} NOT launched - No job file"
                    )
                del local_path
            del it_number
        del it_nnp
    del it0_subsys_nr, it_subsys_nr

    if completed_count == (
        len(exploration_json["subsys_nr"])
        * exploration_json["nb_nnp"]
        * exploration_json["nb_traj"]
    ):
        exploration_json["is_launched"] = True

    write_json_file(
        exploration_json, (control_path / f"exploration_{current_iteration_zfill}.json")
    )

    logging.info(f"-" * 88)
    if completed_count == (
        len(exploration_json["subsys_nr"])
        * exploration_json["nb_nnp"]
        * exploration_json["nb_traj"]
    ):
        logging.info(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
        )
    else:
        logging.critical(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is semi-succes !"
        )
        logging.critical(f"Some SLURM jobs did not launch correctly")
        logging.critical(f"Please launch manually before continuing to the next step")
        logging.critical(
            f'Replace the key "is_launched" to True in the training_{current_iteration_zfill}.json.'
        )
    del completed_count

    # ### Cleaning
    del control_path
    del input_json, default_input_json, default_present, new_input_json
    del config_json
    del current_iteration, current_iteration_zfill
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
