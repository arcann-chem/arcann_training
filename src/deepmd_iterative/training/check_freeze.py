from pathlib import Path
import logging
import sys

# deepmd_iterative imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
)


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

    completed_count = 0
    for it_nnp in range(1, main_config["nb_nnp"] + 1):
        local_path = Path(".").resolve() / str(it_nnp)
        if (
            local_path
            / ("graph_" + str(it_nnp) + "_" + padded_curr_iter + ".pb")
        ).is_file():
            completed_count += 1
        else:
            logging.critical("DP Freeze - " + str(it_nnp) + " not finished/failed.")
        del local_path
    del it_nnp

    if completed_count == main_config["nb_nnp"]:
        training_config["is_frozen"] = True
    else:
        logging.error(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a failure!"
        )
        logging.error(f"Some DP Freeze did not finished correctly.")
        logging.error(f"Please check manually before relaunching this step.")
        logging.error(f"Aborting...")
        return 1
    del completed_count

    write_json_file(
        training_config, (control_path / f"training_{padded_curr_iter}.json")
    )

    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del user_config_filename
    del main_config, training_config
    del curr_iter, padded_curr_iter

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "check_freeze",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_config_filename=sys.argv[3],
        )
    else:
        pass
