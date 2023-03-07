from pathlib import Path
import logging
import sys

# deepmd_iterative imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.file import remove_files_matching_glob
from deepmd_iterative.common.json import load_json_file


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

    # Check if correct folder
    validate_step_folder(step_name)

    # Get iteration
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Get control path, config JSON and training JSON
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))
    training_json = load_json_file(
        (control_path / f"training_{padded_curr_iter}.json")
    )

    # Checks
    if not training_json["is_frozen"]:
        logging.error(f"Lock found. Execute first: training check_freeze")
        logging.error(f"Aborting...")
        return 1
    logging.critical(
        f"This is cleaning step. It should be run after training update_iter"
    )
    continuing = input(f"Are you sur? (Y for Yes, anything else to abort)")
    if continuing == "Y":
        del continuing
    else:
        logging.error(f"Aborting...")
        return 1

    # Delete
    logging.info(f"Deleting DP-Freeze related error files...")
    remove_files_matching_glob(current_path, "**/graph*freeze.out")
    logging.info(f"Deleting DP-Compress related error files...")
    remove_files_matching_glob(current_path, "**/graph*compress.out")
    logging.info(f"Deleting DP-Train related error files...")
    remove_files_matching_glob(current_path, "**/training.out")
    logging.info(f"Deleting SLURM launch files...")
    remove_files_matching_glob(current_path, "**/*.sh")

    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a success !"
    )

    # Cleaning
    del control_path
    del padded_curr_iter, curr_iter
    del training_json, config_json
    del training_path, current_path

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "clean",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
