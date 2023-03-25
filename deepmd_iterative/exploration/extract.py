from pathlib import Path
import logging
import copy
import sys

# Non-standard library imports
import numpy as np

# deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    load_default_json_file,
    backup_and_overwrite_json_file,
)
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.generate_config import set_subsys_params_deviation
from deepmd_iterative.common.exploration import get_last_frame_number


def main(
    step_name: str,
    phase_name: str,
    deepmd_iterative_path,
    fake_machine=None,
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
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)
    
    # Get the control path and load the config JSON file
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))

    # Load the exploration JSON file
    exploration_json = load_json_file(
        (control_path / f"exploration_{current_iteration_zfill}.json")
    )
    
    
    return 0
    
if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "extract",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
