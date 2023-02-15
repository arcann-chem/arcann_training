from pathlib import Path
import logging
import sys

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    json_read
)
from deepmd_iterative.common.files import (
    remove_file_glob
)
from deepmd_iterative.common.checks import validate_step_folder


def main(
    step_name,
    phase_name,
    deepmd_iterative_apath,
    fake_cluster=None,
    input_fn="input.json",
):
    current_apath = Path(".").resolve()
    training_iterative_apath = current_apath.parent
    
    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()}")
    logging.debug(f"Current path :{current_apath}")
    logging.debug(f"Training path: {training_iterative_apath}")
    logging.debug(f"Program path: {deepmd_iterative_apath}")
    logging.info(f"-" * 88)
    
    # ### Check if correct folder
    validate_step_folder()

    # ### Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]

    # ### Get control path and config_json
    control_apath = training_iterative_apath / "control"
    training_json = json_read(
        (control_apath / f"training_{current_iteration_zfill}.json"), True, True
    )

    # ### Checks
    if not training_json["is_frozen"]:
        logging.error(f"Lock found. Execute first: training check_freeze")
        logging.error(f"Aborting...")
        return 1
    logging.critical(f"This is cleaning step. It should be run after training update_iter")
    continuing = input(
        f"Are you sur? (Y for Yes, anything else to abort)"
    )
    if continuing == "Y":
        del continuing
    else:
        logging.error(f"Aborting...")
        return 1

    # ### Delete
    logging.info(f"Deleting DP-Freeze related error files...")
    remove_file_glob(current_apath, "**/graph*freeze.out")
    logging.info(f"Deleting DP-Compress related error files...")
    remove_file_glob(current_apath, "**/graph*compress.out")
    logging.info(f"Deleting DP-Train related error files...")
    remove_file_glob(current_apath, "**/training.out")
    logging.info(f"Deleting SLURM launch files...")
    remove_file_glob(current_apath, "**/*.sh")

    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a success !"
    )
    
    # ### Cleaning
    del control_apath
    del current_iteration_zfill
    del training_json
    del training_iterative_apath, current_apath

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "clean",
            Path(sys.argv[1]),
            fake_cluster=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
