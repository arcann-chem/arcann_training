from pathlib import Path
import logging
import sys
import subprocess

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    json_read
)
from deepmd_iterative.common.files import (
    remove_file_glob
)


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
    if step_name not in current_apath.name:
        logging.error(f"The folder doesn't seems to be for this step: {step_name.capitalize()}")
        logging.error(f"Aborting...")
        return 1

    # ### Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)

    # ### Get control path and config_json
    control_apath = training_iterative_apath / "control"
    config_json = json_read((control_apath / "config.json"), True, True)
    training_json = json_read(
        (control_apath / f"training_{current_iteration_zfill}.json"), True, True
    )

    # ### Checks
    logging.critical(f"This is cleaning step. It should be run after training update_iter")
    continuing = input(
        f"Are you sur? (Y for Yes, anything else to abort)"
    )
    if continuing == "Y":
        del continuing
        True
    else:
        logging.error(f"Aborting...")
        return 1

    # ### Delete
    logging.info("Deleting DP-Freeze related error files...")
    remove_file_glob(current_apath,"**/graph*freeze.out")
    logging.info("Deleting DP-Compress related error files...")
    remove_file_glob(current_apath,"**/graph*compress.out")
    logging.info("Deleting DP-Train related error files...")
    remove_file_glob(current_apath,"**/training.out")
    logging.info("Deleting SLURM launch files...")
    remove_file_glob(current_apath,"**/*.sh")

    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
    )
    
    # ### Cleaning
    del control_apath
    del config_json
    del current_iteration, current_iteration_zfill
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
