"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/10/03
"""
# Standard library modules
import subprocess
import logging
import sys
from pathlib import Path

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.filesystem import (
    change_directory,
    remove_files_matching_glob,
    remove_all_symlink,
)
from deepmd_iterative.common.json import load_json_file


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_config_filename: str = "input.json",
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

    # Get the previous
    prev_iter = curr_iter - 1
    padded_prev_iter = str(prev_iter).zfill(3)

    # Get control path and load the main config (JSON) and the training config (JSON)
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))
    exploration_config = load_json_file(
        (control_path / f"exploration_{padded_curr_iter}.json")
    )

    # Check if we can continue and ask the user
    if not exploration_config["is_extracted"]:
        logging.error(f"Lock found. Please execute 'exploration extract' first.")
        logging.error(f"Aborting...")
        return 1
    logging.critical(f"This is the cleaning step for exploration step.")
    logging.critical(f"It should be run after exploration extract phase.")
    logging.critical(f"This is will delete:")
    logging.critical(f"symbolic links, 'job_*.sh', '*.in', '*.lmp', 'plumed_*.dat'")
    logging.critical(
        f"LAMMPS_*, 'i-PI_DeepMD*', '*.DP-i-PI.client_*.log', '*.DP-i-PI.client_*.err', 'plumed_*.dat'"
    )
    logging.critical(f"in the folder: '{current_path}' and all subdirectories.")
    logging.critical(
        f"It will also create a tar.bz2 file with all starting structures from the previous exploration"
    )
    continuing = input(
        f"Do you want to continue? [Enter 'Y' for yes, or any other key to abort]: "
    )
    if continuing == "Y":
        del continuing
    else:
        logging.error(f"Aborting...")
        return 0

    # TODO Check for i-pi what to delete
    # Delete
    logging.info("Deleting symbolic links...")
    remove_all_symlink(current_path)
    logging.info("Deleting job files...")
    remove_files_matching_glob(current_path, "**/job_*.sh")
    logging.info("Deleting exploration input files...")
    remove_files_matching_glob(current_path, "**/*.in")
    logging.info("Deleting exploration input structure files...")
    remove_files_matching_glob(current_path, "**/*.lmp")
    logging.info("Deleting exploration plumed input files...")
    remove_files_matching_glob(current_path, "**/plumed*.dat")
    logging.info("Deleting job error files...")
    remove_files_matching_glob(current_path, "**/LAMMPS_*")
    remove_files_matching_glob(current_path, "**/i-PI_DeepMD*")
    logging.info("Deleting extra files...")
    remove_files_matching_glob(current_path, "**/*.DP-i-PI.client_*.log")
    remove_files_matching_glob(current_path, "**/*.DP-i-PI.client_*.err")

    if prev_iter > 0:
        logging.info(f"Compressing into a bzip2 tar archive...")
        change_directory(training_path / "starting_structures")
        starting_structures_xyz = list(Path(".").glob(f"{padded_prev_iter}_*.xyz"))
        starting_structures_lmp = list(Path(".").glob(f"{padded_prev_iter}_*.lmp"))
        starting_structures = starting_structures_xyz + starting_structures_lmp
        archive_name = f"starting_structures_{padded_prev_iter}.tar.bz2"
        if starting_structures:
            if (Path(".") / archive_name).is_file():
                logging.info(f"{archive_name} already present, adding .bak extension")
                (Path(".") / f"{archive_name}.bak").write_bytes(
                    (Path(".") / archive_name).read_bytes()
                )
            cmd = ["tar", "-I", "bzip2", "--exclude=*.tar.bz2", "-cf", archive_name]

            # Convert Path objects to string and append to cmd list
            cmd.extend(map(str, starting_structures))
            subprocess.run(cmd)
            del starting_structures, starting_structures_xyz, starting_structures_lmp
            logging.info(
                f"If the tar.bz2 is good, you can remove all files starting with {padded_prev_iter}_ in {training_path / 'starting_structures'}"
            )
        change_directory(current_path)

    logging.info(f"Cleaning done!")

    # End
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del user_config_filename
    del main_config, exploration_config
    del curr_iter, padded_curr_iter

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "clean",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_config_filename=sys.argv[3],
        )
    else:
        pass
