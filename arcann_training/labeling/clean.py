"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15
"""

# Standard library modules
import logging
import sys
from pathlib import Path
import subprocess

# Local imports
from arcann_training.common.check import validate_step_folder
from arcann_training.common.filesystem import (
    remove_file,
    remove_files_matching_glob,
    remove_all_symlink,
)
from arcann_training.common.list import string_list_to_textfile
from arcann_training.common.json import load_json_file


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_config_filename: str = "input.json",
):
    # Get the logger
    arcann_logger = logging.getLogger("ArcaNN")

    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    arcann_logger.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}."
    )
    arcann_logger.debug(f"Current path :{current_path}")
    arcann_logger.debug(f"Training path: {training_path}")
    arcann_logger.debug(f"Program path: {deepmd_iterative_path}")
    arcann_logger.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Get control path and load the main config (JSON) and the training config (JSON)
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))
    labeling_config = load_json_file(
        (control_path / f"labeling_{padded_curr_iter}.json")
    )

    # Check if we can continue
    if not labeling_config["is_extracted"]:
        arcann_logger.error(f"Lock found. Please execute 'labeling extract' first.")
        arcann_logger.error(f"Aborting...")
        return 1
    arcann_logger.warning(f"This is the cleaning step for labeling step.")
    arcann_logger.warning(f"It should be run after labeling extract phase.")
    arcann_logger.warning(f"This is will delete:")
    arcann_logger.warning(
        f"symbolic links, 'job_*.sh', 'job-array_*.sh', 'labeling_*.xyz', 'labeling_*-SCF.wfn', '*labeling*.inp'"
    )
    arcann_logger.warning(f"CP2K.*")
    arcann_logger.warning(f"in the folder: '{current_path}' and all subdirectories.")
    arcann_logger.warning(
        f"It will also create an tar.bz2 file with all important files (except the binary wavefunction files)."
    )

    continuing = input(
        f"Do you want to continue? [Enter 'Y' for yes, or any other key to abort]: "
    )
    if continuing == "Y":
        del continuing
    else:
        arcann_logger.error(f"Aborting...")
        return 1

    # Delete
    arcann_logger.info("Deleting symbolic links...")
    remove_all_symlink(current_path)
    arcann_logger.info("Deleting job files...")
    remove_files_matching_glob(current_path, "**/job_*.sh")
    remove_files_matching_glob(current_path, "**/job-array_*.sh")
    remove_files_matching_glob(current_path, "**/job-array-params*.lst")
    arcann_logger.info(f"Deleting XYZ input files...")
    remove_files_matching_glob(current_path, "**/labeling_*.xyz")
    arcann_logger.info(f"Deleting WFN temporary files...")
    remove_files_matching_glob(current_path, "**/labeling_*-SCF.wfn")
    arcann_logger.info(f"Deleting labeling input files...")
    remove_files_matching_glob(current_path, "**/*labeling*.inp")
    arcann_logger.info("Deleting job error files...")
    remove_files_matching_glob(current_path, "**/CP2K.*")
    remove_files_matching_glob(current_path, "**/ORCA.*")
    arcann_logger.info(f"Cleaning done!")
    arcann_logger.info(f"Compressing into a bzip2 tar archive...")

    archive_name = f"labeling_{padded_curr_iter}_noWFN.tar.bz2"

    all_files = Path(".").glob("**/*")
    files = [
        str(file)
        for file in all_files
        if (
            file.is_file()
            and (
                not (
                    file.name.endswith(".wfn")
                    or file.name.endswith(".gbw")
                    or ".tar" in file.name
                )
            )
        )
    ]
    if files:
        if (Path(".") / archive_name).is_file():
            arcann_logger.info(f"{archive_name} already present, adding .bak extension")
            (Path(".") / f"{archive_name}.bak").write_bytes(
                (Path(".") / archive_name).read_bytes()
            )

        string_list_to_textfile(
            current_path / archive_name.replace(".tar.bz2", ".lst"), files
        )
        cmd = [
            "tar",
            "-I",
            "bzip2",
            "--exclude=*.tar.bz2",
            "-cf",
            archive_name,
            "-T",
            archive_name.replace(".tar.bz2", ".lst"),
        ]
        subprocess.run(cmd)
        remove_file(current_path / archive_name.replace(".tar.bz2", ".lst"))

    del archive_name, files, all_files

    arcann_logger.info(f"Done!")
    arcann_logger.info(
        f"Please note that the wavefunction files are not included in the archive."
    )
    if labeling_config["labeling_program"] == "cp2k":
        arcann_logger.info(
            f"To keep only the wavefunction files from the 2nd step (your reference) in a labeling_{padded_curr_iter}_WFN.tar, execute:"
        )
        arcann_logger.info(
            f"\"find ./ -name '2_*.wfn' | tar -cf labeling_{padded_curr_iter}_WFN.tar --files-from -\" (without the double quotes)."
        )
        arcann_logger.info(
            f"To keep all wavefunction files in a labeling_{padded_curr_iter}_WFN.tar, execute:"
        )
        arcann_logger.info(
            f"\"find ./ -name '*.wfn' | tar -cf labeling_{padded_curr_iter}_WFN.tar --files-from -\" (without the double quotes)."
        )
        arcann_logger.info(f"For cp2k labeling, wavefunction files are in .wfn format.")
        arcann_logger.info(
            f"You can delete any subsys subfolders if labeling_{padded_curr_iter}_noWFN.tar.bz2 is okay, and you have saved or don't need the wavefunction files."
        )

    elif labeling_config["labeling_program"] == "orca":
        arcann_logger.info(
            f"To keep only the wavefunction files from the 1st and only step (your reference) in a labeling_{padded_curr_iter}_WFN.tar, execute:"
        )
        arcann_logger.info(
            f"\"find ./ -name '1_*.gbw' | tar -cf labeling_{padded_curr_iter}_WFN.tar --files-from -\" (without the double quotes)."
        )
        arcann_logger.info(f"For orca labeling, wavefunction files are in .gbw format.")
        arcann_logger.info(
            f"You can delete any subsys subfolders if labeling_{padded_curr_iter}_noWFN.tar.bz2 is okay, and you have saved or don't need the wavefunction files."
        )

    # End
    arcann_logger.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del user_config_filename
    del main_config, labeling_config
    del curr_iter, padded_curr_iter

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
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
