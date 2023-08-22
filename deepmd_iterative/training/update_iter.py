"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/22
"""
# Standard library modules
import logging
import subprocess
import sys
from pathlib import Path

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.filesystem import (
    check_directory,
    check_file_existence,
    remove_file,
    remove_files_matching_glob,
    remove_tree,
)
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
)


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
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}"
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

    # Get control path and load the main config (JSON) and the training config (JSON)
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))
    training_config = load_json_file(
        (control_path / f"training_{padded_curr_iter}.json")
    )

    # Check if we can continue
    if not training_config["is_frozen"]:
        logging.error(f"Lock found. Execute first: training check_freeze")
        logging.error(f"Aborting...")
        return 1

    # Check if pb files are present and delete temp files
    for nnp in range(1, main_config["nnp_count"] + 1):
        local_path = Path(".").resolve() / f"{nnp}"
        check_file_existence(local_path / f"graph_{nnp}_{padded_curr_iter}.pb")
        if training_config["is_compressed"]:
            check_file_existence(
                local_path / f"graph_{nnp}_{padded_curr_iter}_compressed.pb"
            )

        remove_file(local_path / "checkpoint")
        remove_file(local_path / "input_v2_compat.json")
        logging.info("Deleting SLURM out/error files...")
        remove_files_matching_glob(local_path, "DeepMD_*")
        logging.info("Deleting the previous model.ckpt...")
        remove_files_matching_glob(local_path, "model.ckpt-*")
        if (local_path / "model-compression").is_dir():
            logging.info("Deleting the temp model-compression folder...")
            remove_tree(local_path / "model-compression")

    # Prepare the test folder
    (training_path / f"{padded_curr_iter}-test").mkdir(exist_ok=True)
    check_directory((training_path / f"{padded_curr_iter}-test"))

    subprocess.run(
        [
            "rsync",
            "-a",
            f"{training_path / 'data'}",
            str(training_path / f"{padded_curr_iter}-test"),
        ]
    )

    # Copy the pb files to the NNP meta folder
    (training_path / "NNP").mkdir(exist_ok=True)
    check_directory(training_path / "NNP")

    local_path = Path(".").resolve()

    for nnp in range(1, main_config["nnp_count"] + 1):
        if training_config["is_compressed"]:
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    str(
                        local_path
                        / f"{nnp}"
                        / f"graph_{nnp}_{padded_curr_iter}_compressed.pb"
                    ),
                    str((training_path / "NNP")),
                ]
            )
        subprocess.run(
            [
                "rsync",
                "-a",
                str(local_path / f"{nnp}" / f"graph_{nnp}_{padded_curr_iter}.pb"),
                str((training_path / "NNP")),
            ]
        )
    del nnp

    # Next iteration
    curr_iter = curr_iter + 1
    main_config["curr_iter"] = curr_iter
    padded_curr_iter = str(curr_iter).zfill(3)

    for step in ["exploration", "adhoc", "labeling", "training"]:
        (training_path / f"{padded_curr_iter}-{step}").mkdir(exist_ok=True)
        check_directory(training_path / f"{padded_curr_iter}-{step}")
    del step

    # Delete the temp data folder
    if (local_path / "data").is_dir():
        logging.info("Deleting the temp data folder...")
        remove_tree(local_path / "data")
        logging.info("Cleaning done!")
    del local_path

    # Update the config.json
    write_json_file(main_config, (control_path / "config.json"))

    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a succes !"
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
            "update_iter",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_config_filename=sys.argv[3],
        )
    else:
        pass
