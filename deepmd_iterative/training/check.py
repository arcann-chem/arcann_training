"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/30
"""
# Standard library modules
import logging
import sys
from pathlib import Path

# Non-standard imports
import numpy as np

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.list import (
    textfile_to_string_list,
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

    # Get control path, load the main config JSON and the training config JSON
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))
    training_config = load_json_file(
        (control_path / f"training_{padded_curr_iter}.json")
    )

    # Check if we can continue
    if not training_config['is_launched']:
        logging.error(f"Lock found. Execute first: training launch")
        logging.error(f"Aborting...")
        return 1

    # Check the normal termination of the training phase
    # Counters
    s_per_step_per_step_size = []
    completed_count = 0
    for nnp in range(1, main_config['nnp_count'] + 1):
        local_path = Path(".").resolve() / f"{nnp}"
        if (local_path / "training.out").is_file():
            training_out = textfile_to_string_list((local_path / "training.out"))
            if any("finished training" in s for s in training_out):
                training_out_time = [s for s in training_out if "training time" in s]
                training_out_time_split = []
                for n in range(0, len(training_out_time)):
                    training_out_time_split.append(training_out_time[n].split(" "))
                    training_out_time_split[n] = " ".join(
                        training_out_time_split[n]
                    ).split()
                if (
                    local_path / f"model.ckpt-{training_out_time_split[-1][3]}.index"
                ).is_file():
                    (
                        local_path
                        / f"model.ckpt-{training_out_time_split[-1][3]}.index"
                    ).rename(local_path / "model.ckpt.index")
                    (
                        local_path / f"model.ckpt-{training_out_time_split[-1][3]}.meta"
                    ).rename(local_path / "model.ckpt.meta")
                    (
                        local_path
                        / f"model.ckpt-{training_out_time_split[-1][3]}.data-00000-of-00001"
                    ).rename(local_path / "model.ckpt.data-00000-of-00001")
                for n in range(0, len(training_out_time_split)):
                    s_per_step_per_step_size.append(
                        float(training_out_time_split[n][6])
                    )
                del n
                step_size = float(training_out_time_split[-1][3]) - float(
                    training_out_time_split[-2][3]
                )
                completed_count += 1
            else:
                logging.critical(f"DP Train - '{nnp}' not finished/failed")
            del training_out, training_out_time, training_out_time_split
        else:
            logging.critical(f"DP Train - '{nnp}' still running/no outfile")
        del local_path
    del nnp
    logging.debug(f"completed_count: {completed_count}")

    logging.info(f"-" * 88)
    # Update the boolean in the training config JSON
    if completed_count == main_config['nnp_count']:
        training_config['is_checked'] = True

    if ("s_per_step_per_step_size" in locals()) and ("step_size" in locals()):
        training_config['s_per_step'] = np.average(s_per_step_per_step_size) / step_size
        del s_per_step_per_step_size, step_size

    # Dump the JSON (training config JSON)
    write_json_file(
        training_config, (control_path / f"training_{padded_curr_iter}.json")
    )
    
    # End
    logging.info(f"-" * 88)
    if completed_count == main_config['nnp_count']:
        logging.info(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
        )
    else:
        logging.critical(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a failure!"
        )
        logging.critical(f"Some DP Train did not finished correctly")
        logging.critical(f"Please check manually before relaunching this step")
        logging.critical(f"Aborting...")
        return 1
    del completed_count

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
            "check",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_config_filename=sys.argv[3],
        )
    else:
        pass
