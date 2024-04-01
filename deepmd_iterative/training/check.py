"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/04/01
"""

# Standard library modules
import logging
import re
import sys
from pathlib import Path

# Non-standard imports
import numpy as np

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.list import textfile_to_string_list
from deepmd_iterative.common.json import load_json_file, write_json_file, find_key_in_dict


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_input_json_filename: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}.")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Get control path, load the main JSON and the training JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    training_json = load_json_file((control_path / f"training_{padded_curr_iter}.json"))

    # Check if we can continue
    if not training_json["is_launched"]:
        logging.error(f"Lock found. Please execute 'training launch' first.")
        logging.error(f"Aborting...")
        return 1

    # Check the normal termination of the training phase
    # Counters
    # s_per_step_per_step_size = []
    training_times = []
    step_sizes = []
    completed_count = 0
    min_nbor_dist = None
    max_nbor_size = None
    training_input_json = None

    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"
        if (local_path / "training.log").is_file():
            training_out = textfile_to_string_list((local_path / "training.log"))
        elif (local_path / "training.out").is_file():
            training_out = textfile_to_string_list((local_path / "training.out"))
        else:
            training_out = []
        if training_out:
            # Finished correctly
            if any("finished training" in s for s in training_out):
                training_out_time = [s for s in training_out if "training time" in s]

                batch_pattern = r"batch\s*(\d+)\s"
                time_pattern = r"training time (\d+\.\d+) s"

                if min_nbor_dist is None or max_nbor_size is None:
                    for log_text in training_out:
                        if "min nbor dist" in log_text:
                            min_nbor_dist_match = re.search(r"min nbor dist: ([\d\.]+)", log_text)
                            if min_nbor_dist_match:
                                min_nbor_dist = float(min_nbor_dist_match.group(1))
                        elif "max nbor size" in log_text:
                            max_nbor_size_match = re.search(r"max nbor size: \[([ \d]+)\]", log_text)
                            if max_nbor_size_match:
                                max_nbor_size = [int(n) for n in max_nbor_size_match.group(1).split()]

                if training_input_json is None:
                    training_input_json = load_json_file(local_path / "training.json")

                batch_numbers = []

                for entry in training_out_time:
                    batch_match = re.search(batch_pattern, entry)
                    time_match = re.search(time_pattern, entry)

                    if batch_match and time_match:
                        batch_number = int(batch_match.group(1))
                        training_time = float(time_match.group(1))

                        batch_numbers.append(batch_number)
                        training_times.append(training_time)

                del entry, batch_match, time_match, batch_number, training_time
                del time_pattern, batch_pattern

                for suffix in ["index", "meta", "data-00000-of-00001"]:
                    if (local_path / f"model.ckpt-{batch_numbers[-1]}.{suffix}").is_file():
                        (local_path / f"model.ckpt-{batch_numbers[-1]}.{suffix}").rename(local_path / f"model.ckpt.{suffix}")
                del suffix

                step_sizes.extend(np.diff(batch_numbers))
                del batch_numbers
                completed_count += 1
            else:
                logging.critical(f"DP Train - '{nnp}' not finished/failed.")
            del training_out, training_out_time
        else:
            logging.critical(f"DP Train - '{nnp}' still running/no outfile.")
        del local_path
    del nnp
    logging.debug(f"completed_count: {completed_count}")

    # Infos
    if min_nbor_dist is not None:
        training_json["min_nbor_dist"] = min_nbor_dist
        logging.info(f"Your minimum neighbor distance is: {min_nbor_dist:.3f}")
        if min_nbor_dist < 0.7:
            logging.warning(f"Your minimum neighbor distance is lower than 0.1 Angstrom.")
            logging.warning(f"You might have a funky system.")

    if max_nbor_size is not None:
        training_json["max_nbor_size"] = max_nbor_size
        logging.info(f"In the training datasets, the maximum number of type-i neighbors of an atom is: {max_nbor_size}")
        logging.info(f"Your type map was: {main_json['type_map']}")
        logging.info(f"The total is: {sum(max_nbor_size)}")
        selection_list = find_key_in_dict(training_input_json, "sel")
        logging.info(f"In the training parameters, the expected maximum number of type-i neighbors of an atom was: {selection_list[0]} (keyword 'sel').")
        if sum(max_nbor_size) > sum(selection_list[0]):
            logging.warning(f"The maximum number of type-i neighbors of an atom is higher than the expected maximum number of type-i neighbors of an atom (keyword 'sel').")
            logging.warning(f"Please correct this.")
        if sum(selection_list[0]) > 2.0 * sum(max_nbor_size):
            logging.warning(f"The expected maximum number of type-i neighbors of an atom is at least 100% larger that the ones present in the training datasets.")
            logging.warning(f"You may want to decrease the expected maximum number of type-i neighbors of an atom (keyword 'sel').")


    logging.info(f"-" * 88)
    # Update the boolean in the training JSON
    if completed_count == main_json["nnp_count"]:
        training_json["is_checked"] = True

    # If not empty
    if training_times and step_sizes:
        training_json["mean_s_per_step"] = np.average(training_times) / np.average(step_sizes)
        training_json["median_s_per_step"] = np.median(training_times) / np.average(step_sizes)
        training_json["stdeviation_s_per_step"] = np.std(training_times) / np.average(step_sizes)
    else:
        training_json["mean_s_per_step"] = -1
        training_json["median_s_per_step"] = -1
        training_json["stdeviation_s_per_step"] = -1

    logging.debug(f"mean_s_per_step: {training_json['mean_s_per_step']}")
    logging.debug(f"median_s_per_step: {training_json['median_s_per_step']}")
    logging.debug(f"stdeviation_s_per_step: {training_json['stdeviation_s_per_step']}")

    del training_times, step_sizes
    # Dump the JSON files (training)
    write_json_file(training_json, (control_path / f"training_{padded_curr_iter}.json"), read_only=True)

    # End
    logging.info(f"-" * 88)
    if completed_count == main_json["nnp_count"]:
        logging.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")
    else:
        logging.critical(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a failure!")
        logging.critical(f"Some DP Train did not finished correctly.")
        logging.critical(f"Please check manually before re-exectuing this step.")
        logging.critical(f"Aborting...")
        return 1
    del completed_count

    # Cleaning
    del current_path, control_path, training_path
    del user_input_json_filename
    del main_json, training_json
    del curr_iter, padded_curr_iter

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "check",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
