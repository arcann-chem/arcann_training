"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/31
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
from deepmd_iterative.common.json import load_json_file, write_json_file


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
    testing_json = load_json_file((control_path / f"testing_{padded_curr_iter}.json"))
    training_json = load_json_file((control_path / f"training_{padded_curr_iter}.json"))

    # Check if we can continue
    if not testing_json["is_launched"]:
        logging.error(f"Lock found. Please execute 'training launch' first.")
        logging.error(f"Aborting...")
        return 1

    # Regular expressions for each value
    patterns = {"energy_rmse": "Energy RMSE\s+:\s+([\d\.e\+\-]+)\s+eV", "energy_rmse_per_atom": "Energy RMSE/Natoms\s+:\s+([\d\.e\+\-]+)\s+eV", "force_rmse": "Force\s+RMSE\s+:\s+([\d\.e\+\-]+)\s+eV/A", "virial_rmse": "Virial RMSE\s+:\s+([\d\.e\+\-]+)\s+eV", "virial_rmse_per_atom": "Virial RMSE/Natoms\s+:\s+([\d\.e\+\-]+)\s+eV", "number_of_test_data": "# number of test data\s+:\s+(\d+)"}

    completed_count = 0

    datasets = [_.stem for _ in (current_path / "data").iterdir()]

    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"

        if testing_json["is_compressed"]:
            nnp_name = f"graph_{nnp}_{padded_curr_iter}_compressed"
        else:
            nnp_name = f"graph_{nnp}_{padded_curr_iter}"

        for dataset in datasets:
            logging.debug(f"Processing '{nnp}' for '{dataset}'.")

            if (local_path / f"{dataset}.log").is_file():
                testing_out = textfile_to_string_list((local_path / f"{dataset}.log"))
            elif (local_path / f"{dataset}.out").is_file():
                testing_out = textfile_to_string_list((local_path / f"{dataset}.out"))
            else:
                logging.critical(f"DP Test - '{nnp}' for '{dataset}' still running/no outfile.")
                continue

            if testing_out:
                if any("output of dp test" in s for s in testing_out):
                    # Combining the list into a single string
                    testing_out_combined = "\n".join(testing_out)
                    extracted_values_from_list = {}
                    for key, pattern in patterns.items():
                        match = re.search(pattern, testing_out_combined)
                        if match:
                            extracted_values_from_list[key] = float(match.group(1)) if "test data" not in key else int(match.group(1))
                            completed_count += 1
                        else:
                            extracted_values_from_list[key] = False
                            logging.critical(f"DP Test - '{nnp}' for '{dataset}': value {key} not present.")

                    testing_json[nnp_name][dataset] = extracted_values_from_list
                    del testing_out_combined, extracted_values_from_list, match, key, pattern

                    if dataset in training_json["training_datasets"]:
                        testing_json[nnp_name][dataset]["trained"] = True
                    else:
                        testing_json[nnp_name][dataset]["trained"] = False

                    energy_file = local_path / f"{dataset}.e.out"
                    force_file = local_path / f"{dataset}.f.out"
                    virial_file = local_path / f"{dataset}.v.out"

                    for file in [energy_file, force_file, virial_file]:
                        if file.is_file():
                            np.save(file.with_suffix(".npy"), np.genfromtxt(file))
                            logging.info(f"DP Test - '{nnp}' for '{dataset}' - '{file.stem}' saved as NPY.")
                    del energy_file, force_file, virial_file, file
                else:
                    testing_json[nnp_name][dataset] = False
                    logging.critical(f"DP Test - '{nnp}' for '{dataset}' not finished/failed.")
            else:
                testing_json[nnp_name][dataset] = False
                logging.critical(f"DP Test - '{nnp}' for '{dataset}' not finished/failed.")
            del testing_out
        del dataset
    del nnp, local_path, nnp_name

    if completed_count == main_json["nnp_count"] * len(datasets) * len(patterns):
        testing_json["is_checked"] = True

    # Dump the JSON files (training)
    write_json_file(testing_json, (control_path / f"testing_{padded_curr_iter}.json"), read_only=True)

    # End
    logging.info(f"-" * 88)
    logging.debug(f"completed_count: {completed_count}")
    logging.debug(f"expected: {main_json['nnp_count'] * len(datasets) * len(patterns)}")
    logging.debug(f"main_json['nnp_count']: {main_json['nnp_count']}")
    logging.debug(f"len(datasets): {len(datasets)}")
    logging.debug(f"len(patterns): {len(patterns)}")

    if completed_count == main_json["nnp_count"] * len(datasets) * len(patterns):
        logging.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")
    else:
        logging.critical(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a failure!")
        logging.critical(f"Some DP Test did not finished correctly.")
        logging.critical(f"Please check manually before re-exectuing this step.")
        logging.critical(f"Aborting...")
        return 1
    del completed_count

    # Cleaning
    del current_path, control_path, training_path
    del user_input_json_filename
    del main_json, training_json, testing_json
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
