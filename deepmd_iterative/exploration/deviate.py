"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/01
"""

# Standard library modules
import copy
import logging
import sys
from pathlib import Path

# Non-standard library imports
import numpy as np

# Local imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    load_default_json_file,
    backup_and_overwrite_json_file,
)
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.exploration.utils import (
    get_last_frame_number,
    generate_input_exploration_deviation_json,
    get_system_deviation,
)
from deepmd_iterative.common.xyz import parse_xyz_trajectory_file

def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_input_json_filename: str = "input.json",
):
    # Get the logger
    arcann_logger = logging.getLogger("ArcaNN")

    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    arcann_logger.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}.")
    arcann_logger.debug(f"Current path :{current_path}")
    arcann_logger.debug(f"Training path: {training_path}")
    arcann_logger.debug(f"Program path: {deepmd_iterative_path}")
    arcann_logger.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Load the default input JSON
    default_input_json = load_default_json_file(deepmd_iterative_path / "assets" / "default_config.json")[current_step]
    default_input_json_present = bool(default_input_json)
    arcann_logger.debug(f"default_input_json: {default_input_json}")
    arcann_logger.debug(f"default_input_json_present: {default_input_json_present}")

    # Load the user input JSON
    if (current_path / user_input_json_filename).is_file():
        user_input_json = load_json_file((current_path / user_input_json_filename))
    else:
        user_input_json = {}
    user_input_json_present = bool(user_input_json)
    arcann_logger.debug(f"user_input_json: {user_input_json}")
    arcann_logger.debug(f"user_input_json_present: {user_input_json_present}")

    # If the used input JSON is present, load it
    if (current_path / "used_input.json").is_file():
        current_input_json = load_json_file((current_path / "used_input.json"))
    else:
        arcann_logger.warning(f"No used_input.json found. Starting with empty one.")
        arcann_logger.warning(f"You should avoid this by not deleting the used_input.json file.")
        current_input_json = {}
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    # Get control path, load the main JSON and the exploration JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    exploration_json = load_json_file((control_path / f"exploration_{padded_curr_iter}.json"))

    # Load the previous exploration JSON and training JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_training_json = load_json_file((control_path / f"training_{padded_prev_iter}.json"))
        if prev_iter > 0:
            previous_exploration_json = load_json_file((control_path / ("exploration_" + padded_prev_iter + ".json")))
        else:
            previous_exploration_json = {}
    else:
        previous_training_json = {}
        previous_exploration_json = {}

    # Check if we can continue
    if not exploration_json["is_checked"]:
        arcann_logger.error(f"Lock found. Execute first: exploration check.")
        arcann_logger.error(f"Aborting...")
        return 1
    if "is_rechecked" in exploration_json and not exploration_json["is_rechecked"]:
        arcann_logger.critical(f"Lock found. Execute first: exploration recheck.")
        arcann_logger.critical(f"Aborting...")
        return 1

    # Generate/update the merged input JSON
    # Priority: user > previous > default
    current_input_json = generate_input_exploration_deviation_json(
        user_input_json,
        previous_exploration_json,
        default_input_json,
        current_input_json,
        main_json,
    )
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    for system_auto_index, system_auto in enumerate(main_json["systems_auto"]):
        # Set the system params for deviation selection
        (
            max_candidates,
            sigma_low,
            sigma_high,
            sigma_high_limit,
            ignore_first_x_ps,
        ) = get_system_deviation(current_input_json, system_auto_index)

        # Initialize
        exploration_json["systems_auto"][system_auto] = {
            **exploration_json["systems_auto"][system_auto],
            "max_candidates": max_candidates,
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
            "sigma_high_limit": sigma_high_limit,
            "ignore_first_x_ps": ignore_first_x_ps,
            "mean_deviation_max_f": 0,
            "median_deviation_max_f": 0,
            "stdeviation_deviation_max_f": 0,
            "total_count": 0,
            "candidates_count": 0,
            "rejected_count": 0,
        }

        skipped_traj_user = 0
        skipped_traj_stats = 0
        start_row_number = 0

        arcann_logger.debug(f"{exploration_json['systems_auto'][system_auto]['print_every_x_steps']},{exploration_json['systems_auto'][system_auto]['timestep_ps']}")
        arcann_logger.debug(f"{exploration_json['systems_auto'][system_auto]['ignore_first_x_ps']}")

        while start_row_number * exploration_json["systems_auto"][system_auto]["print_every_x_steps"] * exploration_json["systems_auto"][system_auto]["timestep_ps"] < exploration_json["systems_auto"][system_auto]["ignore_first_x_ps"]:
            start_row_number = start_row_number + 1

        arcann_logger.debug(f"start_row_number: {start_row_number}")
        if start_row_number > exploration_json["systems_auto"][system_auto]["nb_steps"] // exploration_json["systems_auto"][system_auto]["print_every_x_steps"]:
            start_row_number = 0
            arcann_logger.warning(f"Your 'ignore_first_x_ps' value is too high.")
            arcann_logger.warning(f"Please reduce it to a value lower than {start_row_number * exploration_json['systems_auto'][system_auto]['print_every_x_steps'] * exploration_json['systems_auto'][system_auto]['timestep_ps']}.")
            arcann_logger.warning(f"Temporarily setting it to 0.")

        for it_nnp in range(1, main_json["nnp_count"] + 1):
            for it_number in range(1, exploration_json["systems_auto"][system_auto]["traj_count"] + 1):
                arcann_logger.debug(f"{system_auto} / {it_nnp} / {it_number}")

                # Get the local path and the name of model_deviation file
                local_path = Path(".").resolve() / str(system_auto) / str(it_nnp) / str(it_number).zfill(5)
                model_deviation_filename = f"model_devi_{system_auto}_{it_nnp}_{padded_curr_iter}.out"
                xyz_qm_filename = f"{system_auto}_{it_nnp}_{padded_curr_iter}_QM.xyz"

                # Create the JSON data for Query-by-Committee
                QbC_stats = load_json_file(local_path / "QbC_stats.json", False, False)
                QbC_indexes = load_json_file(local_path / "QbC_indexes.json", False, False)
                QbC_stats = {
                    **QbC_stats,
                    "sigma_low": sigma_low,
                    "sigma_high": sigma_high,
                    "sigma_high_limit": sigma_high_limit,
                }

                # Get the number of exptected steps
                nb_steps_expected = (exploration_json["systems_auto"][system_auto]["nb_steps"] // exploration_json["systems_auto"][system_auto]["print_every_x_steps"]) + 1 - start_row_number
                arcann_logger.debug(f"nb_steps_expected: {nb_steps_expected}")

                # If it was not skipped
                if not (local_path / "skip").is_file():
                    if exploration_json["systems_auto"][system_auto]["exploration_type"] == "sander_emle":
                        num_atoms, atom_symbols, atom_coords, comments, cell_info, pbc_info, properties_info, max_f_std_info = parse_xyz_trajectory_file(local_path / xyz_qm_filename)
                        model_deviation = np.vstack(([_ for _ in range(0, len(max_f_std_info), exploration_json["systems_auto"][system_auto]["print_every_x_steps"])], max_f_std_info)).T
                        total_row_number = len(max_f_std_info)
                    elif exploration_json["systems_auto"][system_auto]["exploration_type"] == "lammps" or exploration_json["systems_auto"][system_auto]["exploration_type"] == "i-PI":
                        model_deviation = np.genfromtxt(str(local_path / model_deviation_filename))
                        if exploration_json["systems_auto"][system_auto]["exploration_type"] == "lammps":
                            total_row_number = model_deviation.shape[0]
                        elif exploration_json["systems_auto"][system_auto]["exploration_type"] == "i-PI":
                            total_row_number = model_deviation.shape[0] + 1
                    else:
                        arcann_logger.error("Unknown exploration type. Please BUG REPORT!")
                        arcann_logger.error("Aborting...")
                        return 1

                    if nb_steps_expected > (total_row_number - start_row_number):
                        QbC_stats["total_count"] = nb_steps_expected
                        arcann_logger.critical(f"Exploration '{system_auto}' / '{it_nnp}' / '{it_number}'.")
                        arcann_logger.critical(f"Mismatch between expected ('{nb_steps_expected}') number of steps.")
                        arcann_logger.critical(f"and actual ('{total_row_number - start_row_number}') number of steps in the deviation file.")
                        if (local_path / "force").is_file():
                            arcann_logger.warning("but it has been forced, so it should be ok.")
                    elif nb_steps_expected == (total_row_number - start_row_number):
                        QbC_stats["total_count"] = total_row_number - start_row_number
                    else:
                        arcann_logger.error("Unknown error. Please BUG REPORT!")
                        arcann_logger.error("Aborting...")
                        return 1

                    end_row_number = get_last_frame_number(
                        model_deviation,
                        sigma_high_limit,
                        exploration_json["systems_auto"][system_auto]["disturbed_start"],
                    )
                    arcann_logger.debug(f"end_row_number: {end_row_number}, start_row_number: {start_row_number}")
                    if (local_path / "force").is_file():
                        end_row_number = end_row_number - 1

                    if exploration_json["systems_auto"][system_auto]["exploration_type"] == "lammps" or exploration_json["systems_auto"][system_auto]["exploration_type"] == "i-PI":

                        # This part is when sigma_high_limit was never crossed
                        if end_row_number < 0:
                            mean_deviation_max_f = np.mean(model_deviation[start_row_number:, 4])
                            median_deviation_max_f = np.median(model_deviation[start_row_number:, 4])
                            stdeviation_deviation_max_f = np.std(model_deviation[start_row_number:, 4])
                            good = model_deviation[start_row_number:, :][model_deviation[start_row_number:, 4] <= sigma_low]
                            rejected = model_deviation[start_row_number:, :][model_deviation[start_row_number:, 4] >= sigma_high]
                            candidates = model_deviation[start_row_number:, :][(model_deviation[start_row_number:, 4] > sigma_low) & (model_deviation[start_row_number:, 4] < sigma_high)]

                        # This part is when sigma_high_limit was crossed during ignore_first_x_ps (SKIP everything for stats)
                        elif end_row_number <= start_row_number:
                            mean_deviation_max_f = 999.0
                            median_deviation_max_f = 999.0
                            stdeviation_deviation_max_f = 999.0
                            good = np.array([])
                            rejected = model_deviation[start_row_number:, :]
                            candidates = np.array([])
                            # In this case, it is skipped
                            skipped_traj_stats += 1

                        # This part is when sigma_high_limit was crossed (Gets stats before)
                        else:
                            mean_deviation_max_f = np.mean(model_deviation[start_row_number:end_row_number, 4])
                            median_deviation_max_f = np.median(model_deviation[start_row_number:end_row_number, 4])
                            stdeviation_deviation_max_f = np.std(model_deviation[start_row_number:end_row_number, 4])
                            good = model_deviation[start_row_number:end_row_number, :][model_deviation[start_row_number:end_row_number, 4] <= sigma_low]
                            rejected = model_deviation[start_row_number:end_row_number, :][model_deviation[start_row_number:end_row_number, 4] >= sigma_high]
                            candidates = model_deviation[start_row_number:end_row_number, :][(model_deviation[start_row_number:end_row_number, 4] > sigma_low) & (model_deviation[start_row_number:end_row_number, 4] < sigma_high)]
                            # Add the rest to rejected
                            rejected = np.vstack((rejected, model_deviation[end_row_number:, :]))

                    elif exploration_json["systems_auto"][system_auto]["exploration_type"] == "sander_emle":

                        # This part is when sigma_high_limit was never crossed
                        if end_row_number < 0:
                            mean_deviation_max_f = np.mean(model_deviation[start_row_number:])
                            median_deviation_max_f = np.median(model_deviation[start_row_number:])
                            stdeviation_deviation_max_f = np.std(model_deviation[start_row_number:])
                            good = model_deviation[start_row_number:, :][model_deviation[start_row_number:, 1] <= sigma_low]
                            rejected = model_deviation[start_row_number:, :][model_deviation[start_row_number:, 1] >= sigma_high]
                            candidates = model_deviation[start_row_number:, :][(model_deviation[start_row_number:, 1] > sigma_low) & (model_deviation[start_row_number:, 1] < sigma_high)]

                        # This part is when sigma_high_limit was crossed during ignore_first_x_ps (SKIP everything for stats)
                        elif end_row_number <= start_row_number:
                            mean_deviation_max_f = 999.0
                            median_deviation_max_f = 999.0
                            stdeviation_deviation_max_f = 999.0
                            good = np.array([])
                            rejected = model_deviation[start_row_number:, :]
                            candidates = np.array([])
                            # In this case, it is skipped
                            skipped_traj_stats += 1

                        # This part is when sigma_high_limit was crossed (Gets stats before)
                        else:
                            mean_deviation_max_f = np.mean(model_deviation[start_row_number:end_row_number])
                            median_deviation_max_f = np.median(model_deviation[start_row_number:end_row_number])
                            stdeviation_deviation_max_f = np.std(model_deviation[start_row_number:end_row_number])
                            good = model_deviation[start_row_number:end_row_number][model_deviation[start_row_number:end_row_number] <= sigma_low]
                            rejected = model_deviation[start_row_number:end_row_number][model_deviation[start_row_number:end_row_number] >= sigma_high]
                            candidates = model_deviation[start_row_number:end_row_number][(model_deviation[start_row_number:end_row_number] > sigma_low) & (model_deviation[start_row_number:end_row_number] < sigma_high)]
                            # Add the rest to rejected
                            rejected = np.vstack((rejected, model_deviation[end_row_number:, :]))
                    else:
                        arcann_logger.error("Unknown exploration type. Please BUG REPORT!")
                        arcann_logger.error("Aborting...")
                        return 1

                    # Fill JSON files
                    if exploration_json["systems_auto"][system_auto]["exploration_type"] == "sander_emle":
                        QbC_indexes = {
                            **QbC_indexes,
                            "good_indexes": good[:, 0].astype(int).tolist() if good.size > 0 else [],
                            "rejected_indexes": rejected[:, 0].astype(int).tolist() if rejected.size > 0 else [],
                            "candidate_indexes": candidates[:, 0].astype(int).tolist() if candidates.size > 0 else [],
                        }
                    elif exploration_json["systems_auto"][system_auto]["exploration_type"] == "lammps" or exploration_json["systems_auto"][system_auto]["exploration_type"] == "i-PI":
                        QbC_indexes = {
                            **QbC_indexes,
                            "good_indexes": good[:, 0].astype(int).tolist() if good.size > 0 else [],
                            "rejected_indexes": rejected[:, 0].astype(int).tolist() if rejected.size > 0 else [],
                            "candidate_indexes": candidates[:, 0].astype(int).tolist() if candidates.size > 0 else [],
                        }
                    else:
                        arcann_logger.error("Unknown exploration type. Please BUG REPORT!")
                        arcann_logger.error("Aborting...")
                        return 1

                    if exploration_json["systems_auto"][system_auto]["exploration_type"] == "sander_emle":
                        QbC_stats = {
                            **QbC_stats,
                            "mean_deviation_max_f": mean_deviation_max_f,
                            "median_deviation_max_f": median_deviation_max_f,
                            "stdeviation_deviation_max_f": stdeviation_deviation_max_f,
                            "good_count": len(good.astype(int).tolist()) if good.size > 0 else 0,
                            "rejected_count": len(rejected.astype(int).tolist()) if rejected.size > 0 else 0,
                            "candidates_count": len(candidates.astype(int).tolist()) if candidates.size > 0 else 0,
                        }
                    elif exploration_json["systems_auto"][system_auto]["exploration_type"] == "lammps" or exploration_json["systems_auto"][system_auto]["exploration_type"] == "i-PI":
                        QbC_stats = {
                            **QbC_stats,
                            "mean_deviation_max_f": mean_deviation_max_f,
                            "median_deviation_max_f": median_deviation_max_f,
                            "stdeviation_deviation_max_f": stdeviation_deviation_max_f,
                            "good_count": len(good[:, 0].astype(int).tolist()) if good.size > 0 else 0,
                            "rejected_count": len(rejected[:, 0].astype(int).tolist()) if rejected.size > 0 else 0,
                            "candidates_count": len(candidates[:, 0].astype(int).tolist()) if candidates.size > 0 else 0,
                        }
                    else:
                        arcann_logger.error("Unknown exploration type. Please BUG REPORT!")
                        arcann_logger.error("Aborting...")
                        return 1

                    # If the traj is smaller than expected (forced case) add the missing as rejected
                    if (QbC_stats["good_count"] + QbC_stats["rejected_count"] + QbC_stats["candidates_count"]) < nb_steps_expected:
                        QbC_stats["rejected_count"] = QbC_stats["rejected_count"] + nb_steps_expected - (QbC_stats["good_count"] + QbC_stats["rejected_count"] + QbC_stats["candidates_count"])

                    # Only if we have corect stats, add it
                    if (end_row_number > start_row_number) or (end_row_number == -1):
                        exploration_json["systems_auto"][system_auto]["mean_deviation_max_f"] = exploration_json["systems_auto"][system_auto]["mean_deviation_max_f"] + QbC_stats["mean_deviation_max_f"]
                        exploration_json["systems_auto"][system_auto]["median_deviation_max_f"] = exploration_json["systems_auto"][system_auto]["median_deviation_max_f"] + QbC_stats["median_deviation_max_f"]
                        exploration_json["systems_auto"][system_auto]["stdeviation_deviation_max_f"] = exploration_json["systems_auto"][system_auto]["stdeviation_deviation_max_f"] + QbC_stats["stdeviation_deviation_max_f"]
                    del end_row_number

                else:
                    ### If the trajectory was used skiped, count everything as a failure
                    skipped_traj_user = skipped_traj_user + 1
                    # Fill JSON files
                    QbC_indexes = {
                        **QbC_indexes,
                        "good_indexes": [],
                        "rejected_indexes": [],
                        "candidate_indexes": [],
                    }

                    QbC_stats = {
                        **QbC_stats,
                        "total_count": nb_steps_expected,
                        "mean_deviation_max_f": 999.0,
                        "median_deviation_max_f": 999.0,
                        "stdeviation_deviation_max_f": 999.0,
                        "good_count": 0,
                        "rejected_count": nb_steps_expected,
                        "candidates_count": 0,
                    }

                exploration_json["systems_auto"][system_auto]["total_count"] = exploration_json["systems_auto"][system_auto]["total_count"] + QbC_stats["total_count"]
                exploration_json["systems_auto"][system_auto]["candidates_count"] = exploration_json["systems_auto"][system_auto]["candidates_count"] + QbC_stats["candidates_count"]
                exploration_json["systems_auto"][system_auto]["rejected_count"] = exploration_json["systems_auto"][system_auto]["rejected_count"] + QbC_stats["rejected_count"]

                write_json_file(QbC_stats, local_path / "QbC_stats.json", False)
                write_json_file(QbC_indexes, local_path / "QbC_indexes.json", False)
                del (
                    local_path,
                    model_deviation_filename,
                    QbC_stats,
                    QbC_indexes,
                    nb_steps_expected,
                )

            del it_number

        # Average for the system (with adjustment, remove the skipped ones)
        exploitable_traj = (exploration_json["nnp_count"] * exploration_json["systems_auto"][system_auto]["traj_count"]) - (skipped_traj_user + skipped_traj_stats)
        if exploitable_traj == 0:
            arcann_logger.critical("All trajectories were skipped (either by you or discarded by ArcaNN).")
            arcann_logger.critical("You should either change the 'ignore_first_x_ps' value to a lower value to ensure some structure for this exploration.")
            arcann_logger.critical("For the next run (or if you redo one), you should reduce 'timestep_ps' and 'print_interval_mult'.")
            arcann_logger.critical("Aborting...")
            return 1
        else:
            if exploitable_traj / (exploration_json["nnp_count"] * exploration_json["systems_auto"][system_auto]["traj_count"]) < 0.25:
                arcann_logger.warning("Less than 25% of your exploration trajectories are exploitable. Be careful for the next one.")
            exploration_json["systems_auto"][system_auto]["mean_deviation_max_f"] = exploration_json["systems_auto"][system_auto]["mean_deviation_max_f"] / exploitable_traj
            exploration_json["systems_auto"][system_auto]["median_deviation_max_f"] = exploration_json["systems_auto"][system_auto]["median_deviation_max_f"] / exploitable_traj
            exploration_json["systems_auto"][system_auto]["stdeviation_deviation_max_f"] = exploration_json["systems_auto"][system_auto]["stdeviation_deviation_max_f"] / exploitable_traj

        del it_nnp
        del (
            max_candidates,
            sigma_low,
            sigma_high,
            sigma_high_limit,
            ignore_first_x_ps,
            skipped_traj_user,
            start_row_number,
        )

    del system_auto_index, system_auto

    total_candidates_selected = 0

    for system_auto_index, system_auto in enumerate(exploration_json["systems_auto"]):
        # Set the system params for deviation selection
        (
            max_candidates,
            sigma_low,
            sigma_high,
            sigma_high_limit,
            ignore_first_x_ps,
        ) = get_system_deviation(current_input_json, system_auto_index)

        # Initialize
        exploration_json["systems_auto"][system_auto] = {
            **exploration_json["systems_auto"][system_auto],
            "selected_count": 0,
            "discarded_count": 0,
        }

        for it_nnp in range(1, main_json["nnp_count"] + 1):
            for it_number in range(1, exploration_json["systems_auto"][system_auto]["traj_count"] + 1):
                # Get the local path and the name of model_deviation file
                local_path = Path(".").resolve() / str(system_auto) / str(it_nnp) / str(it_number).zfill(5)

                model_deviation_filename = f"model_devi_{system_auto}_{it_nnp}_{padded_curr_iter}.out"
                xyz_qm_filename = f"{system_auto}_{it_nnp}_{padded_curr_iter}_QM.xyz"

                # Create the JSON data for Query-by-Committee
                QbC_stats = load_json_file(local_path / "QbC_stats.json", True, False)
                QbC_indexes = load_json_file(local_path / "QbC_indexes.json", True, False)

                # If it was not skipped
                if not (local_path / "skip").is_file():
                    # If candidates_count is over max_candidates
                    if exploration_json["systems_auto"][system_auto]["candidates_count"] <= max_candidates:
                        selection_factor = 1
                    else:
                        selection_factor = QbC_stats["candidates_count"] / exploration_json["systems_auto"][system_auto]["candidates_count"]

                    # Get the local max_candidates
                    QbC_stats["selection_factor"] = selection_factor
                    max_candidates_local = int(np.ceil(max_candidates * selection_factor))

                    if selection_factor == 1:
                        QbC_stats["max_candidates_local"] = -1
                    else:
                        QbC_stats["max_candidates_local"] = max_candidates_local

                    candidate_indexes = np.array(QbC_indexes["candidate_indexes"])

                    # Selection of candidates (as linearly as possible, keeping the first and the last ones)
                    if len(candidate_indexes) > max_candidates_local:
                        selected_indexes = candidate_indexes[np.round(np.linspace(0, len(candidate_indexes) - 1, max_candidates_local)).astype(int)]
                    else:
                        selected_indexes = candidate_indexes
                    discarded_indexes = np.setdiff1d(candidate_indexes, selected_indexes)

                    QbC_indexes = {
                        **QbC_indexes,
                        "selected_indexes": selected_indexes.astype(int).tolist() if selected_indexes.size > 0 else [],
                        "discarded_indexes": discarded_indexes.astype(int).tolist() if discarded_indexes.size > 0 else [],
                    }
                    QbC_stats = {
                        **QbC_stats,
                        "selected_count": len(selected_indexes.astype(int).tolist()) if selected_indexes.size > 0 else 0,
                        "discarded_count": len(discarded_indexes.astype(int).tolist()) if discarded_indexes.size > 0 else 0,
                    }

                    total_candidates_selected += len(selected_indexes.astype(int).tolist())

                    # Now we get the starting point (the min of selected, or the last good)
                    # Min of selected
                    if selected_indexes.shape[0] > 0:
                        if exploration_json["systems_auto"][system_auto]["exploration_type"] == "sander_emle":
                            num_atoms, atom_symbols, atom_coords, comments, cell_info, pbc_info, properties_info, max_f_std_info = parse_xyz_trajectory_file(local_path / xyz_qm_filename)
                            model_deviation = np.array(max_f_std_info)
                        elif exploration_json["systems_auto"][system_auto]["exploration_type"] == "lammps" or exploration_json["systems_auto"][system_auto]["exploration_type"] == "i-PI":
                            model_deviation = np.genfromtxt(str(local_path / model_deviation_filename))
                        min_val = 1e30
                        for selected_idx in selected_indexes:
                            if exploration_json["systems_auto"][system_auto]["exploration_type"] == "sander_emle":
                                temp_min = model_deviation[selected_idx]
                            else:
                                temp_min = model_deviation[:, 4][np.where(model_deviation[:, 0] == selected_idx)]
                            if temp_min < min_val:
                                min_val = temp_min
                                min_index = selected_idx
                        QbC_stats["minimum_index"] = int(min_index)
                    # Last of good
                    elif len(QbC_indexes["good_indexes"]) > 0:
                        QbC_stats["minimum_index"] = int(QbC_indexes["good_indexes"][-1])
                    # Nothing
                    else:
                        QbC_stats["minimum_index"] = -1

                else:
                    QbC_indexes = {
                        **QbC_indexes,
                        "selected_indexes": [],
                        "discarded_indexes": [],
                    }
                    QbC_stats = {
                        **QbC_stats,
                        "selection_factor": 0,
                        "max_candidates_local": 0,
                        "selected_count": 0,
                        "discarded_count": 0,
                        "minimum_index": -1,
                    }

                exploration_json["systems_auto"][system_auto]["selected_count"] = exploration_json["systems_auto"][system_auto]["selected_count"] + QbC_stats["selected_count"]
                exploration_json["systems_auto"][system_auto]["discarded_count"] = exploration_json["systems_auto"][system_auto]["discarded_count"] + QbC_stats["discarded_count"]

                write_json_file(QbC_stats, local_path / "QbC_stats.json", False)
                write_json_file(QbC_indexes, local_path / "QbC_indexes.json", False)
                del local_path, model_deviation_filename, QbC_stats, QbC_indexes
            del it_number
        del it_nnp

        del max_candidates, sigma_low, sigma_high, sigma_high_limit, ignore_first_x_ps
    del system_auto_index, system_auto

    arcann_logger.info(f"A total of {total_candidates_selected} structures have been selected for labeling...")
    del total_candidates_selected

    arcann_logger.info(f"-" * 88)
    # Update the booleans in the exploration JSON
    exploration_json["is_deviated"] = True

    # Dump the JSON files (exploration and merged input)
    write_json_file(exploration_json, (control_path / f"exploration_{padded_curr_iter}.json"))
    backup_and_overwrite_json_file(current_input_json, (current_path / "used_input.json"), read_only=True)

    # End
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")

    # Cleaning
    del control_path
    del main_json
    del curr_iter, padded_curr_iter
    del exploration_json
    del training_path, current_path

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "deviate",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
