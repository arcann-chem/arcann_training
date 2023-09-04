"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/31
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

    # Load the default input JSON
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "assets" / "default_config.json"
    )[current_step]
    default_input_json_present = bool(default_input_json)
    logging.debug(f"default_input_json: {default_input_json}")
    logging.debug(f"default_input_json_present: {default_input_json_present}")

    # Load the user input JSON
    if (current_path / user_input_json_filename).is_file():
        user_input_json = load_json_file((current_path / user_input_json_filename))
    else:
        user_input_json = {}
    user_input_json_present = bool(user_input_json)
    logging.debug(f"user_input_json: {user_input_json}")
    logging.debug(f"user_input_json_present: {user_input_json_present}")

    # Make a deepcopy of it to create the merged input JSON
    merged_input_json = copy.deepcopy(user_input_json)

    # Get control path, load the main JSON and the exploration JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    exploration_json = load_json_file(
        (control_path / f"exploration_{padded_curr_iter}.json")
    )

    # Load the previous exploration JSON and training JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_training_json = load_json_file(
            (control_path / f"training_{padded_prev_iter}.json")
        )
        if prev_iter > 0:
            previous_exploration_json = load_json_file(
                (control_path / ("exploration_" + padded_prev_iter + ".json"))
            )
        else:
            previous_exploration_json = {}
    else:
        previous_training_json = {}
        previous_exploration_json = {}

    # Check if we can continue
    if not exploration_json["is_checked"]:
        logging.error(f"Lock found. Execute first: exploration check.")
        logging.error(f"Aborting...")
        return 1
    if (
        exploration_json["exploration_type"] == "i-PI"
        and not exploration_json["is_rechecked"]
    ):
        logging.critical(f"Lock found. Execute first: first: exploration recheck.")
        logging.critical(f"Aborting...")
        return 1

    # Generate/update the merged input JSON
    # Priority: user > previous > default
    merged_input_json = generate_input_exploration_deviation_json(
        user_input_json,
        previous_exploration_json,
        default_input_json,
        merged_input_json,
        main_json,
    )
    logging.debug(f"merged_input_json: {merged_input_json}")
    for system_auto_index, system_auto in enumerate(main_json["systems_auto"]):
        # Set the system params for deviation selection
        (
            max_candidates,
            sigma_low,
            sigma_high,
            sigma_high_limit,
            ignore_first_x_ps,
        ) = get_system_deviation(merged_input_json, system_auto_index)

        # Initialize
        exploration_json["systems_auto"][system_auto] = {
            **exploration_json["systems_auto"][system_auto],
            "max_candidates": max_candidates,
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
            "sigma_high_limit": sigma_high_limit,
            "ignore_first_x_ps": ignore_first_x_ps,
            "mean_deviation_max_f": 0,
            "std_deviation_max_f": 0,
            "total_count": 0,
            "candidates_count": 0,
            "rejected_count": 0,
        }

        skipped_traj_user = 0
        skipped_traj_stats = 0
        start_row_number = 0

        while (
            start_row_number
            * exploration_json["systems_auto"][system_auto]["print_every_x_steps"]
            * exploration_json["systems_auto"][system_auto]["timestep_ps"]
            < exploration_json["systems_auto"][system_auto]["ignore_first_x_ps"]
        ):
            start_row_number = start_row_number + 1
        logging.debug(f"start_row_number: {start_row_number}")

        for it_nnp in range(1, main_json["nnp_count"] + 1):
            for it_number in range(1, exploration_json["traj_count"] + 1):
                logging.debug(f"{system_auto} / {it_nnp} / {it_number}")

                # Get the local path and the name of model_deviation file
                local_path = (
                    Path(".").resolve()
                    / str(system_auto)
                    / str(it_nnp)
                    / str(it_number).zfill(5)
                )
                model_deviation_filename = (
                    f"model_devi_{system_auto}_{it_nnp}_{padded_curr_iter}.out"
                )

                # Create the JSON data for Query-by-Committee
                QbC_stats = load_json_file(local_path / "QbC_stats.json", False, False)
                QbC_indexes = load_json_file(
                    local_path / "QbC_indexes.json", False, False
                )
                QbC_stats = {
                    **QbC_stats,
                    "sigma_low": sigma_low,
                    "sigma_high": sigma_high,
                    "sigma_high_limit": sigma_high_limit,
                }

                # Get the number of exptected steps
                nb_steps_expected = (
                    (
                        exploration_json["systems_auto"][system_auto]["nb_steps"]
                        // exploration_json["systems_auto"][system_auto][
                            "print_every_x_steps"
                        ]
                    )
                    + 1
                    - start_row_number
                )

                # If it was not skipped
                if not (local_path / "skip").is_file():
                    model_deviation = np.genfromtxt(
                        str(local_path / model_deviation_filename)
                    )
                    if exploration_json["exploration_type"] == "lammps":
                        total_row_number = model_deviation.shape[0]
                    elif exploration_json["exploration_type"] == "i-PI":
                        total_row_number = model_deviation.shape[0] + 1

                    if nb_steps_expected > (total_row_number - start_row_number):
                        QbC_stats["total_count"] = nb_steps_expected
                        logging.critical(
                            f"Exploration '{system_auto}' / '{it_nnp}' / '{it_number}'."
                        )
                        logging.critical(
                            f"Mismatch between expected ('{nb_steps_expected}') number of steps."
                        )
                        logging.critical(
                            f"and actual ('{total_row_number - start_row_number}') number of steps in the deviation file."
                        )
                        if (local_path / "force").is_file():
                            logging.warning(
                                "but it has been forced, so it should be ok."
                            )
                    elif nb_steps_expected == (total_row_number - start_row_number):
                        QbC_stats["total_count"] = total_row_number - start_row_number
                    else:
                        logging.error("Unknown error. Please BUG REPORT!")
                        logging.error("Aborting...")
                        return 1

                    end_row_number = get_last_frame_number(
                        model_deviation,
                        sigma_high_limit,
                        exploration_json["systems_auto"][system_auto][
                            "disturbed_start"
                        ],
                    )
                    logging.debug(
                        f"end_row_number: {end_row_number}, start_row_number: {start_row_number}"
                    )

                    # This part is when sigma_high_limit was never crossed
                    if end_row_number == -1:
                        mean_deviation_max_f = np.mean(
                            model_deviation[start_row_number:, 4]
                        )
                        std_deviation_max_f = np.std(
                            model_deviation[start_row_number:, 4]
                        )
                        good = model_deviation[start_row_number:, :][
                            model_deviation[start_row_number:, 4] <= sigma_low
                        ]
                        rejected = model_deviation[start_row_number:, :][
                            model_deviation[start_row_number:, 4] >= sigma_high
                        ]
                        candidates = model_deviation[start_row_number:, :][
                            (model_deviation[start_row_number:, 4] > sigma_low)
                            & (model_deviation[start_row_number:, 4] < sigma_high)
                        ]

                    # This part is when sigma_high_limit was crossed during ignore_first_x_ps (SKIP everything for stats)
                    elif end_row_number <= start_row_number:
                        mean_deviation_max_f = 999.0
                        std_deviation_max_f = 999.0
                        # mean_deviation_max_f = np.mean(model_deviation[start_row_number:, 4])
                        # std_deviation_max_f = np.std(model_deviation[start_row_number:, 4])
                        good = np.array([])
                        rejected = model_deviation[start_row_number:, :]
                        candidates = np.array([])
                        # In this case, it is skipped
                        skipped_traj_stats += 1

                    # This part is when sigma_high_limit was crossed (Gets stats before)
                    else:
                        mean_deviation_max_f = np.mean(
                            model_deviation[start_row_number:end_row_number, 4]
                        )
                        std_deviation_max_f = np.std(
                            model_deviation[start_row_number:end_row_number, 4]
                        )
                        good = model_deviation[start_row_number:end_row_number, :][
                            model_deviation[start_row_number:end_row_number, 4]
                            <= sigma_low
                        ]
                        rejected = model_deviation[start_row_number:end_row_number, :][
                            model_deviation[start_row_number:end_row_number, 4]
                            >= sigma_high
                        ]
                        candidates = model_deviation[
                            start_row_number:end_row_number, :
                        ][
                            (
                                model_deviation[start_row_number:end_row_number, 4]
                                > sigma_low
                            )
                            & (
                                model_deviation[start_row_number:end_row_number, 4]
                                < sigma_high
                            )
                        ]
                        # Add the rest to rejected
                        rejected = np.vstack(
                            (rejected, model_deviation[end_row_number:, :])
                        )

                    # Fill JSON files
                    QbC_indexes = {
                        **QbC_indexes,
                        "good_indexes": good[:, 0].astype(int).tolist()
                        if good.size > 0
                        else [],
                        "rejected_indexes": rejected[:, 0].astype(int).tolist()
                        if rejected.size > 0
                        else [],
                        "candidate_indexes": candidates[:, 0].astype(int).tolist()
                        if candidates.size > 0
                        else [],
                    }

                    QbC_stats = {
                        **QbC_stats,
                        "mean_deviation_max_f": mean_deviation_max_f,
                        "std_deviation_max_f": std_deviation_max_f,
                        "good_count": len(good[:, 0].astype(int).tolist())
                        if good.size > 0
                        else 0,
                        "rejected_count": len(rejected[:, 0].astype(int).tolist())
                        if rejected.size > 0
                        else 0,
                        "candidates_count": len(candidates[:, 0].astype(int).tolist())
                        if candidates.size > 0
                        else 0,
                    }

                    # If the traj is smaller than expected (forced case) add the missing as rejected
                    if (
                        QbC_stats["good_count"]
                        + QbC_stats["rejected_count"]
                        + QbC_stats["candidates_count"]
                    ) < nb_steps_expected:
                        QbC_stats["rejected_count"] = (
                            QbC_stats["rejected_count"]
                            + nb_steps_expected
                            - (
                                QbC_stats["good_count"]
                                + QbC_stats["rejected_count"]
                                + QbC_stats["candidates_count"]
                            )
                        )

                    # Only if we have corect stats, add it
                    if (end_row_number > start_row_number) or (end_row_number == -1):
                        exploration_json["systems_auto"][system_auto][
                            "mean_deviation_max_f"
                        ] = (
                            exploration_json["systems_auto"][system_auto][
                                "mean_deviation_max_f"
                            ]
                            + QbC_stats["mean_deviation_max_f"]
                        )
                        exploration_json["systems_auto"][system_auto][
                            "std_deviation_max_f"
                        ] = (
                            exploration_json["systems_auto"][system_auto][
                                "std_deviation_max_f"
                            ]
                            + QbC_stats["std_deviation_max_f"]
                        )
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
                        "std_deviation_max_f": 999.0,
                        "good_count": 0,
                        "rejected_count": nb_steps_expected,
                        "candidates_count": 0,
                    }

                exploration_json["systems_auto"][system_auto]["total_count"] = (
                    exploration_json["systems_auto"][system_auto]["total_count"]
                    + QbC_stats["total_count"]
                )
                exploration_json["systems_auto"][system_auto]["candidates_count"] = (
                    exploration_json["systems_auto"][system_auto]["candidates_count"]
                    + QbC_stats["candidates_count"]
                )
                exploration_json["systems_auto"][system_auto]["rejected_count"] = (
                    exploration_json["systems_auto"][system_auto]["rejected_count"]
                    + QbC_stats["rejected_count"]
                )

                write_json_file(QbC_stats, local_path / "QbC_stats.json", False)
                write_json_file(
                    QbC_indexes, local_path / "QbC_indexes.json", False, indent=None
                )
                del (
                    local_path,
                    model_deviation_filename,
                    QbC_stats,
                    QbC_indexes,
                    nb_steps_expected,
                )

            del it_number

        # Average for the system (with adjustment, remove the skipped ones)
        exploitable_traj = (
            exploration_json["nnp_count"] * exploration_json["traj_count"]
        ) - (skipped_traj_user + skipped_traj_stats)
        if exploitable_traj == 0:
            logging.critical(
                "All trajectories were skipped (either by you or discarded by ArcaNN)."
            )
            logging.critical(
                "You should either change the 'ignore_first_x_ps' value to a lower value to ensure some structure for this exploration."
            )
            logging.critical(
                "For the next run (or if you redo one), you should reduce 'timestep_ps' and 'print_every_x_steps'."
            )
            logging.critical("Aborting...")
            return 1
        else:
            if (
                exploitable_traj
                / exploration_json["nnp_count"]
                * exploration_json["traj_count"]
                < 0.25
            ):
                logging.warning(
                    "Less than 25% of your exploration trajectories are exploitable. Be careful for the next one."
                )
            exploration_json["systems_auto"][system_auto]["mean_deviation_max_f"] = (
                exploration_json["systems_auto"][system_auto]["mean_deviation_max_f"]
                / exploitable_traj
            )
            exploration_json["systems_auto"][system_auto]["std_deviation_max_f"] = (
                exploration_json["systems_auto"][system_auto]["std_deviation_max_f"]
                / exploitable_traj
            )

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

    for system_auto_index, system_auto in enumerate(exploration_json["systems_auto"]):
        # Set the system params for deviation selection
        (
            max_candidates,
            sigma_low,
            sigma_high,
            sigma_high_limit,
            ignore_first_x_ps,
        ) = get_system_deviation(merged_input_json, system_auto_index)

        # Initialize
        exploration_json["systems_auto"][system_auto] = {
            **exploration_json["systems_auto"][system_auto],
            "kept_count": 0,
            "discarded_count": 0,
        }

        for it_nnp in range(1, main_json["nnp_count"] + 1):
            for it_number in range(1, exploration_json["traj_count"] + 1):
                # Get the local path and the name of model_deviation file
                local_path = (
                    Path(".").resolve()
                    / str(system_auto)
                    / str(it_nnp)
                    / str(it_number).zfill(5)
                )
                model_deviation_filename = (
                    f"model_devi_{system_auto}_{it_nnp}_{padded_curr_iter}.out"
                )

                # Create the JSON data for Query-by-Committee
                QbC_stats = load_json_file(local_path / "QbC_stats.json", True, False)
                QbC_indexes = load_json_file(
                    local_path / "QbC_indexes.json", True, False
                )

                # If it was not skipped
                if not (local_path / "skip").is_file():
                    # If candidates_count is over max_candidates
                    if (
                        exploration_json["systems_auto"][system_auto][
                            "candidates_count"
                        ]
                        <= max_candidates
                    ):
                        selection_factor = 1
                    else:
                        selection_factor = (
                            QbC_stats["candidates_count"]
                            / exploration_json["systems_auto"][system_auto][
                                "candidates_count"
                            ]
                        )

                    # Get the local max_candidates
                    QbC_stats["selection_factor"] = selection_factor
                    max_candidates_local = int(
                        np.floor(max_candidates * selection_factor)
                    )
                    if selection_factor == 1:
                        QbC_stats["max_candidates_local"] = -1
                    else:
                        QbC_stats["max_candidates_local"] = max_candidates_local

                    candidate_indexes = np.array(QbC_indexes["candidate_indexes"])

                    # Selection of candidates (as linearly as possible, keeping the first and the last ones)
                    if len(candidate_indexes) > max_candidates_local:
                        step_size = len(candidate_indexes) / (max_candidates_local - 1)
                        kept_indexes = candidate_indexes[::step_size]
                        kept_indexes = np.concatenate(
                            (
                                [candidate_indexes[0]],
                                kept_indexes,
                                [candidate_indexes[-1]],
                            )
                        )
                        # kept_indexes = candidate_indexes[np.round(np.linspace(0, len(candidate_indexes)-1, max_candidates_local)).astype(int)]
                    else:
                        kept_indexes = candidate_indexes
                    discarded_indexes = np.setdiff1d(candidate_indexes, kept_indexes)

                    QbC_indexes = {
                        **QbC_indexes,
                        "kept_indexes": kept_indexes.astype(int).tolist()
                        if kept_indexes.size > 0
                        else [],
                        "discarded_indexes": discarded_indexes.astype(int).tolist()
                        if discarded_indexes.size > 0
                        else [],
                    }
                    QbC_stats = {
                        **QbC_stats,
                        "kept_count": len(kept_indexes.astype(int).tolist())
                        if kept_indexes.size > 0
                        else 0,
                        "discarded_count": len(discarded_indexes.astype(int).tolist())
                        if discarded_indexes.size > 0
                        else 0,
                    }

                    # Now we get the starting point (the min of kept, or the last good)
                    # Min of kept
                    if kept_indexes.shape[0] > 0:
                        model_deviation = np.genfromtxt(
                            str(local_path / model_deviation_filename)
                        )
                        min_val = 1e30
                        for kept_idx in kept_indexes:
                            temp_min = model_deviation[:, 4][
                                np.where(model_deviation[:, 0] == kept_idx)
                            ]
                            if temp_min < min_val:
                                min_val = temp_min
                                min_index = kept_idx
                        QbC_stats["minimum_index"] = int(min_index)
                    # Last of good
                    elif len(QbC_indexes["good_indexes"]) > 0:
                        QbC_stats["minimum_index"] = int(
                            QbC_indexes["good_indexes"][-1]
                        )
                    # Nothing
                    else:
                        QbC_stats["minimum_index"] = -1

                else:
                    QbC_indexes = {
                        **QbC_indexes,
                        "kept_indexes": [],
                        "discarded_indexes": [],
                    }
                    QbC_stats = {
                        **QbC_stats,
                        "selection_factor": 0,
                        "max_candidates_local": 0,
                        "kept_count": 0,
                        "discarded_count": 0,
                        "minimum_index": -1,
                    }

                exploration_json["systems_auto"][system_auto]["kept_count"] = (
                    exploration_json["systems_auto"][system_auto]["kept_count"]
                    + QbC_stats["kept_count"]
                )
                exploration_json["systems_auto"][system_auto]["discarded_count"] = (
                    exploration_json["systems_auto"][system_auto]["discarded_count"]
                    + QbC_stats["discarded_count"]
                )

                write_json_file(QbC_stats, local_path / "QbC_stats.json", False)
                write_json_file(
                    QbC_indexes, local_path / "QbC_indexes.json", False, indent=None
                )
                del local_path, model_deviation_filename, QbC_stats, QbC_indexes
            del it_number
        del it_nnp

        del max_candidates, sigma_low, sigma_high, sigma_high_limit, ignore_first_x_ps
    del system_auto_index, system_auto

    logging.info(f"-" * 88)
    # Update the booleans in the exploration JSON
    exploration_json["is_deviated"] = True

    # Dump the JSON files (exploration and merged input)
    write_json_file(
        exploration_json, (control_path / f"exploration_{padded_curr_iter}.json")
    )
    backup_and_overwrite_json_file(
        merged_input_json, (current_path / user_input_json_filename)
    )

    # End
    logging.info(f"-" * 88)
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del control_path
    del main_json
    del curr_iter, padded_curr_iter
    del exploration_json
    del training_path, current_path

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "deviation",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
