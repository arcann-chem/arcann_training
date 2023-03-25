from pathlib import Path
import logging
import sys
import copy

# Non-standard library imports
import numpy as np

# deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    load_default_json_file,
    backup_and_overwrite_json_file,
)
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.generate_config import set_subsys_params_deviation
from deepmd_iterative.common.exploration import get_last_frame_number
from deepmd_iterative.common.json_parameters import set_new_input_explordevi_json

def main(
    step_name: str,
    phase_name: str,
    deepmd_iterative_path: Path,
    fake_machine = None,
    input_fn: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()}")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(step_name)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Load the master input JSON file for the program
    default_present = False
    default_json = load_default_json_file(deepmd_iterative_path / "data" / "input_defaults.json")[step_name]
    if bool(default_json):
        default_present = True
    logging.debug(f"default_json: {default_json}")
    logging.debug(f"default_present: {default_present}")

    # Load the user input JSON file
    if (current_path / input_fn).is_file():
        input_json = load_json_file((current_path / input_fn))
        input_present = True
    else:
        input_json = {}
        input_present = False
    logging.debug(f"input_json: {input_json}")
    logging.debug(f"input_present: {input_present}")

    # Make a deepcopy
    new_input_json = copy.deepcopy(input_json)

    # Get control path, config JSON and exploration JSON
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))
    exploration_json = load_json_file(
        (control_path / f"exploration_{padded_curr_iter}.json")
    )

    # Load the previous training JSON and previous exploration JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        prevtraining_json = load_json_file((control_path / f"training_{padded_prev_iter}.json"))
        if prev_iter > 0:
            prevexploration_json = load_json_file((control_path / ("exploration_" + padded_prev_iter + ".json")))
        else:
            prevexploration_json = {}
    else:
        prevtraining_json = {}
        prevexploration_json = {}

    # Checks
    if not exploration_json["is_checked"]:
        logging.error(f"Lock found. Execute first: exploration check.")
        logging.error(f"Aborting...")
        return 1
    if (
        exploration_json["exploration_type"] == "i-PI"
        and not exploration_json["is_rechecked"]
    ):
        logging.critical("Lock found. Execute first: first: exploration recheck.")
        logging.critical("Aborting...")
        return 1

    # Fill the missing values from the input. We don't do exploration because it is subsys dependent and single value and not list
    new_input_json = set_new_input_explordevi_json(input_json,prevexploration_json,default_json,new_input_json,config_json)
    logging.debug(f"new_input_json: {new_input_json}")
    
    for it0_subsys_nr, it_subsys_nr in enumerate(config_json["subsys_nr"]):

        set_new_input_explordevi_json
        # Set the subsys params for deviation selection
        (
            max_candidates,
            sigma_low,
            sigma_high,
            sigma_high_limit,
            ignore_first_x_ps,
        ) = set_subsys_params_deviation(
            input_json,
            new_input_json,
            default_input_json,
            config_json,
            step_name,
            default_present,
            it0_subsys_nr,
        )

        # Initialize
        exploration_json["subsys_nr"][it_subsys_nr] = {
            **exploration_json["subsys_nr"][it_subsys_nr],
            "max_candidates": max_candidates,
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
            "sigma_high_limit": sigma_high_limit,
            "ignore_first_x_ps": ignore_first_x_ps,
            "mean_deviation_max_f": 0,
            "std_deviation_max_f": 0,
            "nb_total": 0,
            "nb_candidates": 0,
            "nb_rejected": 0,
        }

        skipped_traj = 0
        start_row_number = 0

        while (
            start_row_number
            * exploration_json["subsys_nr"][it_subsys_nr]["print_every_x_steps"]
            * exploration_json["subsys_nr"][it_subsys_nr]["timestep_ps"]
            < exploration_json["subsys_nr"][it_subsys_nr]["ignore_first_x_ps"]
        ):
            start_row_number = start_row_number + 1
        logging.debug(f"start_row_number: {start_row_number}")

        for it_nnp in range(1, config_json["nb_nnp"] + 1):
            for it_number in range(1, exploration_json["nb_traj"] + 1):

                logging.debug(f"{it_subsys_nr} / {it_nnp} / {it_number}")

                # Get the local path and the name of model_deviation file
                local_path = (
                    Path(".").resolve()
                    / str(it_subsys_nr)
                    / str(it_nnp)
                    / str(it_number).zfill(5)
                )
                model_deviation_filename = (
                    f"model_devi_{it_subsys_nr}_{it_nnp}_{padded_curr_iter}.out"
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
                        exploration_json["subsys_nr"][it_subsys_nr]["nb_steps"]
                        // exploration_json["subsys_nr"][it_subsys_nr][
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
                        QbC_stats["nb_total"] = nb_steps_expected
                        logging.critical(
                            f"Exploration {it_subsys_nr}/{it_nnp}/{it_number}"
                        )
                        logging.critical(
                            f"Mismatch between expected ({nb_steps_expected}) number of steps"
                        )
                        logging.critical(
                            f"and actual ({total_row_number}) number of steps in the deviation file."
                        )
                        if (local_path / "forced").is_file():
                            logging.warning(
                                "but it has been forced, so it should be ok"
                            )
                    elif nb_steps_expected == (total_row_number - start_row_number):
                        QbC_stats["nb_total"] = total_row_number - start_row_number
                    else:
                        logging.error("Unknown error. Please BUG REPORT")
                        logging.error("Aborting...")
                        return 1

                    end_row_number = get_last_frame_number(
                        model_deviation,
                        sigma_high_limit,
                        exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"],
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
                        mean_deviation_max_f = np.mean(
                            model_deviation[start_row_number:, 4]
                        )
                        std_deviation = np.std(model_deviation[start_row_number:, 4])
                        good = np.array([])
                        rejected = model_deviation[start_row_number:, 0]
                        candidates = np.array([])
                        # In this case, it is skipped
                        skipped_traj += 1

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
                        ### Add the rest to rejected
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
                        "nb_good": len(good[:, 0].astype(int).tolist())
                        if good.size > 0
                        else 0,
                        "nb_rejected": len(rejected[:, 0].astype(int).tolist())
                        if rejected.size > 0
                        else 0,
                        "nb_candidates": len(candidates[:, 0].astype(int).tolist())
                        if candidates.size > 0
                        else 0,
                    }

                    # If the traj is smaller than expected (forced case) add the missing as rejected
                    if (
                        QbC_stats["nb_good"]
                        + QbC_stats["nb_rejected"]
                        + QbC_stats["nb_candidates"]
                    ) < nb_steps_expected:
                        QbC_stats["nb_rejected"] = (
                            QbC_stats["nb_rejected"]
                            + QbC_stats
                            - (
                                QbC_stats["nb_good"]
                                + QbC_stats["nb_rejected"]
                                + QbC_stats["nb_candidates"]
                            )
                        )

                    # Only if we have corect stats, add it
                    if (end_row_number > start_row_number) or (end_row_number == -1):
                        exploration_json["subsys_nr"][it_subsys_nr][
                            "mean_deviation_max_f"
                        ] = (
                            exploration_json["subsys_nr"][it_subsys_nr][
                                "mean_deviation_max_f"
                            ]
                            + QbC_stats["mean_deviation_max_f"]
                        )
                        exploration_json["subsys_nr"][it_subsys_nr][
                            "std_deviation_max_f"
                        ] = (
                            exploration_json["subsys_nr"][it_subsys_nr][
                                "std_deviation_max_f"
                            ]
                            + QbC_stats["std_deviation_max_f"]
                        )
                    del end_row_number

                else:
                    ### If the trajectory was used skiped, count everything as a failure
                    skipped_traj = skipped_traj + 1
                    # Fill JSON files
                    QbC_indexes = {
                        **QbC_indexes,
                        "good_indexes": [],
                        "rejected_indexes": [],
                        "candidate_indexes": [],
                    }

                    QbC_stats = {
                        **QbC_stats,
                        "nb_total": nb_steps_expected,
                        "mean_deviation_max_f": 999.0,
                        "std_deviation_max_f": 999.0,
                        "nb_good": 0,
                        "nb_rejected": nb_steps_expected,
                        "nb_candidates": 0,
                    }

                exploration_json["subsys_nr"][it_subsys_nr]["nb_total"] = (
                    exploration_json["subsys_nr"][it_subsys_nr]["nb_total"]
                    + QbC_stats["nb_total"]
                )
                exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"] = (
                    exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"]
                    + QbC_stats["nb_candidates"]
                )
                exploration_json["subsys_nr"][it_subsys_nr]["nb_rejected"] = (
                    exploration_json["subsys_nr"][it_subsys_nr]["nb_rejected"]
                    + QbC_stats["nb_rejected"]
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

        # Average for the subsys (with adjustment, remove the skipped ones)
        exploration_json["subsys_nr"][it_subsys_nr][
            "mean_deviation_max_f"
        ] = exploration_json["subsys_nr"][it_subsys_nr]["mean_deviation_max_f"] / (
            exploration_json["nb_nnp"]
            + len(range(1, exploration_json["nb_traj"] + 1))
            - skipped_traj
        )
        exploration_json["subsys_nr"][it_subsys_nr][
            "std_deviation_max_f"
        ] = exploration_json["subsys_nr"][it_subsys_nr]["std_deviation_max_f"] / (
            exploration_json["nb_nnp"]
            + len(range(1, exploration_json["nb_traj"] + 1))
            - skipped_traj
        )

        del it_nnp
        del (
            max_candidates,
            sigma_low,
            sigma_high,
            sigma_high_limit,
            ignore_first_x_ps,
            skipped_traj,
            start_row_number,
        )

    del it0_subsys_nr, it_subsys_nr
    for it0_subsys_nr, it_subsys_nr in enumerate(exploration_json["subsys_nr"]):

        # Set the subsys params for deviation selection
        (
            max_candidates,
            sigma_low,
            sigma_high,
            sigma_high_limit,
            ignore_first_x_ps,
        ) = set_subsys_params_deviation(
            input_json,
            new_input_json,
            default_input_json,
            config_json,
            step_name,
            default_present,
            it0_subsys_nr,
        )

        # Initialize
        exploration_json["subsys_nr"][it_subsys_nr] = {
            **exploration_json["subsys_nr"][it_subsys_nr],
            "nb_kept": 0,
            "nb_discarded": 0,
        }

        for it_nnp in range(1, config_json["nb_nnp"] + 1):
            for it_number in range(1, exploration_json["nb_traj"] + 1):

                # Get the local path and the name of model_deviation file
                local_path = (
                    Path(".").resolve()
                    / str(it_subsys_nr)
                    / str(it_nnp)
                    / str(it_number).zfill(5)
                )
                model_deviation_filename = (
                    f"model_devi_{it_subsys_nr}_{it_nnp}_{padded_curr_iter}.out"
                )

                # Create the JSON data for Query-by-Committee
                QbC_stats = load_json_file(local_path / "QbC_stats.json", True, False)
                QbC_indexes = load_json_file(
                    local_path / "QbC_indexes.json", True, False
                )

                # If it was not skipped
                if not (local_path / "skip").is_file():

                    # If nb_candidates is over max_candidates
                    if (
                        exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"]
                        <= max_candidates
                    ):
                        selection_factor = 1
                    else:
                        selection_factor = (
                            QbC_stats["nb_candidates"]
                            / exploration_json["subsys_nr"][it_subsys_nr][
                                "nb_candidates"
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
                        "nb_kept": len(kept_indexes.astype(int).tolist())
                        if kept_indexes.size > 0
                        else 0,
                        "nb_discarded": len(discarded_indexes.astype(int).tolist())
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
                        "nb_kept": 0,
                        "nb_discarded": 0,
                        "minimum_index": -1,
                    }

                exploration_json["subsys_nr"][it_subsys_nr]["nb_kept"] = (
                    exploration_json["subsys_nr"][it_subsys_nr]["nb_kept"]
                    + QbC_stats["nb_kept"]
                )
                exploration_json["subsys_nr"][it_subsys_nr]["nb_discarded"] = (
                    exploration_json["subsys_nr"][it_subsys_nr]["nb_discarded"]
                    + QbC_stats["nb_discarded"]
                )

                write_json_file(QbC_stats, local_path / "QbC_stats.json", False)
                write_json_file(
                    QbC_indexes, local_path / "QbC_indexes.json", False, indent=None
                )
                del local_path, model_deviation_filename, QbC_stats, QbC_indexes
            del it_number
        del it_nnp

        del max_candidates, sigma_low, sigma_high, sigma_high_limit, ignore_first_x_ps
    del it0_subsys_nr, it_subsys_nr

    exploration_json["is_deviated"] = True
    write_json_file(
        exploration_json, (control_path / f"exploration_{padded_curr_iter}.json")
    )
    backup_and_overwrite_json_file(new_input_json, (current_path / input_fn))
    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
    )

    # Cleaning
    del control_path
    del config_json
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
            input_fn=sys.argv[3],
        )
    else:
        pass
