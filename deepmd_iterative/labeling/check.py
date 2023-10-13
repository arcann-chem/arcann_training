"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/10/13
"""
# Standard library modules
import logging
import sys
from pathlib import Path

# Local imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.list import (
    textfile_to_string_list,
    string_list_to_textfile,
)
from deepmd_iterative.common.filesystem import (
    remove_file,
    remove_files_matching_glob,
)
from deepmd_iterative.common.check import validate_step_folder


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

    # Get control path, load the main JSON and the exploration JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    labeling_json = load_json_file((control_path / f"labeling_{padded_curr_iter}.json"))

    # Load the previous labeling JSON
    if curr_iter > 1:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_labeling_json = load_json_file(
            (control_path / f"labeling_{padded_prev_iter}.json")
        )
        del prev_iter, padded_prev_iter
    else:
        previous_labeling_json = {}

    # Check if we can continue
    if not labeling_json["is_launched"]:
        logging.error(f"Lock found. Execute first: labeling launch.")
        logging.error(f"Aborting...")
        return 1

    # Check the normal termination of the labeling phase
    # Counters
    candidates_expected_count = 0
    candidates_skipped_count = 0
    candidates_step_count = {0: 0, 1: 0}

    for system_auto_index, system_auto in enumerate(labeling_json["systems_auto"]):
        logging.info(
            f"Processing system: {system_auto} ({system_auto_index + 1}/{len(labeling_json['systems_auto'])})"
        )
        system_path = current_path / system_auto

        system_candidates_count = labeling_json["systems_auto"][system_auto][
            "candidates_count"
        ]
        system_disturbed_candidates_count = labeling_json["systems_auto"][system_auto][
            "disturbed_candidates_count"
        ]

        candidates_expected_count += (
            system_candidates_count + system_disturbed_candidates_count
        )

        # One step, if it is skipped, we don't care if one is not converged or failed
        system_candidates_skipped_count = 0
        system_disturbed_candidates_skipped_count = 0
        system_candidates_skipped = []
        system_disturbed_candidates_skipped = []

        # Because two steps and we care of the status of both
        system_tinings_sum = {0: 0, 1: 0}
        system_timings = {0: [], 1: []}
        system_candidates_converged_count = {0: 0, 1: 0}
        system_candidates_not_converged = {0: [], 1: []}
        system_candidates_failed = {0: [], 1: []}

        logging.debug(
            f"system_candidates_count + system_disturbed_candidates_count: {system_candidates_count + system_disturbed_candidates_count}"
        )
        if system_candidates_count + system_disturbed_candidates_count == 0:
            # TODO Because no candidates, we "fake-fill" with previous labeling values for the timings
            labeling_json["systems_auto"][system_auto][
                "timings_s"
            ] = previous_labeling_json["systems_auto"][system_auto]["timings_s"]
            labeling_json["systems_auto"][system_auto]["candidates_skipped_count"] = 0
            labeling_json["systems_auto"][system_auto][
                "disturbed_candidates_skipped_count"
            ] = 0
            continue

        # TODO Use a function to parse CP2K output
        for labeling_step in range(system_candidates_count):
            padded_labeling_step = str(labeling_step).zfill(5)
            labeling_step_path = system_path / padded_labeling_step

            if (labeling_step_path / "skip").is_file():
                # If the step was skipped
                candidates_skipped_count += 1
                system_candidates_skipped_count += 1
                system_candidates_skipped.append(f"{labeling_step_path}\n")
            else:
                system_output_cp2k_file = {}
                system_output_cp2k = {}
                for step in [0, 1]:
                    system_output_cp2k_file[step] = (
                        labeling_step_path
                        / f"{step+1}_labeling_{padded_labeling_step}.out"
                    )
                    if system_output_cp2k_file[step].is_file():
                        system_output_cp2k[step] = textfile_to_string_list(
                            system_output_cp2k_file[step]
                        )

                        if any(
                            "SCF run converged in" in _
                            for _ in system_output_cp2k[step]
                        ):
                            candidates_step_count[step] += 1
                            system_candidates_converged_count[step] += 1
                            if any(
                                "T I M I N G" in _ for _ in system_output_cp2k[step]
                            ):
                                system_timings[step] = [
                                    _
                                    for _ in system_output_cp2k[step]
                                    if "CP2K                                 1  1.0"
                                    in _
                                ]
                                system_tinings_sum[step] += float(
                                    system_timings[step][0].split(" ")[-1]
                                )

                        elif any(
                            "SCF run converged in" in _
                            for _ in system_output_cp2k[step]
                        ):
                            system_candidates_not_converged[step].append(
                                f"{system_output_cp2k_file[step]}"
                            )

                        else:
                            system_candidates_failed[step].append(
                                f"{system_output_cp2k_file[step]}"
                            )
                    else:
                        system_candidates_failed[step].append(
                            f"{system_output_cp2k_file[step]}"
                        )

                del system_output_cp2k_file, system_output_cp2k

        if system_disturbed_candidates_count > 0:
            for labeling_step in range(
                system_candidates_count,
                system_candidates_count + system_disturbed_candidates_count,
            ):
                padded_labeling_step = str(labeling_step).zfill(5)
                labeling_step_path = system_path / padded_labeling_step

                if (labeling_step_path / "skip").is_file():
                    # If the step was skipped
                    candidates_skipped_count += 1
                    system_disturbed_candidates_skipped_count += 1
                    system_disturbed_candidates_skipped.append(
                        f"{labeling_step_path}\n"
                    )
                else:
                    system_output_cp2k_file = {}
                    system_output_cp2k = {}
                    for step in [0, 1]:
                        system_output_cp2k_file[step] = (
                            labeling_step_path
                            / f"{step+1}_labeling_{padded_labeling_step}.out"
                        )

                        if system_output_cp2k_file[step].is_file():
                            system_output_cp2k[step] = textfile_to_string_list(
                                system_output_cp2k_file[step]
                            )

                            if any(
                                "SCF run converged in" in _
                                for _ in system_output_cp2k[step]
                            ):
                                candidates_step_count[step] += 1
                                system_candidates_converged_count[step] += 1
                                if any(
                                    "T I M I N G" in _ for _ in system_output_cp2k[step]
                                ):
                                    system_timings[step] = [
                                        _
                                        for _ in system_output_cp2k[step]
                                        if "CP2K                                 1  1.0"
                                        in _
                                    ]
                                    system_tinings_sum[step] += float(
                                        system_timings[step][0].split(" ")[-1]
                                    )
                            elif any(
                                "SCF run converged in" in _
                                for _ in system_output_cp2k[step]
                            ):
                                system_candidates_not_converged[step].append(
                                    f"{system_output_cp2k_file[step]}"
                                )

                            else:
                                system_candidates_failed[step].append(
                                    f"{system_output_cp2k_file[step]}"
                                )
                        else:
                            system_candidates_failed[step].append(
                                f"{system_output_cp2k_file[step]}"
                            )

                    del system_output_cp2k_file, system_output_cp2k

        if (
            candidates_step_count[0] == 0 or candidates_step_count[1] == 0
        ) and candidates_skipped_count == 0:
            logging.critical(
                "ALL jobs have failed/not converged/still running (second step)."
            )
            logging.critical("Please check manually before relaunching this step")
            logging.critical('Or create files named "skip" to skip some configurations')
            logging.critical("Aborting...")
            return 1

        timings = {}
        # For the very special case where there are no converged subsystems (e.g., if you skipped all jobs)
        for step, default_timing in enumerate([900.0, 3600.0]):
            if system_candidates_converged_count[step] != 0:
                timings[step] = (
                    system_tinings_sum[step] / system_candidates_converged_count[step]
                )
            else:
                timings[step] = default_timing
        del step, default_timing
        del system_tinings_sum, system_candidates_converged_count, system_timings

        labeling_json["systems_auto"][system_auto]["timings_s"] = [
            timings[0],
            timings[1],
        ]
        labeling_json["systems_auto"][system_auto][
            "candidates_skipped_count"
        ] = system_candidates_skipped_count
        labeling_json["systems_auto"][system_auto][
            "disturbed_candidates_skipped_count"
        ] = system_disturbed_candidates_skipped_count
        del timings

        for step in [0, 1]:
            not_converged_file = (
                system_path / f"{system_auto}_step{step+1}_not_converged.txt"
            )
            remove_file(not_converged_file)
            if len(system_candidates_not_converged[step]) > 0:
                string_list_to_textfile(
                    not_converged_file, system_candidates_not_converged[step]
                )
                logging.warning(
                    f"{system_auto} | step {step+1}: {len(system_candidates_not_converged[step])} jobs did not converge. List in '{not_converged_file}'"
                )
            del not_converged_file

            failed_file = system_path / f"{system_auto}_step{step+1}_failed.txt"
            remove_file(failed_file)
            if len(system_candidates_failed[step]) > 0:
                string_list_to_textfile(failed_file, system_candidates_failed[step])
                logging.warning(
                    f"{system_auto} | step {step+1}: {len(system_candidates_failed[step])} jobs did not converge. List in '{failed_file}'."
                )
            del failed_file

        del step
        del system_candidates_not_converged, system_candidates_failed

        remove_file(system_path / f"{system_auto}_skipped.txt")
        if (
            system_candidates_skipped_count > 0
            or system_disturbed_candidates_skipped_count > 0
        ):
            skipped_file = system_path / f"{system_auto}_skipped.txt"
            system_candidates_skipped_total = (
                system_candidates_skipped + system_disturbed_candidates_skipped
            )
            system_candidates_skipped_total_count = (
                system_candidates_skipped_count
                + system_disturbed_candidates_skipped_count
            )
            string_list_to_textfile(skipped_file, system_candidates_skipped_total)
            logging.info(
                f"{system_auto}: {system_candidates_skipped_total_count} jobs skipped ({system_candidates_skipped_count}|{system_disturbed_candidates_skipped_count}). List in '{skipped_file}'."
            )

            del system_candidates_skipped_total, system_candidates_skipped_total_count
            del skipped_file
            del (
                system_candidates_skipped_count,
                system_disturbed_candidates_skipped_count,
            )
            del system_candidates_skipped, system_disturbed_candidates_skipped

        del (
            system_candidates_count,
            system_disturbed_candidates_count,
        )
        del system_path, labeling_step, labeling_step_path, padded_labeling_step

        logging.info(
            f"Processed system: {system_auto} ({system_auto_index + 1}/{len(labeling_json['systems_auto'])})"
        )
    del system_auto, system_auto_index

    if candidates_expected_count != (
        candidates_step_count[0] + candidates_skipped_count
    ):
        logging.warning(
            "Some jobs have failed/not converged/still running (first step). Check manually if needed."
        )

    if candidates_expected_count != (
        candidates_step_count[1] + candidates_skipped_count
    ):
        logging.critical(
            "Some jobs have failed/not converged/still running (second step). Check manually."
        )
        logging.critical("Or create files named 'skip' to skip some configurations.")
        logging.critical("Aborting")
        return 1

    logging.info(f"-" * 88)
    # Update the booleans in the exploration JSON
    if candidates_expected_count == (
        candidates_step_count[1] + candidates_skipped_count
    ):
        labeling_json["is_checked"] = True
    del candidates_expected_count, candidates_skipped_count, candidates_step_count

    # for system_auto_index, system_auto in enumerate(labeling_json["systems_auto"]):
    #     system_path = current_path / system_auto
    #     logging.info("Deleting SLURM out/error files...")
    #     remove_files_matching_glob(system_path, "**/CP2K.*")
    #     remove_files_matching_glob(system_path, "CP2K.*")
    #     logging.info("Cleaning done!")
    # del system_auto, system_auto_index, system_path

    # Dump the JSON files (exploration JSONN)
    write_json_file(labeling_json, (control_path / f"labeling_{padded_curr_iter}.json"))

    # End
    logging.info(f"-" * 88)
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del (user_input_json_filename,)
    del main_json, labeling_json
    del curr_iter, padded_curr_iter

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "labeling",
            "check",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
