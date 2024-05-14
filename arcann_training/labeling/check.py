"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/14
"""

# Standard library modules
import logging
import sys
from pathlib import Path
import re

# Local imports
from arcann_training.common.json import load_json_file, write_json_file
from arcann_training.common.list import textfile_to_string_list, string_list_to_textfile
from arcann_training.common.filesystem import remove_file
from arcann_training.common.check import validate_step_folder


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

    # Get control path, load the main JSON and the exploration JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    labeling_json = load_json_file((control_path / f"labeling_{padded_curr_iter}.json"))

    # Load the previous labeling JSON
    if curr_iter > 1:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_labeling_json = load_json_file((control_path / f"labeling_{padded_prev_iter}.json"))
        del prev_iter, padded_prev_iter
    else:
        previous_labeling_json = {}

    labeling_program = labeling_json["labeling_program"]
    arcann_logger.debug(f"labeling_program: {labeling_program}")

    # Check if we can continue
    if not labeling_json["is_launched"]:
        arcann_logger.error(f"Lock found. Execute first: labeling launch.")
        arcann_logger.error(f"Aborting...")
        return 1

    # Check the normal termination of the labeling phase
    # Counters
    candidates_expected_count = 0
    candidates_skipped_count = 0
    candidates_step_count = {0: 0, 1: 0}

    for system_auto_index, system_auto in enumerate(labeling_json["systems_auto"]):
        arcann_logger.info(f"Processing system: {system_auto} ({system_auto_index + 1}/{len(labeling_json['systems_auto'])})")
        system_path = current_path / system_auto

        system_candidates_count = labeling_json["systems_auto"][system_auto]["candidates_count"]
        system_disturbed_candidates_count = labeling_json["systems_auto"][system_auto]["disturbed_candidates_count"]

        candidates_expected_count += system_candidates_count + system_disturbed_candidates_count

        # One step, if it is skipped, we don't care if one is not converged or failed
        system_candidates_skipped_count = 0
        system_disturbed_candidates_skipped_count = 0
        system_candidates_skipped = []
        system_disturbed_candidates_skipped = []

        # Because two steps and we care of the status of both
        system_timings_sum = {0: 0, 1: 0}
        system_timings = {0: [], 1: []}
        system_candidates_converged_count = {0: 0, 1: 0}
        system_candidates_not_converged = {0: [], 1: []}
        system_candidates_failed = {0: [], 1: []}

        arcann_logger.debug(f"system_candidates_count + system_disturbed_candidates_count: {system_candidates_count + system_disturbed_candidates_count}")
        if system_candidates_count + system_disturbed_candidates_count == 0:
            # TODO Because no candidates, we "fake-fill" with previous labeling values for the timings
            if curr_iter > 1:
                labeling_json["systems_auto"][system_auto]["timings_s"] = previous_labeling_json["systems_auto"][system_auto]["timings_s"]
            else:
                labeling_json["systems_auto"][system_auto]["timings_s"] = [1800, 3600]
            labeling_json["systems_auto"][system_auto]["candidates_skipped_count"] = 0
            labeling_json["systems_auto"][system_auto]["disturbed_candidates_skipped_count"] = 0
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
                if labeling_program == "cp2k":
                    system_output_cp2k_file = {}
                    system_output_cp2k = {}
                    for step in [0, 1]:
                        system_output_cp2k_file[step] = labeling_step_path / f"{step+1}_labeling_{padded_labeling_step}.out"
                        if system_output_cp2k_file[step].is_file():
                            system_output_cp2k[step] = textfile_to_string_list(system_output_cp2k_file[step])

                            if any("SCF run converged in" in _ for _ in system_output_cp2k[step]):
                                candidates_step_count[step] += 1
                                system_candidates_converged_count[step] += 1
                                if any("T I M I N G" in _ for _ in system_output_cp2k[step]):
                                    system_timings[step] = [_ for _ in system_output_cp2k[step] if "CP2K                                 1  1.0" in _]
                                    system_timings_sum[step] += float(system_timings[step][0].split(" ")[-1])

                            elif any("SCF run NOT converged" in _ for _ in system_output_cp2k[step]):
                                system_candidates_not_converged[step].append(f"{system_output_cp2k_file[step]}")

                            else:
                                system_candidates_failed[step].append(f"{system_output_cp2k_file[step]}")
                        else:
                            system_candidates_failed[step].append(f"{system_output_cp2k_file[step]}")
                    del system_output_cp2k_file, system_output_cp2k

                elif labeling_program == "orca":
                    system_output_orca_file = labeling_step_path / f"1_labeling_{padded_labeling_step}.out"
                    if system_output_orca_file.is_file():
                        system_output_orca = textfile_to_string_list(system_output_orca_file)
                        if any("ORCA TERMINATED NORMALLY" in _ for _ in system_output_orca) and any("SCF CONVERGED" in _ for _ in system_output_orca):
                            candidates_step_count[0] += 1
                            system_candidates_converged_count[0] += 1
                            if any("Sum of individual times" in _ for _ in system_output_orca):
                                arcann_logger.debug(f"ORCA")
                                system_timings[0] = [_ for _ in system_output_orca if "Sum of individual times" in _][-1]
                                pattern = r"(\d+\.\d+) sec"
                                matches = re.search(pattern, system_timings[0])
                                if matches:
                                    system_timings_sum[0] += float(matches.group(1))
                        else:
                            system_candidates_failed[0].append(f"{system_output_orca_file}")
                    else:
                        system_candidates_failed[0].append(f"{system_output_orca_file}")

        if system_disturbed_candidates_count > 0:
            for labeling_step in range(system_candidates_count, system_candidates_count + system_disturbed_candidates_count):
                padded_labeling_step = str(labeling_step).zfill(5)
                labeling_step_path = system_path / padded_labeling_step

                if (labeling_step_path / "skip").is_file():
                    # If the step was skipped
                    candidates_skipped_count += 1
                    system_disturbed_candidates_skipped_count += 1
                    system_disturbed_candidates_skipped.append(f"{labeling_step_path}\n")
                else:
                    if labeling_program == "cp2k":
                        system_output_cp2k_file = {}
                        system_output_cp2k = {}
                        for step in [0, 1]:
                            system_output_cp2k_file[step] = labeling_step_path / f"{step+1}_labeling_{padded_labeling_step}.out"
                            if system_output_cp2k_file[step].is_file():
                                system_output_cp2k[step] = textfile_to_string_list(system_output_cp2k_file[step])
                                if any("SCF run converged in" in _ for _ in system_output_cp2k[step]):
                                    candidates_step_count[step] += 1
                                    system_candidates_converged_count[step] += 1
                                    if any("T I M I N G" in _ for _ in system_output_cp2k[step]):
                                        system_timings[step] = [_ for _ in system_output_cp2k[step] if "CP2K                                 1  1.0" in _]
                                        system_timings_sum[step] += float(system_timings[step][0].split(" ")[-1])
                                elif any("SCF run NOT converged" in _ for _ in system_output_cp2k[step]):
                                    system_candidates_not_converged[step].append(f"{system_output_cp2k_file[step]}")
                                else:
                                    system_candidates_failed[0].append(f"{system_output_orca_file}")
                            else:
                                system_candidates_failed[step].append(f"{system_output_cp2k_file[step]}")
                        del system_output_cp2k_file, system_output_cp2k

                    elif labeling_program == "orca":
                        system_output_orca_file = labeling_step_path / f"1_labeling_{padded_labeling_step}.out"
                        if system_output_orca_file.is_file():
                            system_output_orca = textfile_to_string_list(system_output_orca_file)
                            if any("ORCA TERMINATED NORMALLY" in _ for _ in system_output_orca) and any("SCF CONVERGED" in _ for _ in system_output_orca):
                                candidates_step_count[0] += 1
                                system_candidates_converged_count[0] += 1
                                if any("Sum of individual times" in _ for _ in system_output_orca):
                                    system_timings[0] = [_ for _ in system_output_orca if "Sum of individual times" in _][-1]
                                    pattern = r"(\d+\.\d+) sec"
                                    matches = re.search(pattern, system_timings[0])
                                    if matches:
                                        system_timings_sum[0] += float(matches.group(1))
                            else:
                                system_candidates_failed[0].append(f"{system_output_orca_file}")
                        else:
                            system_candidates_failed[0].append(f"{system_output_orca_file}")

        if (candidates_step_count[0] == 0 or candidates_step_count[1] == 0) and candidates_skipped_count == 0 and labeling_program == "cp2k":
            arcann_logger.critical("ALL jobs have failed/not converged/still running (second step).")
            arcann_logger.critical("Please check manually before relaunching this step")
            arcann_logger.critical('Or create files named "skip" to skip some configurations')
            arcann_logger.critical("Aborting...")
            return 1
        elif (candidates_step_count[0] == 0) and candidates_step_count == 0 and labeling_program == "orca":
            arcann_logger.critical("ALL jobs have failed/not converged/still running (first step).")
            arcann_logger.critical("Please check manually before relaunching this step")
            arcann_logger.critical('Or create files named "skip" to skip some configurations')
            arcann_logger.critical("Aborting...")
            return 1

        timings = {}
        # For the very special case where there are no converged subsystems (e.g., if you skipped all jobs)
        for step, default_timing in enumerate([900.0, 3600.0]):
            if system_candidates_converged_count[step] != 0:
                timings[step] = system_timings_sum[step] / system_candidates_converged_count[step]
            else:
                timings[step] = default_timing
        del step, default_timing
        del system_timings_sum, system_candidates_converged_count, system_timings

        labeling_json["systems_auto"][system_auto]["timings_s"] = [timings[0], timings[1]]
        labeling_json["systems_auto"][system_auto]["candidates_skipped_count"] = system_candidates_skipped_count
        labeling_json["systems_auto"][system_auto]["disturbed_candidates_skipped_count"] = system_disturbed_candidates_skipped_count
        del timings

        for step in [0, 1]:
            if labeling_program == "orca" and step == 1:
                continue
            not_converged_file = system_path / f"{system_auto}_step{step+1}_not_converged.txt"
            remove_file(not_converged_file)
            if len(system_candidates_not_converged[step]) > 0:
                string_list_to_textfile(not_converged_file, system_candidates_not_converged[step])
                arcann_logger.warning(f"{system_auto} | step {step+1}: {len(system_candidates_not_converged[step])} jobs did not converge. List in '{not_converged_file}'")
            del not_converged_file

            failed_file = system_path / f"{system_auto}_step{step+1}_failed.txt"
            remove_file(failed_file)
            if len(system_candidates_failed[step]) > 0:
                string_list_to_textfile(failed_file, system_candidates_failed[step])
                arcann_logger.warning(f"{system_auto} | step {step+1}: {len(system_candidates_failed[step])} jobs did not converge. List in '{failed_file}'.")
            del failed_file

        del step
        del system_candidates_not_converged, system_candidates_failed

        remove_file(system_path / f"{system_auto}_skipped.txt")
        if system_candidates_skipped_count > 0 or system_disturbed_candidates_skipped_count > 0:
            skipped_file = system_path / f"{system_auto}_skipped.txt"
            system_candidates_skipped_total = system_candidates_skipped + system_disturbed_candidates_skipped
            system_candidates_skipped_total_count = system_candidates_skipped_count + system_disturbed_candidates_skipped_count
            string_list_to_textfile(skipped_file, system_candidates_skipped_total)
            arcann_logger.info(f"{system_auto}: {system_candidates_skipped_total_count} jobs skipped ({system_candidates_skipped_count}|{system_disturbed_candidates_skipped_count}). List in '{skipped_file}'.")

            del system_candidates_skipped_total, system_candidates_skipped_total_count
            del skipped_file
            del (
                system_candidates_skipped_count,
                system_disturbed_candidates_skipped_count,
            )
            del system_candidates_skipped, system_disturbed_candidates_skipped

        del system_candidates_count, system_disturbed_candidates_count
        del system_path, labeling_step, labeling_step_path, padded_labeling_step

        arcann_logger.info(f"Processed system: {system_auto} ({system_auto_index + 1}/{len(labeling_json['systems_auto'])})")
    del system_auto, system_auto_index

    # Check first step, write warning if not converged/failed and CP2K, abort if not converged/failed and ORCA
    if (candidates_expected_count != (candidates_step_count[0] + candidates_skipped_count)) and labeling_program == "cp2k":
        arcann_logger.warning("Some jobs have failed/not converged/still running (first step). Check manually if needed.")
    elif (candidates_expected_count != (candidates_step_count[0] + candidates_skipped_count)) and labeling_program == "orca":
        arcann_logger.critical("Some jobs have failed/not converged/still running (first step). Check manually.")
        arcann_logger.critical("Or create files named 'skip' to skip some configurations.")
        arcann_logger.critical("Aborting")
        return 1
    # Check second step, abort if not converged/failed and CP2K
    if (candidates_expected_count != (candidates_step_count[1] + candidates_skipped_count)) and labeling_program == "cp2k":
        arcann_logger.critical("Some jobs have failed/not converged/still running (second step). Check manually.")
        arcann_logger.critical("Or create files named 'skip' to skip some configurations.")
        arcann_logger.critical("Aborting")
        return 1

    arcann_logger.info(f"-" * 88)
    # Update the booleans in the exploration JSON
    arcann_logger.debug(f"candidates_expected_count: {candidates_expected_count}")
    arcann_logger.debug(f"candidates_skipped_count: {candidates_skipped_count}")
    arcann_logger.debug(f"candidates_step_count: {candidates_step_count}")
    if candidates_expected_count == (candidates_step_count[1] + candidates_skipped_count) and labeling_program == "cp2k":
        labeling_json["is_checked"] = True
    elif candidates_expected_count == (candidates_step_count[0] + candidates_skipped_count) and labeling_program == "orca":
        labeling_json["is_checked"] = True
    del candidates_expected_count, candidates_skipped_count, candidates_step_count

    # Dump the JSON files (exploration JSONN)
    write_json_file(labeling_json, (control_path / f"labeling_{padded_curr_iter}.json"))

    # End
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")

    # Cleaning
    del current_path, control_path, training_path
    del (user_input_json_filename,)
    del main_json, labeling_json
    del curr_iter, padded_curr_iter

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
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
