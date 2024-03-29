"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/29
"""

# Standard library modules
import logging
import sys
from pathlib import Path

# Non-standard library imports
import numpy as np

# Local imports
from deepmd_iterative.common.json import load_json_file, write_json_file, get_key_in_dict, load_default_json_file, backup_and_overwrite_json_file
from deepmd_iterative.common.list import textfile_to_string_list
from deepmd_iterative.common.check import validate_step_folder, check_vmd, check_dcd_is_valid, check_nc_is_valid


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

    # Load the default input JSON
    default_input_json = load_default_json_file(deepmd_iterative_path / "assets" / "default_config.json")[current_step]
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

    # If the used input JSON is present, load it
    if (current_path / "used_input.json").is_file():
        current_input_json = load_json_file((current_path / "used_input.json"))
    else:
        logging.warning(f"No used_input.json found. Starting with empty one.")
        logging.warning(f"You should avoid this by not deleting the used_input.json file.")
        current_input_json = {}
    logging.debug(f"current_input_json: {current_input_json}")

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
    if not exploration_json["is_launched"]:
        logging.error(f"Lock found. Execute first: exploration launch.")
        logging.error(f"Aborting...")
        return 1

    # Check if the vmd package is installed
    vmd_bin = check_vmd(get_key_in_dict("vmd_path", user_input_json, previous_exploration_json, default_input_json))
    current_input_json["vmd_path"] = vmd_bin

    exploration_json["vmd_path"] = vmd_bin

    # Check the normal termination of the exploration phase
    # Counters
    completed_count = 0
    skipped_count = 0
    forced_count = 0

    for system_auto_index, system_auto in enumerate(main_json["systems_auto"]):
        # Counters
        average_per_step = 0
        system_count = 0
        timings = []
        # Update the exploration config JSON
        exploration_json["systems_auto"][system_auto] = {
            **exploration_json["systems_auto"][system_auto],
            "completed_count": 0,
            "forced_count": 0,
            "skipped_count": 0,
        }

        for it_nnp in range(1, main_json["nnp_count"] + 1):
            for it_number in range(1, exploration_json["systems_auto"][system_auto]["traj_count"] + 1):
                local_path = Path(".").resolve() / str(system_auto) / str(it_nnp) / (str(it_number).zfill(5))
                logging.debug(f"Checking local_path: {local_path}")

                # If there is a skip, we skip
                if (local_path / "skip").is_file():
                    skipped_count += 1
                    exploration_json["systems_auto"][system_auto]["skipped_count"] += 1
                    logging.warning(f"'{local_path}' skipped.")
                    continue

                # LAMMPS
                if exploration_json["systems_auto"][system_auto]["exploration_type"] == "lammps":
                    traj_file = local_path / f"{system_auto}_{it_nnp}_{padded_curr_iter}.dcd"
                    lammps_output_file = local_path / f"{system_auto}_{it_nnp}_{padded_curr_iter}.log"
                    model_deviation_filename = local_path / f"model_devi_{system_auto}_{it_nnp}_{padded_curr_iter}.out"

                    # Log if files are missing.
                    if not all([traj_file.is_file(), lammps_output_file.is_file(), model_deviation_filename.is_file()]):
                        logging.critical(f"'{local_path}': missing files. Check manually.")
                        del lammps_output_file, traj_file, model_deviation_filename
                        continue

                    # Check if DCD is unreadable
                    if not check_dcd_is_valid(traj_file, vmd_bin):
                        (local_path / "skip").touch(exist_ok=True)
                        skipped_count += 1
                        exploration_json["systems_auto"][system_auto]["skipped_count"] += 1
                        logging.warning(f"'{traj_file}' present but invalid.")
                        logging.warning(f"'{local_path}' auto-skipped.")
                        del lammps_output_file, traj_file, model_deviation_filename
                        continue

                    # Check if output is valid (or forced)
                    lammps_output = textfile_to_string_list(lammps_output_file)
                    if (local_path / "force").is_file():
                        forced_count += 1
                        exploration_json["systems_auto"][system_auto]["forced_count"] += 1
                        logging.warning(f"'{local_path}' forced.")
                    elif any("Total wall time:" in f for f in lammps_output):
                        system_count += 1
                        completed_count += 1
                        exploration_json["systems_auto"][system_auto]["completed_count"] += 1
                        timings_str = [zzz for zzz in lammps_output if "Loop time of" in zzz]
                        timings.append(float(timings_str[0].split(" ")[3]))
                        del timings_str
                    else:
                        logging.critical(f"'{lammps_output_file}' failed. Check manually.")
                    del lammps_output, lammps_output_file, traj_file, model_deviation_filename

                elif exploration_json["systems_auto"][system_auto]["exploration_type"] == "sander_ml":
                    traj_file = local_path / f"{system_auto}_{it_nnp}_{padded_curr_iter}.nc"
                    sander_emle_output_file = local_path / f"{system_auto}_{it_nnp}_{padded_curr_iter}.out"
                    model_deviation_filename = local_path / f"model_devi_{system_auto}_{it_nnp}_{padded_curr_iter}.out"

                    # Log if files are missing.
                    if not all([traj_file.is_file(), lammps_output_file.is_file(), model_deviation_filename.is_file()]):
                        logging.critical(f"'{local_path}': missing files. Check manually.")
                        del lammps_output_file, traj_file, model_deviation_filename
                        continue

                    # Check if NC is unreadable
                    if not check_nc_is_valid(traj_file, vmd_bin):
                        (local_path / "skip").touch(exist_ok=True)
                        skipped_count += 1
                        exploration_json["systems_auto"][system_auto]["skipped_count"] += 1
                        logging.warning(f"'{traj_file}' present but invalid.")
                        logging.warning(f"'{local_path}' auto-skipped.")
                        del sander_emle_output_file, traj_file, model_deviation_filename
                        continue

                    # Check if output is valid (or forced)
                    sander_emle_ouput = textfile_to_string_list(sander_emle_output_file)
                    if (local_path / "force").is_file():
                        forced_count += 1
                        exploration_json["systems_auto"][system_auto]["forced_count"] += 1
                        logging.warning(f"'{local_path}' forced.")
                    # TODO : Check if the output is valid
                    elif any("Total wall time:" in f for f in sander_emle_ouput):
                        system_count += 1
                        completed_count += 1
                        exploration_json["systems_auto"][system_auto]["completed_count"] += 1
                        timings_str = [zzz for zzz in sander_emle_ouput if "Loop time of" in zzz]
                        timings.append(float(timings_str[0].split(" ")[3]))
                        del timings_str
                    else:
                        logging.critical(f"'{sander_emle_output_file}' failed. Check manually.")
                    del sander_emle_ouput, sander_emle_output_file, traj_file, model_deviation_filename

                # i-PI
                elif exploration_json["systems_auto"][system_auto]["exploration_type"] == "i-PI":
                    ipi_output_file = local_path / f"{system_auto}_{it_nnp}_{padded_curr_iter}.i-PI.server.log"

                    # Log if files are missing.
                    if not ipi_output_file.is_file():
                        logging.critical(f"'{local_path}': missing files. Check manually.")
                        del ipi_output_file
                        continue

                    # Check if output is valid (or forced)
                    ipi_output = textfile_to_string_list(ipi_output_file)
                    if (local_path / "force").is_file():
                        forced_count += 1
                        exploration_json["systems_auto"][system_auto]["forced_count"] += 1
                        logging.warning(f"'{local_path}' forced.")
                    elif any("SIMULATION: Exiting cleanly" in f for f in ipi_output):
                        system_count += 1
                        completed_count += 1
                        exploration_json["systems_auto"][system_auto]["completed_count"] += 1
                        ipi_time = [zzz for zzz in ipi_output if "Average timings at MD step" in zzz]
                        ipi_time2 = [zzz[zzz.index("step:") + len("step:") : zzz.index("\n")] for zzz in ipi_time]
                        timings.append(np.average(np.asarray(ipi_time2, dtype="float32")))
                        del ipi_time, ipi_time2
                    else:
                        logging.critical(f"'{ipi_output_file}' failed. Check manually.")
                        del ipi_output
                    del ipi_output_file
                else:
                    logging.error(f"'{exploration_json['exploration_type']}' unknown. Check manually.")
                    return 1
                del local_path

        # timings = timings_sum / system_count

        if exploration_json["systems_auto"][system_auto]["exploration_type"] == "lammps":
            average_per_step = np.array(timings) / exploration_json["systems_auto"][system_auto]["nb_steps"]
        elif exploration_json["systems_auto"][system_auto]["exploration_type"] == "i-PI":
            average_per_step = np.array(timings)
        else:
            average_per_step = np.NaN

        if np.isnan(np.average(average_per_step)) and prev_iter > 0:
            logging.warning(f"Using previous mean_s_per_step for '{system_auto}'")
            for timings_key in ["mean_s_per_step", "median_s_per_step", "stdeviation_s_per_step"]:
                exploration_json["systems_auto"][system_auto][timings_key] = previous_exploration_json["systems_auto"][system_auto][timings_key]
            del timings_key
        elif np.isnan(np.average(average_per_step)) and prev_iter == 0:
            if "job_walltime_h" in current_input_json:
                logging.warning(f"Using job_walltime_h to calculate mean_s_per_step for '{system_auto}'.")
                average_per_step = (current_input_json["job_walltime_h"][system_auto_index] * 3600) / exploration_json["systems_auto"][system_auto]["nb_steps"]
            else:
                logging.error(f"Missing input job_walltime_h for '{system_auto}', set to the default to calculcate mean_s_per_step.")
                average_per_step = 3600. / exploration_json["systems_auto"][system_auto]["nb_steps"]

            exploration_json["systems_auto"][system_auto][timings_key] = average_per_step
            for timings_key in ["median_s_per_step", "stdeviation_s_per_step"]:
                exploration_json["systems_auto"][system_auto][timings_key] = 0
            del timings_key
        else:
            exploration_json["systems_auto"][system_auto]["mean_s_per_step"] = np.average(average_per_step)
            exploration_json["systems_auto"][system_auto]["median_s_per_step"] = np.median(average_per_step)
            exploration_json["systems_auto"][system_auto]["stdeviation_s_per_step"] = np.std(average_per_step)

        del timings, average_per_step, system_count

    del system_auto, it_nnp, it_number

    logging.info(f"-" * 88)
    # Update the booleans in the exploration JSON
    if (completed_count + skipped_count + forced_count) == (exploration_json["nnp_count"] * sum([exploration_json["systems_auto"][_]["traj_count"] for _ in exploration_json["systems_auto"]])):
        exploration_json["is_checked"] = True

    # Dump the JSON files (exploration JSON)
    write_json_file(exploration_json, (control_path / f"exploration_{padded_curr_iter}.json"), read_only=True)
    backup_and_overwrite_json_file(current_input_json, (current_path / "used_input.json"), read_only=True)

    # End
    logging.info(f"-" * 88)
    if (completed_count + skipped_count + forced_count) != (exploration_json["nnp_count"] * sum([exploration_json["systems_auto"][_]["traj_count"] for _ in exploration_json["systems_auto"]])):
        logging.error(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a failure!")
        logging.error(f"Please check manually before relaunching this step.")
        logging.error(f"Or create files named 'skip' or 'force' to skip or force.")
        logging.error(f"Aborting...")
        return 1

    del completed_count

    if (skipped_count + forced_count) != 0:
        logging.warning(f"'{skipped_count}' systems were skipped.")
        logging.warning(f"'{forced_count}' systems were forced.")
    logging.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")
    del skipped_count, forced_count

    # Cleaning
    del control_path
    del main_json
    del curr_iter, padded_curr_iter
    del exploration_json
    del training_path, current_path

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "check",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
