"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/04
"""
# Standard library modules
import logging
import sys
from pathlib import Path

# Non-standard library imports
import numpy as np

# Local imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.list import textfile_to_string_list
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
    exploration_json = load_json_file(
        (control_path / f"exploration_{padded_curr_iter}.json")
    )

    # Check if we can continue
    if not exploration_json["is_launched"]:
        logging.error(f"Lock found. Execute first: exploration launch.")
        logging.error(f"Aborting...")
        return 1

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
            for it_number in range(
                1, exploration_json["systems_auto"][system_auto]["traj_count"] + 1
            ):
                local_path = (
                    Path(".").resolve()
                    / str(system_auto)
                    / str(it_nnp)
                    / (str(it_number).zfill(5))
                )

                # LAMMPS
                if (
                    exploration_json["systems_auto"][system_auto]["exploration_type"]
                    == "lammps"
                ):
                    lammps_output_file = (
                        local_path / f"{system_auto}_{it_nnp}_{padded_curr_iter}.log"
                    )
                    if lammps_output_file.is_file():
                        lammps_output = textfile_to_string_list(lammps_output_file)
                        if (local_path / "skip").is_file():
                            skipped_count += 1
                            exploration_json["systems_auto"][system_auto][
                                "skipped_count"
                            ] += 1
                            logging.warning(f"'{lammps_output_file}' skipped.")
                        elif (local_path / "force").is_file():
                            forced_count += 1
                            exploration_json["systems_auto"][system_auto][
                                "forced_count"
                            ] += 1
                            logging.warning(f"'{lammps_output_file}' forced.")
                        elif any("Total wall time:" in f for f in lammps_output):
                            system_count += 1
                            completed_count += 1
                            exploration_json["systems_auto"][system_auto][
                                "completed_count"
                            ] += 1
                            timings_str = [
                                zzz for zzz in lammps_output if "Loop time of" in zzz
                            ]
                            timings.append(float(timings_str[0].split(" ")[3]))
                            del timings_str
                        else:
                            logging.critical(
                                f"'{lammps_output_file}' failed. Check manually."
                            )
                        del lammps_output
                    elif (local_path / "skip").is_file():
                        skipped_count += 1
                        exploration_json["systems_auto"][system_auto][
                            "skipped_count"
                        ] += 1
                        logging.warning(f"'{lammps_output_file}' skipped.")
                    else:
                        logging.critical(
                            f"'{lammps_output_file}' failed. Check manually."
                        )
                    del lammps_output_file

                # i-PI
                elif (
                    exploration_json["systems_auto"][system_auto]["exploration_type"]
                    == "i-PI"
                ):
                    ipi_output_file = (
                        local_path
                        / f"{system_auto}_{it_nnp}_{padded_curr_iter}.i-PI.server.log"
                    )
                    if ipi_output_file.is_file():
                        ipi_output = textfile_to_string_list(ipi_output_file)
                        if (local_path / "skip").is_file():
                            skipped_count += 1
                            exploration_json["systems_auto"][system_auto][
                                "skipped_count"
                            ] += 1
                            logging.warning(f"'{ipi_output_file}' skipped.")
                        elif (local_path / "force").is_file():
                            forced_count += 1
                            exploration_json["systems_auto"][system_auto][
                                "forced_count"
                            ] += 1
                            logging.warning(f"'{ipi_output_file}' forced.")
                        elif any("SIMULATION: Exiting cleanly" in f for f in ipi_output):
                            system_count += 1
                            completed_count += 1
                            exploration_json["systems_auto"][system_auto][
                                "completed_count"
                            ] += 1
                            ipi_time = [
                                zzz
                                for zzz in ipi_output
                                if "Average timings at MD step" in zzz
                            ]
                            ipi_time2 = [
                                zzz[zzz.index("step:") + len("step:") : zzz.index("\n")]
                                for zzz in ipi_time
                            ]
                            timings.append(
                                np.average(np.asarray(ipi_time2, dtype="float32"))
                            )
                            del ipi_time, ipi_time2
                        else:
                            logging.critical(
                                f"'{ipi_output_file}' failed. Check manually."
                            )
                        del ipi_output
                    elif (local_path / "skip").is_file():
                        skipped_count += 1
                        exploration_json["systems_auto"][system_auto][
                            "skipped_count"
                        ] += 1
                        logging.warning(f"'{ipi_output_file}' skipped.")
                    else:
                        logging.critical(f"'{ipi_output_file}' failed. Check manually.")
                    del ipi_output_file

                else:
                    logging.error(
                        f"'{exploration_json['exploration_type']}' unknown. Check manually."
                    )
                    return 1

                del local_path

        # timings = timings_sum / system_count

        if (
            exploration_json["systems_auto"][system_auto]["exploration_type"]
            == "lammps"
        ):
            average_per_step = (
                np.array(timings)
                / exploration_json["systems_auto"][system_auto]["nb_steps"]
            )
        elif (
            exploration_json["systems_auto"][system_auto]["exploration_type"] == "i-PI"
        ):
            average_per_step = np.array(timings)

        exploration_json["systems_auto"][system_auto]["mean_s_per_step"] = np.average(
            average_per_step
        )
        exploration_json["systems_auto"][system_auto]["median_s_per_step"] = np.median(
            average_per_step
        )
        exploration_json["systems_auto"][system_auto][
            "stdeviation_s_per_step"
        ] = np.std(average_per_step)

        del timings, average_per_step, system_count

    del system_auto, it_nnp, it_number

    logging.info(f"-" * 88)
    # Update the booleans in the exploration JSON
    if (completed_count + skipped_count + forced_count) == (
        exploration_json["nnp_count"]
        * sum(
            [
                exploration_json["systems_auto"][_]["traj_count"]
                for _ in exploration_json["systems_auto"]
            ]
        )
    ):
        exploration_json["is_checked"] = True

    # Dump the JSON files (exploration JSON)
    write_json_file(
        exploration_json,
        (control_path / f"exploration_{padded_curr_iter}.json"),
    )

    # End
    logging.info(f"-" * 88)
    if (completed_count + skipped_count + forced_count) != (
        exploration_json["nnp_count"]
        * sum(
            [
                exploration_json["systems_auto"][_]["traj_count"]
                for _ in exploration_json["systems_auto"]
            ]
        )
    ):
        logging.error(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a failure!"
        )
        logging.error(f"Please check manually before relaunching this step.")
        logging.error(f"Or create files named 'skip' or 'force' to skip or force.")
        logging.error(f"Aborting...")
        return 1

    del completed_count

    if (skipped_count + forced_count) != 0:
        logging.warning(f"'{skipped_count}' systems were skipped.")
        logging.warning(f"'{forced_count}' systems were forced.")
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )
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
