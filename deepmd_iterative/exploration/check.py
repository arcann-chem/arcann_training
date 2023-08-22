"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/22
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
from deepmd_iterative.common.filesystem import (
    remove_files_matching_glob,
)
from deepmd_iterative.common.check import validate_step_folder


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_config_filename: str = "input.json",
):
    current_path = Path(".").resolve()
    training_path = current_path.parent

    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}"
    )
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if correct folder
    validate_step_folder(current_step)

    # Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)

    # Get control path and main_config
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))
    exploration_config = load_json_file(
        (control_path / f"exploration_{current_iteration_zfill}.json")
    )

    # Checks
    if not exploration_config["is_launched"]:
        logging.error(f"Lock found. Execute first: exploration launch")
        logging.error(f"Aborting...")
        return 1

    # Check the normal termination of the exploration phase
    # Counters
    completed_count = 0
    skipped_count = 0
    forced_count = 0

    for system_auto_index, system_auto in enumerate(main_config["systems_auto"]):
        # Counters
        average_per_step = 0
        system_count = 0
        timings_sum = 0
        timings = []
        exploration_config["systems_auto"][system_auto] = {
            **exploration_config["systems_auto"][system_auto],
            "completed_count": 0,
            "forced_count": 0,
            "skipped_count": 0,
        }

        for it_nnp in range(1, main_config["nnp_count"] + 1):
            for it_number in range(1, exploration_config["traj_count"] + 1):
                local_path = (
                    Path(".").resolve()
                    / str(system_auto)
                    / str(it_nnp)
                    / (str(it_number).zfill(5))
                )

                # LAMMPS
                if exploration_config["exploration_type"] == "lammps":
                    lammps_output_file = (
                        local_path
                        / f"{system_auto}_{it_nnp}_{current_iteration_zfill}.log"
                    )
                    if lammps_output_file.is_file():
                        lammps_output = textfile_to_string_list(lammps_output_file)
                        if any("Total wall time:" in f for f in lammps_output):
                            system_count += 1
                            completed_count += 1
                            exploration_config["systems_auto"][system_auto][
                                "completed_count"
                            ] += 1
                            timings = [
                                zzz for zzz in lammps_output if "Loop time of" in zzz
                            ]
                            timings_sum += float(timings[0].split(" ")[3])
                        elif (local_path / "skip").is_file():
                            skipped_count += 1
                            exploration_config["systems_auto"][system_auto][
                                "skipped_count"
                            ] += 1
                            logging.warning(f"{lammps_output_file} skipped")
                        elif (local_path / "force").is_file():
                            forced_count += 1
                            exploration_config["systems_auto"][system_auto][
                                "forced_count"
                            ] += 1
                            logging.warning(f"{lammps_output_file} forced")
                        else:
                            logging.critical(
                                f"{lammps_output_file} failed. Check manually"
                            )
                        del lammps_output
                    elif (local_path / "skip").is_file():
                        skipped_count += 1
                        exploration_config["systems_auto"][system_auto][
                            "skipped_count"
                        ] += 1
                        logging.warning(f"{lammps_output_file} skipped")
                    else:
                        logging.critical(f"{lammps_output_file} failed. Check manually")
                    del lammps_output_file

                # i-PI
                elif exploration_config["exploration_type"] == "i-PI":
                    ipi_output_file = (
                        local_path
                        / f"{system_auto}_{it_nnp}_{current_iteration_zfill}.i-PI.server.log"
                    )
                    if ipi_output_file.is_file():
                        ipi_output = textfile_to_string_list(ipi_output_file)
                        if any("SIMULATION: Exiting cleanly" in f for f in ipi_output):
                            system_count += 1
                            completed_count += 1
                            exploration_config["systems_auto"][system_auto][
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
                            timings = np.average(np.asarray(ipi_time2, dtype="float32"))
                            timings_sum += timings
                            del ipi_time, ipi_time2, timings
                        elif (local_path / "skip").is_file():
                            skipped_count += 1
                            exploration_config["systems_auto"][system_auto][
                                "skipped_count"
                            ] += 1
                            logging.warning(f"{ipi_output_file} skipped")
                        elif (local_path / "force").is_file():
                            forced_count += 1
                            exploration_config["systems_auto"][system_auto][
                                "forced_count"
                            ] += 1
                            logging.warning(f"{ipi_output_file} forced")
                        else:
                            logging.critical(
                                f"{ipi_output_file} failed. Check manually"
                            )
                        del ipi_output
                    elif (local_path / "skip").is_file():
                        skipped_count += 1
                        exploration_config["systems_auto"][system_auto][
                            "skipped_count"
                        ] += 1
                        logging.warning(f"{ipi_output_file} skipped")
                    else:
                        logging.critical(f"{ipi_output_file} failed. Check manually")
                    del ipi_output_file

                else:
                    logging.error(
                        f"{exploration_config['exploration_type']} unknown. Check manually"
                    )
                    return 1

                del local_path

        timings = timings_sum / system_count

        if exploration_config["exploration_type"] == "lammps":
            average_per_step = (
                timings / exploration_config["systems_auto"][system_auto]["nb_steps"]
            )
        elif exploration_config["exploration_type"] == "i-PI":
            average_per_step = timings
        exploration_config["systems_auto"][system_auto]["s_per_step"] = average_per_step
        del timings, average_per_step, system_count, timings_sum

    del system_auto, it_nnp, it_number

    if (completed_count + skipped_count + forced_count) != (
        len(exploration_config["systems_auto"])
        * exploration_config["nnp_count"]
        * exploration_config["traj_count"]
    ):
        logging.error(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a failure !"
        )
        logging.error("Please check manually before relaunching this step")
        logging.error('Or create files named "skip" or "force" to skip or force')
        logging.error("Aborting...")
        return 1

    exploration_config["is_checked"] = True
    write_json_file(
        exploration_config,
        (control_path / f"exploration_{current_iteration_zfill}.json"),
    )
    logging.info("Deleting SLURM out/error files...")
    logging.info("Deleting NNP PB files...")
    logging.info("Removing extra log/error files...")
    for system_auto in exploration_config["systems_auto"]:
        for it_nnp in range(1, exploration_config["nnp_count"] + 1):
            for it_number in range(1, exploration_config["traj_count"] + 1):
                local_path = (
                    Path(".").resolve()
                    / str(system_auto)
                    / str(it_nnp)
                    / (str(it_number).zfill(5))
                )
                if exploration_config["exploration_type"] == "lammps":
                    remove_files_matching_glob(local_path, "LAMMPS_*")
                    remove_files_matching_glob(local_path, "*.pb")
                elif exploration_config["exploration_type"] == "i-PI":
                    remove_files_matching_glob(local_path, "i-PI_DeepMD*")
                    remove_files_matching_glob(local_path, "*.DP-i-PI.client_*.log")
                    remove_files_matching_glob(local_path, "*.DP-i-PI.client_*.err")
                del local_path
            del it_number
        del it_nnp
    del system_auto
    del completed_count
    logging.info("Cleaning done!")

    if (skipped_count + forced_count) != 0:
        logging.warning(f"{skipped_count} systems were skipped")
        logging.warning(f"{forced_count} systems were forced")
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success !"
    )
    del skipped_count, forced_count

    # Cleaning
    del control_path
    del main_config
    del current_iteration, current_iteration_zfill
    del exploration_config
    del training_path, current_path

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "check",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_config_filename=sys.argv[3],
        )
    else:
        pass
