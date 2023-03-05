from pathlib import Path
import logging
import sys

# ### Non-standard imports
import numpy as np

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.file import (
    file_to_list_of_strings,
    remove_files_matching_glob,
)
from deepmd_iterative.common.check import validate_step_folder


def main(
    step_name,
    phase_name,
    deepmd_iterative_path,
    fake_machine=None,
    input_fn="input.json",
):
    current_path = Path(".").resolve()
    training_path = current_path.parent

    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()}")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # ### Check if correct folder
    validate_step_folder(step_name)

    # ### Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)

    # ### Get control path and config_json
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))
    exploration_json = load_json_file(
        (control_path / f"exploration_{current_iteration_zfill}.json")
    )

    # ### Checks
    if not exploration_json["is_launched"]:
        logging.error(f"Lock found. Execute first: exploration launch")
        logging.error(f"Aborting...")
        return 1

    # ### Check the normal termination of the exploration phase
    # ### Counters
    completed_count = 0
    skipped_count = 0
    forced_count = 0

    for it0_subsys_nr, it_subsys_nr in enumerate(config_json["subsys_nr"]):
        # ### Counters
        average_per_step = 0
        subsys_count = 0
        timings_sum = 0
        timings = []
        exploration_json['subsys_nr'][it_subsys_nr]['nb_completed'] = 0
        exploration_json['subsys_nr'][it_subsys_nr]['nb_forced'] = 0
        exploration_json['subsys_nr'][it_subsys_nr]['nb_skipped'] = 0

        for it_nnp in range(1, config_json["nb_nnp"] + 1):
            for it_number in range(1, exploration_json["nb_traj"] + 1):

                local_path = (
                    Path(".").resolve()
                    / str(it_subsys_nr)
                    / str(it_nnp)
                    / (str(it_number).zfill(5))
                )

                # ### LAMMPS
                if exploration_json["exploration_type"] == "lammps":
                    lammps_output_file = (
                        local_path
                        / f"{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}.log"
                    )
                    if lammps_output_file.is_file():
                        lammps_output = file_to_list_of_strings(lammps_output_file)
                        if any("Total wall time:" in f for f in lammps_output):
                            subsys_count += 1
                            completed_count += 1
                            exploration_json['subsys_nr'][it_subsys_nr]['nb_completed'] += 1
                            timings = [
                                zzz for zzz in lammps_output if "Loop time of" in zzz
                            ]
                            timings_sum += float(timings[0].split(" ")[3])
                        elif (local_path / "skip").is_file():
                            skipped_count += 1
                            exploration_json['subsys_nr'][it_subsys_nr]['nb_skipped'] += 1
                            logging.warning(f"{lammps_output_file} skipped")
                        elif (local_path / "force").is_file():
                            forced_count += 1
                            exploration_json['subsys_nr'][it_subsys_nr]['nb_forced'] += 1
                            logging.warning(f"{lammps_output_file} forced")
                        else:
                            logging.critical(
                                f"{lammps_output_file} failed. Check manually"
                            )
                        del lammps_output
                    elif (local_path / "skip").is_file():
                        skipped_count += 1
                        exploration_json['subsys_nr'][it_subsys_nr]['nb_skipped'] += 1
                        logging.warning(f"{lammps_output_file} skipped")
                    else:
                        logging.critical(f"{lammps_output_file} failed. Check manually")
                    del lammps_output_file

                # ### i-PI
                elif exploration_json["exploration_type"] == "i-PI":
                    ipi_output_file = (
                        local_path
                        / f"{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}.i-PI.server.log"
                    )
                    if ipi_output_file.is_file():
                        ipi_output = file_to_list_of_strings(ipi_output_file)
                        if any("SIMULATION: Exiting cleanly" in f for f in ipi_output):
                            subsys_count += 1
                            completed_count += 1
                            exploration_json['subsys_nr'][it_subsys_nr]['nb_completed'] += 1
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
                            exploration_json['subsys_nr'][it_subsys_nr]['nb_skipped'] += 1
                            logging.warning(f"{ipi_output_file} skipped")
                        elif (local_path / "force").is_file():
                            forced_count += 1
                            exploration_json['subsys_nr'][it_subsys_nr]['nb_forced'] += 1
                            logging.warning(f"{ipi_output_file} forced")
                        else:
                            logging.critical(
                                f"{ipi_output_file} failed. Check manually"
                            )
                        del ipi_output
                    elif (local_path / "skip").is_file():
                        skipped_count += 1
                        exploration_json['subsys_nr'][it_subsys_nr]['nb_skipped'] += 1
                        logging.warning(f"{ipi_output_file} skipped")
                    else:
                        logging.critical(f"{ipi_output_file} failed. Check manually")
                    del ipi_output_file

                else:
                    logging.error(
                        f"{exploration_json['exploration_type']} unknown. Check manually"
                    )
                    return 1

                del local_path

        timings = timings_sum / subsys_count

        if exploration_json["exploration_type"] == "lammps":
            average_per_step = (
                timings / exploration_json["subsys_nr"][it_subsys_nr]["nb_steps"]
            )
        elif exploration_json["exploration_type"] == "i-PI":
            average_per_step = timings
        exploration_json["subsys_nr"][it_subsys_nr]["s_per_step"] = average_per_step
        del timings, average_per_step, subsys_count, timings_sum

    del it_subsys_nr, it_nnp, it_number

    if (completed_count + skipped_count + forced_count) != (
        len(exploration_json["subsys_nr"])
        * exploration_json["nb_nnp"]
        * exploration_json["nb_traj"]
    ):
        logging.error(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a failure !"
        )
        logging.error("Please check manually before relaunching this step")
        logging.error('Or create files named "skip" or "force" to skip or force')
        logging.error("Aborting...")
        return 1

    exploration_json["is_checked"] = True
    write_json_file(
        exploration_json, (control_path / f"exploration_{current_iteration_zfill}.json")
    )

    for it_subsys_nr in exploration_json["subsys_nr"]:
        for it_nnp in range(1, exploration_json["nb_nnp"] + 1):
            for it_number in range(1, exploration_json["nb_traj"] + 1):
                local_path = (
                    Path(".").resolve()
                    / str(it_subsys_nr)
                    / str(it_nnp)
                    / (str(it_number).zfill(5))
                )
                if exploration_json["exploration_type"] == "lammps":
                    logging.info("Deleting SLURM out/error files...")
                    remove_files_matching_glob(local_path, "LAMMPS_*")
                    logging.info("Deleting NNP PB files...")
                    remove_files_matching_glob(local_path, "*.pb")
                    logging.info("Cleaning done!")
                elif exploration_json["exploration_type"] == "i-PI":
                    logging.info("Deleting SLURM out/error files...")
                    remove_files_matching_glob(local_path, "i-PI_DeepMD*")
                    logging.info("Removing DP-i-PI log/error files...")
                    remove_files_matching_glob(local_path, "*.DP-i-PI.client_*.log")
                    remove_files_matching_glob(local_path, "*.DP-i-PI.client_*.err")
                del local_path
            del it_number
        del it_nnp
    del it_subsys_nr
    del completed_count

    if (skipped_count + forced_count) != 0:
        logging.warning(f"{skipped_count} systems were skipped")
        logging.warning(f"{forced_count} systems were forced")
    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
    )
    del skipped_count, forced_count

    # ### Cleaning
    del control_path
    del config_json
    del current_iteration, current_iteration_zfill
    del exploration_json
    del training_path, current_path

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "check",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
