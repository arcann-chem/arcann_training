"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/02/15
"""
# Standard library modules
import copy
import logging
import sys
from copy import deepcopy
from pathlib import Path

# Local imports
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.labeling.utils import (
    generate_input_labeling_json,
    get_system_labeling,
)
from deepmd_iterative.common.filesystem import (
    check_file_existence,
)
from deepmd_iterative.common.json import (
    backup_and_overwrite_json_file,
    get_key_in_dict,
    load_default_json_file,
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.list import (
    replace_substring_in_string_list,
    string_list_to_textfile,
    textfile_to_string_list,
)
from deepmd_iterative.common.machine import (
    get_machine_keyword,
    get_machine_spec_for_step,
)
from deepmd_iterative.common.slurm import replace_in_slurm_file_general
from deepmd_iterative.common.xyz import read_xyz_trajectory, write_xyz_frame


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
    logging.debug(f"curr_iter, padded_curr_iter: {curr_iter}, {padded_curr_iter}")

    # Load the default input JSON
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "assets" / "default_config.json"
    )[current_step]
    default_input_json_present = bool(default_input_json)
    if default_input_json_present and not (current_path / "default_input.json").is_file():
        write_json_file(default_input_json, (current_path / "default_input.json"))
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

    # Get control path and load the main JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))

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

    # Get the machine keyword (Priority: user > previous > default)
    # And update the merged input JSON
    user_machine_keyword = get_machine_keyword(
        user_input_json, previous_labeling_json, default_input_json, "label"
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    # Set it to None if bool, because: get_machine_spec_for_step needs None
    user_machine_keyword = (
        None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")

    # From the keyword (or default), get the machine spec (or for the fake one)
    (
        machine,
        machine_walltime_format,
        machine_job_scheduler,
        machine_launch_command,
        machine_max_jobs,
        machine_max_array_size,
        user_machine_keyword,
        machine_spec,
    ) = get_machine_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "labeling",
        fake_machine,
        user_machine_keyword,
    )
    arch_type = machine_spec["arch_type"]
    logging.debug(f"machine: {machine}")
    logging.debug(f"machine_walltime_format: {machine_walltime_format}")
    logging.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    logging.debug(f"machine_launch_command: {machine_launch_command}")
    logging.debug(f"machine_max_jobs: {machine_max_jobs}")
    logging.debug(f"machine_max_array_size: {machine_max_array_size}")
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    logging.debug(f"machine_spec: {machine_spec}")

    merged_input_json["user_machine_keyword_label"] = user_machine_keyword
    logging.debug(f"merged_input_json: {merged_input_json}")

    if fake_machine is not None:
        logging.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        logging.info(f"Machine identified: '{machine}'.")
    del fake_machine

    # Check if we can continue
    exploration_json = load_json_file(
        (control_path / f"exploration_{padded_curr_iter}.json")
    )
    if not exploration_json["is_extracted"]:
        logging.error(f"Lock found. Run/Check first: exploration extract.")
        logging.error(f"Aborting...")
        return 1

    # Generate/update the merged input JSON
    # Priority: user > previous > default
    merged_input_json = generate_input_labeling_json(
        user_input_json,
        previous_labeling_json,
        default_input_json,
        merged_input_json,
        main_json,
    )
    logging.debug(f"merged_input_json: {merged_input_json}")

    # Generate the labeling JSON
    labeling_json = {}
    labeling_json = {
        **labeling_json,
        "user_machine_keyword_label": user_machine_keyword,
    }

    # Check if the job file exists
    job_file_array_name = f"job-array_CP2K_label_{arch_type}_{machine}.sh"
    job_file_name = f"job_CP2K_label_{arch_type}_{machine}.sh"

    # 0 is array, 1 is individual
    master_job_file = {}
    for filename_idx, filename in enumerate([job_file_array_name, job_file_name]):
        if (training_path / "user_files" / filename).is_file():
            master_job_file[filename_idx] = textfile_to_string_list(
                training_path / "user_files" / filename
            )
        else:
            logging.error(
                f"No JOB file provided for '{current_step.capitalize()} / {current_phase.capitalize()}' for this machine."
            )
            logging.error(f"Aborting...")
            return 1

        logging.debug(
            f"master_job_file: {master_job_file[filename_idx][0:5]}, {master_job_file[filename_idx][-5:-1]}"
        )
    del filename_idx, filename

    merged_input_json["job_email"] = get_key_in_dict(
        "job_email", user_input_json, previous_labeling_json, default_input_json
    )
    del job_file_array_name, job_file_name

    labeling_json["systems_auto"] = {}

    total_to_label = 0

    job_array_params_file = {}
    job_array_params_file["cp2k"] = [
        "PATH/CP2K_INPUT_F1/CP2K_INPUT_F2/CP2K_WFRST_F/CP2K_XYZ_F/"
    ]

    # Loop through each system and set its labeling
    for system_auto_index, system_auto in enumerate(exploration_json["systems_auto"]):
        logging.info(
            f"Processing system: {system_auto} ({system_auto_index + 1}/{len(exploration_json['systems_auto'])})"
        )

        labeling_json["systems_auto"][system_auto] = {}
        candidates_count = exploration_json["systems_auto"][system_auto][
            "selected_count"
        ]

        if (
            exploration_json["systems_auto"][system_auto]["disturbed_candidate_value"]
            > 0
        ):
            disturbed_candidates_count = candidates_count
        else:
            disturbed_candidates_count = 0
        labeling_count = candidates_count + disturbed_candidates_count

        total_to_label += labeling_count

        (
            system_walltime_first_job_h,
            system_walltime_second_job_h,
            system_nb_nodes,
            system_nb_mpi_per_node,
            system_nb_threads_per_mpi,
        ) = get_system_labeling(merged_input_json, system_auto_index)
        logging.debug(
            f"{system_walltime_first_job_h,system_walltime_second_job_h,system_nb_nodes,system_nb_mpi_per_node,system_nb_threads_per_mpi}"
        )

        if labeling_count == 0:
            labeling_json["systems_auto"][system_auto][
                "walltime_first_job_h"
            ] = system_walltime_first_job_h
            labeling_json["systems_auto"][system_auto][
                "walltime_second_job_h"
            ] = system_walltime_second_job_h
            labeling_json["systems_auto"][system_auto]["nb_nodes"] = system_nb_nodes
            labeling_json["systems_auto"][system_auto][
                "nb_mpi_per_node"
            ] = system_nb_mpi_per_node
            labeling_json["systems_auto"][system_auto][
                "nb_threads_per_mpi"
            ] = system_nb_threads_per_mpi
            labeling_json["systems_auto"][system_auto][
                "candidates_count"
            ] = candidates_count
            labeling_json["systems_auto"][system_auto][
                "disturbed_candidates_count"
            ] = disturbed_candidates_count
            continue

        if curr_iter > 1 and (
            "walltime_first_job_h" not in user_input_json
            or user_input_json["walltime_first_job_h"][system_auto_index] == -1
        ):
            system_walltime_first_job_h = max(
                previous_labeling_json["systems_auto"][system_auto]["timings_s"][0]
                / 3600
                * 1.5,
                1 / 2,
            )
        if curr_iter > 1 and (
            "walltime_second_job_h" not in user_input_json
            or user_input_json["walltime_second_job_h"][system_auto_index] == -1
        ):
            system_walltime_second_job_h = max(
                previous_labeling_json["systems_auto"][system_auto]["timings_s"][0]
                / 3600
                * 1.5,
                1 / 2,
            )

        # TODO Do we update or leave it as is (-1 if default, user value else)
        merged_input_json["walltime_first_job_h"][
            system_auto_index
        ] = system_walltime_first_job_h
        merged_input_json["walltime_second_job_h"][
            system_auto_index
        ] = system_walltime_second_job_h

        system_path = current_path / system_auto
        system_path.mkdir(exist_ok=True)

        # Replace slurm
        walltime_approx_s = int(
            (system_walltime_first_job_h + system_walltime_second_job_h) * 3600
        )

        system_master_job_file = deepcopy(master_job_file)
        for _ in system_master_job_file:
            system_master_job_file[_] = replace_substring_in_string_list(
                system_master_job_file[_], "_R_nb_NODES_", f"{system_nb_nodes}"
            )
            system_master_job_file[_] = replace_substring_in_string_list(
                system_master_job_file[_],
                "_R_nb_MPIPERNODE_",
                f"{system_nb_mpi_per_node}",
            )
            system_master_job_file[_] = replace_substring_in_string_list(
                system_master_job_file[_],
                "_R_nb_THREADSPERMPI_",
                f"{system_nb_threads_per_mpi}",
            )
            system_master_job_file[_] = replace_substring_in_string_list(
                system_master_job_file[_],
                "_R_nb_MPI_",
                f"{system_nb_nodes * system_nb_mpi_per_node}",
            )
            system_master_job_file[_] = replace_in_slurm_file_general(
                system_master_job_file[_],
                machine_spec,
                walltime_approx_s,
                machine_walltime_format,
                merged_input_json["job_email"],
            )
        del walltime_approx_s

        # TODO Logic to put the array slurm HERE
        if total_to_label <= machine_spec["machine_max_array_size"]:
            system_master_job_file[0] = replace_substring_in_string_list(system_master_job_file[0], "_R_NEW_START_", "0")
            if total_to_label <= machine_spec["machine_max_jobs"]:
                system_master_job_file[0] = replace_substring_in_string_list(system_master_job_file[0], "_R_ARRAY_START_", "1")
                system_master_job_file[0] = replace_substring_in_string_list(system_master_job_file[0], "_R_ARRAY_END_", str(total_to_label))
                if system_auto_index != len(exploration_json["systems_auto"]) - 1:
                    system_master_job_file[0] = replace_substring_in_string_list(system_master_job_file[0], "_R_LAUNCHNEXT_", "1")
                    system_master_job_file[0] = replace_substring_in_string_list(system_master_job_file[0], "_R_NEXT_JOB_FILE_", "0")
                    system_master_job_file[0] = replace_substring_in_string_list(system_master_job_file[0], "_R_CD_WHERE_", "${SLURM_SUBMIT_DIR}/../"+exploration_json["systems_auto"][system_auto_index+1])
                else:
                    True
                string_list_to_textfile(system_path / f"job-array_CP2K_label_{arch_type}_{machine}.sh", system_master_job_file[0])

            else:
                slurm_file_array_subsys_dict={}
                quotient = total_to_label // machine_spec["machine_max_jobs"]
                remainder = total_to_label % machine_spec["machine_max_jobs"]

                for i in range(0,quotient+1):
                    if i < quotient:
                        slurm_file_array_subsys_dict[i] = replace_substring_in_string_list(system_master_job_file[0], "_R_ARRAY_START_", str(machine_spec["machine_max_jobs"]*i + 1))
                        slurm_file_array_subsys_dict[i] = replace_substring_in_string_list(slurm_file_array_subsys_dict[i], "_R_ARRAY_END_", str(machine_spec["machine_max_jobs"]*(i+1)))
                        slurm_file_array_subsys_dict[i] = replace_substring_in_string_list(slurm_file_array_subsys_dict[i], "_R_LAUNCHNEXT_", "1")
                        slurm_file_array_subsys_dict[i] = replace_substring_in_string_list(slurm_file_array_subsys_dict[i], "_R_NEXT_JOB_FILE_", "0")
                        slurm_file_array_subsys_dict[i] = replace_substring_in_string_list(slurm_file_array_subsys_dict[i], "_R_CD_WHERE_", "${SLURM_SUBMIT_DIR}")
                        string_list_to_textfile(system_path / f"job-array_CP2K_label_{arch_type}_{machine}_{i}.sh", slurm_file_array_subsys_dict[i])
                    else:
                        slurm_file_array_subsys_dict[i] = replace_substring_in_string_list(system_master_job_file[0], "_R_ARRAY_START_", str(machine_spec["machine_max_jobs"]*i + 1))
                        slurm_file_array_subsys_dict[i] = replace_substring_in_string_list(slurm_file_array_subsys_dict[i], "_R_ARRAY_END_", str(machine_spec["machine_max_jobs"]*i + remainder))
                        if system_auto_index != len(exploration_json["systems_auto"]) - 1:
                            slurm_file_array_subsys_dict[i] = replace_substring_in_string_list(slurm_file_array_subsys_dict[i], "_R_LAUNCHNEXT_", "1")
                            slurm_file_array_subsys_dict[i] = replace_substring_in_string_list(slurm_file_array_subsys_dict[i], "_R_NEXT_JOB_FILE_", "0")
                            slurm_file_array_subsys_dict[i] = replace_substring_in_string_list(slurm_file_array_subsys_dict[i], "_R_CD_WHERE_", "${SLURM_SUBMIT_DIR}/../"+exploration_json["systems_auto"][system_auto_index+1])
                        else:
                            True
                        string_list_to_textfile(system_path / f"job-array_CP2K_label_{arch_type}_{machine}_{i}.sh", slurm_file_array_subsys_dict[i])
                del quotient, remainder, i
                del slurm_file_array_subsys_dict
        else:
            slurm_file_array_subsys_dict={}
            quotient = total_to_label // machine_spec["machine_max_array_size"]
            remainder = total_to_label % machine_spec["machine_max_array_size"]
            m = 0
            for i in range(0,quotient+1):
                if i < quotient:
                    for j in range(0,machine_spec["machine_max_array_size"] // machine_spec["machine_max_jobs"]):
                        slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(system_master_job_file[0], "_R_NEW_START", str(machine_spec["machine_max_array_size"]*i))
                        slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_ARRAY_START_", str(machine_spec["machine_max_jobs"]*j + 1))
                        slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_ARRAY_END_", str(machine_spec["machine_max_jobs"]*(j+1)))
                        slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_LAUNCHNEXT_", "1")
                        slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_NEXT_JOB_FILE_", "0")
                        slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_CD_WHERE_", "${SLURM_SUBMIT_DIR}")
                        string_list_to_textfile(system_path / f"job-array_CP2K_label_{arch_type}_{machine}_{m}.sh", slurm_file_array_subsys_dict[m])
                        m += 1
                else:
                    quotient2 = remainder // machine_spec["machine_max_jobs"]
                    remainder2 = remainder % machine_spec["machine_max_jobs"]
                    for j in range(0,quotient2+1):
                        if j < quotient2:
                            slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(system_master_job_file[0], "_R_NEW_START", str(machine_spec["machine_max_array_size"]*i))
                            slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_ARRAY_START_", str(machine_spec["machine_max_jobs"]*j + 1))
                            slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_ARRAY_END_", str(machine_spec["machine_max_jobs"]*(j+1)))
                            slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_LAUNCHNEXT_", "1")
                            slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_NEXT_JOB_FILE_", str(m+1))
                            slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_CD_WHERE_", "${SLURM_SUBMIT_DIR}")
                            string_list_to_textfile(system_path / f"job-array_CP2K_label_{arch_type}_{machine}_{m}.sh", slurm_file_array_subsys_dict[m])
                            m += 1
                        else:
                            slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(system_master_job_file[0], "_R_NEW_START", str(machine_spec["machine_max_array_size"]*i))
                            slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_ARRAY_START_", str(machine_spec["machine_max_jobs"]*j + 1))
                            slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_ARRAY_END_", str(machine_spec["machine_max_jobs"]*j + remainder2))
                            if system_auto_index != len(exploration_json["systems_auto"]) - 1:
                                slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_LAUNCHNEXT_", "1")
                                slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_NEXT_JOB_FILE_", "0")
                                slurm_file_array_subsys_dict[m] = replace_substring_in_string_list(slurm_file_array_subsys_dict[m], "_R_CD_WHERE_", "${SLURM_SUBMIT_DIR}/../"+exploration_json["systems_auto"][system_auto_index+1])
                            else:
                                True
                            string_list_to_textfile(system_path / f"job-array_CP2K_label_{arch_type}_{machine}_{m}.sh", slurm_file_array_subsys_dict[m])
                            m += 1
            del quotient, remainder, quotient2, remainder2, i, j

        system_master_job_file[0] = replace_substring_in_string_list(
            system_master_job_file[0], "_R_ARRAY_START_", "0"
        )
        system_master_job_file[0] = replace_substring_in_string_list(
            system_master_job_file[0], "_R_ARRAY_END_", f"{labeling_count-1}"
        )
        system_master_job_file[0] = replace_substring_in_string_list(
            system_master_job_file[0],
            "_R_CP2K_JOBNAME_",
            f"CP2K_{system_auto}_{padded_curr_iter}",
        )

        string_list_to_textfile(
            system_path / f"job-array_CP2K_label_{arch_type}_{machine}.sh",
            system_master_job_file[0],
        )

        # Labeling input first job
        system_first_job_input = textfile_to_string_list(
            training_path
            / "user_files"
            / f"1_{system_auto}_labeling_XXXXX_{machine}.inp"
        )
        system_first_job_input = replace_substring_in_string_list(
            system_first_job_input,
            "_R_WALLTIME_",
            f"{system_walltime_first_job_h * 3600}",
        )
        system_first_job_input = replace_substring_in_string_list(
            system_first_job_input,
            "_R_CELL_",
            " ".join(
                [str(zzz) for zzz in main_json["systems_auto"][system_auto]["cell"]]
            ),
        )

        # Labeling input second job
        system_second_job_input = textfile_to_string_list(
            training_path
            / "user_files"
            / f"2_{system_auto}_labeling_XXXXX_{machine}.inp"
        )
        system_second_job_input = replace_substring_in_string_list(
            system_second_job_input,
            "_R_WALLTIME_",
            f"{system_walltime_second_job_h * 3600}",
        )
        system_second_job_input = replace_substring_in_string_list(
            system_second_job_input,
            "_R_CELL_",
            " ".join(
                [str(zzz) for zzz in main_json["systems_auto"][system_auto]["cell"]]
            ),
        )

        # Regular
        xyz_file = (
            training_path
            / f"{padded_curr_iter}-exploration"
            / system_auto
            / f"candidates_{padded_curr_iter}_{system_auto}.xyz"
        )
        num_atoms, atom_symbols, atom_coords, cell_info = read_xyz_trajectory(xyz_file)
        del xyz_file

        if atom_coords.shape[0] != candidates_count:
            logging.error(
                f"The number of structures in the xyz does not match the number of candidates."
            )
            logging.error(f"Aborting...")
            return 1

        for labeling_step in range(atom_coords.shape[0]):
            padded_labeling_step = str(labeling_step).zfill(5)
            labeling_step_path = system_path / padded_labeling_step
            labeling_step_path.mkdir(exist_ok=True)

            first_job_input_t = deepcopy(system_first_job_input)
            first_job_input_t = replace_substring_in_string_list(
                first_job_input_t, "XXXXX", padded_labeling_step
            )
            string_list_to_textfile(
                labeling_step_path / f"1_labeling_{padded_labeling_step}.inp",
                first_job_input_t,
            )
            del first_job_input_t

            second_job_input_t = deepcopy(system_second_job_input)
            second_job_input_t = replace_substring_in_string_list(
                second_job_input_t, "XXXXX", padded_labeling_step
            )
            string_list_to_textfile(
                labeling_step_path / f"2_labeling_{padded_labeling_step}.inp",
                second_job_input_t,
            )
            del second_job_input_t

            job_file_t = deepcopy(system_master_job_file[1])
            job_file_t = replace_substring_in_string_list(
                job_file_t, "XXXXX", padded_labeling_step
            )
            job_file_t = replace_substring_in_string_list(
                job_file_t, "_R_CP2K_JOBNAME_", f"CP2K_{system_auto}_{padded_curr_iter}"
            )
            string_list_to_textfile(
                labeling_step_path
                / f"job_CP2K_label_{padded_labeling_step}_{arch_type}_{machine}.sh",
                job_file_t,
            )
            del job_file_t

            write_xyz_frame(
                labeling_step_path / f"labeling_{padded_labeling_step}.xyz",
                labeling_step,
                num_atoms,
                atom_symbols,
                atom_coords,
                cell_info,
            )
            job_array_params_line = (
                str(system_auto) + "_" + str(padded_labeling_step) + "/"
            )
            job_array_params_line += f"1_labeling_{padded_labeling_step}" + "/"
            job_array_params_line += f"2_labeling_{padded_labeling_step}" + "/"
            job_array_params_line += f"labeling_{padded_labeling_step}-SCF.wfn" + "/"
            job_array_params_line += f"labeling_{padded_labeling_step}.xyz" + "/"
            job_array_params_line += "" + "/"
            job_array_params_file["cp2k"].append(job_array_params_line)
            del padded_labeling_step, labeling_step_path

        del labeling_step
        del num_atoms, atom_symbols, atom_coords, cell_info

        # Disturbed
        xyz_file_disturbed = (
            training_path
            / f"{padded_curr_iter}-exploration"
            / system_auto
            / f"candidates_{padded_curr_iter}_{system_auto}_disturbed.xyz"
        )
        if xyz_file_disturbed.is_file():
            num_atoms, atom_symbols, atom_coords, cell_info = read_xyz_trajectory(
                xyz_file_disturbed
            )
            del xyz_file_disturbed

            if atom_coords.shape[0] != candidates_count:
                logging.error(
                    f"The number of structures in the xyz does not match the number of candidates."
                )
                logging.error(f"Aborting...")
                return 1

            for labeling_step_idx, labeling_step in enumerate(
                range(candidates_count, candidates_count + atom_coords.shape[0])
            ):
                padded_labeling_step = str(labeling_step).zfill(5)
                labeling_step_path = system_path / padded_labeling_step
                labeling_step_path.mkdir(exist_ok=True)

                first_job_input_t = deepcopy(system_first_job_input)
                first_job_input_t = replace_substring_in_string_list(
                    first_job_input_t, "XXXXX", padded_labeling_step
                )
                string_list_to_textfile(
                    labeling_step_path / f"1_labeling_{padded_labeling_step}.inp",
                    first_job_input_t,
                )
                del first_job_input_t

                second_job_input_t = deepcopy(system_second_job_input)
                second_job_input_t = replace_substring_in_string_list(
                    second_job_input_t, "XXXXX", padded_labeling_step
                )
                string_list_to_textfile(
                    labeling_step_path / f"2_labeling_{padded_labeling_step}.inp",
                    second_job_input_t,
                )
                del second_job_input_t

                job_file_t = deepcopy(system_master_job_file[1])
                job_file_t = replace_substring_in_string_list(
                    job_file_t, "XXXXX", padded_labeling_step
                )
                job_file_t = replace_substring_in_string_list(
                    job_file_t,
                    "_R_CP2K_JOBNAME_",
                    f"CP2K_{system_auto}_{padded_curr_iter}",
                )
                string_list_to_textfile(
                    labeling_step_path
                    / f"job_CP2K_label_{padded_labeling_step}_{arch_type}_{machine}.sh",
                    job_file_t,
                )
                del job_file_t

                write_xyz_frame(
                    labeling_step_path / f"labeling_{padded_labeling_step}.xyz",
                    labeling_step_idx,
                    num_atoms,
                    atom_symbols,
                    atom_coords,
                    cell_info,
                )
                job_array_params_line = (
                    str(system_auto) + "_" + str(padded_labeling_step) + "/"
                )
                job_array_params_line += f"1_labeling_{padded_labeling_step}" + "/"
                job_array_params_line += f"2_labeling_{padded_labeling_step}" + "/"
                job_array_params_line += (
                    f"labeling_{padded_labeling_step}-SCF.wfn" + "/"
                )
                job_array_params_line += f"labeling_{padded_labeling_step}.xyz" + "/"
                job_array_params_line += "" + "/"
                job_array_params_file["cp2k"].append(job_array_params_line)

                del padded_labeling_step, labeling_step_path

            del labeling_step
            del num_atoms, atom_symbols, atom_coords, cell_info

        # Update labeling JSON
        labeling_json["systems_auto"][system_auto][
            "walltime_first_job_h"
        ] = system_walltime_first_job_h
        labeling_json["systems_auto"][system_auto][
            "walltime_second_job_h"
        ] = system_walltime_second_job_h
        labeling_json["systems_auto"][system_auto]["nb_nodes"] = system_nb_nodes
        labeling_json["systems_auto"][system_auto][
            "nb_mpi_per_node"
        ] = system_nb_mpi_per_node
        labeling_json["systems_auto"][system_auto][
            "nb_threads_per_mpi"
        ] = system_nb_threads_per_mpi
        labeling_json["systems_auto"][system_auto][
            "candidates_count"
        ] = candidates_count
        labeling_json["systems_auto"][system_auto][
            "disturbed_candidates_count"
        ] = disturbed_candidates_count

        # System dependent cleaning
        del system_first_job_input, system_second_job_input, system_master_job_file
        del (
            system_walltime_first_job_h,
            system_walltime_second_job_h,
            system_nb_nodes,
            system_nb_mpi_per_node,
            system_nb_threads_per_mpi,
        )
        del candidates_count, disturbed_candidates_count, labeling_count

        logging.info(
            f"Processed system: {system_auto} ({system_auto_index + 1}/{len(main_json['systems_auto'])})"
        )
    del system_auto_index, system_auto
    logging.info(f"{total_to_label} structures will be labeled.")
    del total_to_label

    string_list_to_textfile(
        current_path / f"job-array-params_CP2K_label_{arch_type}_{machine}.lst",
        job_array_params_file["cp2k"],
    )

    # Set booleans in the exploration JSON
    labeling_json = {
        **labeling_json,
        "is_locked": True,
        "is_launched": False,
        "is_checked": False,
        "is_extracted": False,
    }

    # Dump the JSON files (main, exploration and merged input)
    write_json_file(main_json, (control_path / "config.json"))
    write_json_file(labeling_json, (control_path / f"labeling_{padded_curr_iter}.json"))
    backup_and_overwrite_json_file(
        merged_input_json, (current_path / user_input_json_filename)
    )

    # End
    logging.info(f"-" * 88)
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del (
        default_input_json,
        default_input_json_present,
        user_input_json,
        user_input_json_present,
        user_input_json_filename,
    )
    del (
        main_json,
        merged_input_json,
        labeling_json,
        previous_labeling_json,
        exploration_json,
    )
    del curr_iter, padded_curr_iter
    del (
        machine,
        machine_walltime_format,
        machine_job_scheduler,
        machine_launch_command,
        user_machine_keyword,
        machine_spec,
    )
    del master_job_file

    logging.debug(f"LOCAL")
    logging.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "labeling",
            "preparation",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
