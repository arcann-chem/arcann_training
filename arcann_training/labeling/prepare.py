"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15
"""

# Standard library modules
import logging
import sys
from pathlib import Path
from copy import deepcopy

# Non-standard library imports
import numpy as np

# Local imports
from arcann_training.common.check import validate_step_folder
from arcann_training.labeling.utils import generate_input_labeling_json, get_system_labeling
from arcann_training.common.json import backup_and_overwrite_json_file, get_key_in_dict, load_default_json_file, load_json_file, write_json_file
from arcann_training.common.list import replace_substring_in_string_list, string_list_to_textfile, textfile_to_string_list
from arcann_training.common.machine import get_machine_keyword, get_machine_spec_for_step
from arcann_training.common.slurm import replace_in_slurm_file_general
from arcann_training.common.xyz import parse_xyz_trajectory_file, write_xyz_frame


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
    arcann_logger.debug(f"curr_iter, padded_curr_iter: {curr_iter}, {padded_curr_iter}")

    # Load the default input JSON
    default_input_json = load_default_json_file(deepmd_iterative_path / "assets" / "default_config.json")[current_step]
    default_input_json_present = bool(default_input_json)
    if default_input_json_present and not (current_path / "default_input.json").is_file():
        write_json_file(default_input_json, (current_path / "default_input.json"))
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

    # Create a empty (None/Null) current input JSON
    current_input_json = {}
    for key in default_input_json:
        current_input_json[key] = None
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    # Get control path and load the main JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))

    # Load the previous labeling JSON
    if curr_iter > 1:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_labeling_json = load_json_file((control_path / f"labeling_{padded_prev_iter}.json"))
        del prev_iter, padded_prev_iter
    else:
        previous_labeling_json = {}

    # Get the machine keyword (Priority: user > previous > default)
    # And update the merged input JSON
    user_machine_keyword = get_machine_keyword(user_input_json, previous_labeling_json, default_input_json, "label")
    arcann_logger.debug(f"user_machine_keyword: {user_machine_keyword}")
    # Set it to None if bool, because: get_machine_spec_for_step needs None
    user_machine_keyword = None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    arcann_logger.debug(f"user_machine_keyword: {user_machine_keyword}")

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
    arcann_logger.debug(f"machine: {machine}")
    arcann_logger.debug(f"machine_walltime_format: {machine_walltime_format}")
    arcann_logger.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    arcann_logger.debug(f"machine_launch_command: {machine_launch_command}")
    arcann_logger.debug(f"machine_max_jobs: {machine_max_jobs}")
    arcann_logger.debug(f"machine_max_array_size: {machine_max_array_size}")
    arcann_logger.debug(f"user_machine_keyword: {user_machine_keyword}")
    arcann_logger.debug(f"machine_spec: {machine_spec}")

    current_input_json["user_machine_keyword_label"] = user_machine_keyword
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    if fake_machine is not None:
        arcann_logger.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        arcann_logger.info(f"Machine identified: '{machine}'.")
    del fake_machine

    # Check if we can continue
    exploration_json = load_json_file((control_path / f"exploration_{padded_curr_iter}.json"))
    if not exploration_json["is_extracted"]:
        arcann_logger.error(f"Lock found. Run/Check first: exploration extract.")
        arcann_logger.error(f"Aborting...")
        return 1

    # Generate/update the merged input JSON
    # Priority: user > previous > default
    current_input_json = generate_input_labeling_json(
        user_input_json,
        previous_labeling_json,
        default_input_json,
        current_input_json,
        main_json,
    )
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    labeling_program = current_input_json["labeling_program"]
    arcann_logger.debug(f"labeling_program: {labeling_program}")

    # Generate the labeling JSON
    labeling_json = {}
    labeling_json = {
        **labeling_json,
        "labeling_program": labeling_program,
        "user_machine_keyword_label": user_machine_keyword,
    }
    labeling_program_up = labeling_program.upper()
    # Check if the job file exists
    job_file_array_name = f"job-array_{labeling_program_up}_label_{machine_spec['arch_type']}_{machine}.sh"
    job_file_name = f"job_{labeling_program_up}_label_{machine_spec['arch_type']}_{machine}.sh"

    # 0 is array, 1 is individual
    master_job_file = {}
    for filename_idx, filename in enumerate([job_file_array_name, job_file_name]):
        if (training_path / "user_files" / filename).is_file():
            master_job_file[filename_idx] = textfile_to_string_list(training_path / "user_files" / filename)
        else:
            arcann_logger.error(f"No JOB file provided for '{current_step.capitalize()} / {current_phase.capitalize()}' for this machine.")
            arcann_logger.error(f"Aborting...")
            return 1
        arcann_logger.debug(f"master_job_file: {master_job_file[filename_idx][0:5]}, {master_job_file[filename_idx][-5:-1]}")
    del filename_idx, filename

    current_input_json["job_email"] = get_key_in_dict("job_email", user_input_json, previous_labeling_json, default_input_json)
    del job_file_array_name, job_file_name

    labeling_json["systems_auto"] = {}

    job_array_params_file = {f"{labeling_program}": [
        f":SYSTEM:INDEX:{labeling_program_up}_INPUT_F1:{labeling_program_up}_INPUT_F2:{labeling_program_up}_WFRST_F:{labeling_program_up}_XYZ_F:NODES:MPI_PER_NODE:THREADS_PER_MPI:WALLTIME_S:"]}

    # Get the list of systems to label to get the next one
    total_to_label = 0
    system_auto_list = []
    # First loop to get the total number of jobs and the lsit of system
    for system_auto_index, system_auto in enumerate(exploration_json["systems_auto"]):
        candidates_count = exploration_json["systems_auto"][system_auto]["selected_count"]
        if exploration_json["systems_auto"][system_auto]["disturbed_candidate_value"] > 0:
            disturbed_candidates_count = candidates_count
        else:
            disturbed_candidates_count = 0
        labeling_count = candidates_count + disturbed_candidates_count
        system_auto_list.append([system_auto, labeling_count])
        total_to_label += labeling_count

    labeling_json["total_to_label"] = total_to_label
    # Second loop to create the jobs
    for system_auto_index, system_auto in enumerate(exploration_json["systems_auto"]):

        arcann_logger.info(f"Processing system: {system_auto} ({system_auto_index + 1}/{len(exploration_json['systems_auto'])})")

        labeling_json["systems_auto"][system_auto] = {}
        candidates_count = exploration_json["systems_auto"][system_auto]["selected_count"]

        if exploration_json["systems_auto"][system_auto]["disturbed_candidate_value"] > 0:
            disturbed_candidates_count = candidates_count
        else:
            disturbed_candidates_count = 0

        labeling_count = candidates_count + disturbed_candidates_count
        arcann_logger.debug(f"candidates_count, disturbed_candidates_count, labeling_count: {candidates_count}, {disturbed_candidates_count}, {labeling_count}")

        (
            system_labeling_program,
            system_walltime_first_job_h,
            system_walltime_second_job_h,
            system_nb_nodes,
            system_nb_mpi_per_node,
            system_nb_threads_per_mpi,
        ) = get_system_labeling(current_input_json, system_auto_index)

        arcann_logger.debug(f"{system_labeling_program, system_walltime_first_job_h,system_walltime_second_job_h,system_nb_nodes,system_nb_mpi_per_node,system_nb_threads_per_mpi}")

        if labeling_count == 0:
            labeling_json["systems_auto"][system_auto]["walltime_first_job_h"] = system_walltime_first_job_h
            labeling_json["systems_auto"][system_auto]["walltime_second_job_h"] = system_walltime_second_job_h
            labeling_json["systems_auto"][system_auto]["nb_nodes"] = system_nb_nodes
            labeling_json["systems_auto"][system_auto]["nb_mpi_per_node"] = system_nb_mpi_per_node
            labeling_json["systems_auto"][system_auto]["nb_threads_per_mpi"] = system_nb_threads_per_mpi
            labeling_json["systems_auto"][system_auto]["candidates_count"] = candidates_count
            labeling_json["systems_auto"][system_auto]["disturbed_candidates_count"] = disturbed_candidates_count
            continue

        if curr_iter > 1 and ("walltime_first_job_h" not in user_input_json or user_input_json["walltime_first_job_h"][system_auto_index] == -1):
            system_walltime_first_job_h = max(previous_labeling_json["systems_auto"][system_auto]["timings_s"][0] / 3600 * 1.5, 0.5)
        if curr_iter > 1 and ("walltime_second_job_h" not in user_input_json or user_input_json["walltime_second_job_h"][system_auto_index] == -1):
            system_walltime_second_job_h = max(previous_labeling_json["systems_auto"][system_auto]["timings_s"][1] / 3600 * 1.5, 0.5)

        current_input_json["walltime_first_job_h"][system_auto_index] = system_walltime_first_job_h
        current_input_json["walltime_second_job_h"][system_auto_index] = system_walltime_second_job_h

        system_path = current_path / system_auto
        system_path.mkdir(exist_ok=True)

        # Replace slurm
        walltime_approx_s = int((system_walltime_first_job_h + system_walltime_second_job_h) * 3600)

        system_master_job_file = deepcopy(master_job_file)
        for _ in system_master_job_file:
            system_master_job_file[_] = replace_substring_in_string_list(system_master_job_file[_], "_R_nb_NODES_", f"{system_nb_nodes}")
            system_master_job_file[_] = replace_substring_in_string_list(system_master_job_file[_], "_R_nb_MPI_", f"{system_nb_nodes * system_nb_mpi_per_node}")
            system_master_job_file[_] = replace_substring_in_string_list(system_master_job_file[_], "_R_nb_MPIPERNODE_", f"{system_nb_mpi_per_node}")
            system_master_job_file[_] = replace_substring_in_string_list(system_master_job_file[_], "_R_nb_THREADSPERMPI_", f"{system_nb_threads_per_mpi}")
            system_master_job_file[_] = replace_in_slurm_file_general(system_master_job_file[_], machine_spec, walltime_approx_s, machine_walltime_format, current_input_json["job_email"])

        if system_auto_list[system_auto_index][0] != system_auto:
            arcann_logger.error(f"System auto list and system auto index do not match. PLEASE REPORT THIS BUG.")
            arcann_logger.error(f"Aborting...")
            return

        system_master_job_file[0] = replace_substring_in_string_list(system_master_job_file[0], f"_R_{labeling_program_up}_JOBNAME_", f"{labeling_program_up}_{system_auto}_{padded_curr_iter}")

        # Find the index of the next system with non-zero selected_count
        next_index = system_auto_index + 1
        while next_index < len(system_auto_list):
            if system_auto_list[next_index][1] != 0:
                break
            next_index += 1
        else:
            next_index = -1

        # Calculate the effective capacity increment based on machine_max_array_size and machine_max_jobs
        if machine_max_jobs <= 0:
            effective_capacity_increment = 0
        else:
            effective_capacity_increment = (machine_max_array_size // machine_max_jobs) * machine_max_jobs

        jobs_processed = 0
        block_start = 0
        batch_number = 0

        # List to store the details of each batch for verification
        slurm_file_array_subsys_dict = {}

        while jobs_processed < labeling_count:
            # Calculate start and end indices for jobs within the current batch
            if machine_max_jobs <= 0:
                batch_size = labeling_count
                batch_start = 0
                batch_end = labeling_count - 1
            else:
                batch_size = min(machine_max_jobs, labeling_count - jobs_processed)
                batch_start = (batch_number % (machine_max_array_size // machine_max_jobs)) * machine_max_jobs
                batch_end = batch_start + batch_size - 1

            # Replace placeholders in the system_master_job_file with batch-specific values
            slurm_file_array_subsys_dict[batch_number] = replace_substring_in_string_list(system_master_job_file[0], "_R_NEW_START_", f"{block_start}")
            slurm_file_array_subsys_dict[batch_number] = replace_substring_in_string_list(slurm_file_array_subsys_dict[batch_number], "_R_ARRAY_START_", f"{batch_start}")
            slurm_file_array_subsys_dict[batch_number] = replace_substring_in_string_list(slurm_file_array_subsys_dict[batch_number], "_R_ARRAY_END_", f"{batch_end}")

            if jobs_processed + batch_size == labeling_count:
                if machine_max_jobs <= 0 or next_index == -1 or total_to_label <= machine_max_jobs:
                    slurm_file_array_subsys_dict[batch_number] = replace_substring_in_string_list(slurm_file_array_subsys_dict[batch_number], "_R_LAUNCHNEXT_", "0")
                else:
                    slurm_file_array_subsys_dict[batch_number] = replace_substring_in_string_list(slurm_file_array_subsys_dict[batch_number], "_R_LAUNCHNEXT_", "1")
                    slurm_file_array_subsys_dict[batch_number] = replace_substring_in_string_list(slurm_file_array_subsys_dict[batch_number], "_R_NEXT_JOB_FILE_", "0")
                    slurm_file_array_subsys_dict[batch_number] = replace_substring_in_string_list(slurm_file_array_subsys_dict[batch_number], "_R_CD_WHERE_", "${SLURM_SUBMIT_DIR}/../" + system_auto_list[next_index][0])
            else:
                slurm_file_array_subsys_dict[batch_number] = replace_substring_in_string_list(slurm_file_array_subsys_dict[batch_number], "_R_LAUNCHNEXT_", "1")
                slurm_file_array_subsys_dict[batch_number] = replace_substring_in_string_list(slurm_file_array_subsys_dict[batch_number], "_R_NEXT_JOB_FILE_", f"{batch_number + 1}")
                slurm_file_array_subsys_dict[batch_number] = replace_substring_in_string_list(slurm_file_array_subsys_dict[batch_number], "_R_CD_WHERE_", "${SLURM_SUBMIT_DIR}")

            # Save the batch-specific slurm file
            string_list_to_textfile(system_path / f"job-array_{labeling_program_up}_label_{machine_spec['arch_type']}_{machine}_{batch_number}.sh", slurm_file_array_subsys_dict[batch_number])

            jobs_processed += batch_size
            batch_number += 1

            if batch_number % (machine_max_array_size // machine_max_jobs) == 0:
                block_start += effective_capacity_increment

        # Labeling input first job
        system_first_job_input = textfile_to_string_list(training_path / "user_files" / f"1_{system_auto}_labeling_XXXXX_{machine}.inp")
        system_first_job_input = replace_substring_in_string_list(system_first_job_input, "_R_WALLTIME_", f"{system_walltime_first_job_h * 3600}")
        system_first_job_input = replace_substring_in_string_list(system_first_job_input, "_R_NB_MPI_", f"{system_nb_nodes * system_nb_mpi_per_node}")

        # Labeling input second job
        if labeling_program == "cp2k":
            system_second_job_input = textfile_to_string_list(training_path / "user_files" / f"2_{system_auto}_labeling_XXXXX_{machine}.inp")
            system_second_job_input = replace_substring_in_string_list(system_second_job_input, "_R_WALLTIME_", f"{system_walltime_second_job_h * 3600}")
            system_second_job_input = replace_substring_in_string_list(system_second_job_input, "_R_NB_MPI_", f"{system_nb_nodes * system_nb_mpi_per_node}")

        # Regular
        xyz_file = training_path / f"{padded_curr_iter}-exploration" / system_auto / f"candidates_{padded_curr_iter}_{system_auto}.xyz"
        num_atoms, atom_symbols, atom_coords, comments, cell_info, pbc_info, properties_info, max_f_std_info = parse_xyz_trajectory_file(xyz_file)
        del xyz_file

        if atom_coords.shape[0] != candidates_count:
            arcann_logger.error(f"The number of structures in the xyz does not match the number of candidates.")
            arcann_logger.error(f"Aborting...")
            return 1

        arcann_logger.info(f"Processing {candidates_count} structures for system: {system_auto}.")

        for labeling_step in range(atom_coords.shape[0]):
            padded_labeling_step = str(labeling_step).zfill(5)
            labeling_step_path = system_path / padded_labeling_step
            labeling_step_path.mkdir(exist_ok=True)

            first_job_input_t = deepcopy(system_first_job_input)
            first_job_input_t = replace_substring_in_string_list(first_job_input_t, "_R_PADDEDSTEP_", padded_labeling_step)
            if labeling_program == "cp2k":
                first_job_input_t = replace_substring_in_string_list(first_job_input_t, "_R_CELL_", " ".join([str(_) for _ in [cell_info[0][i] for i in [0, 4, 8]]]))

            string_list_to_textfile(labeling_step_path / f"1_labeling_{padded_labeling_step}.inp", first_job_input_t)
            del first_job_input_t

            if labeling_program == "cp2k":
                second_job_input_t = deepcopy(system_second_job_input)
                second_job_input_t = replace_substring_in_string_list(second_job_input_t, "_R_PADDEDSTEP_", padded_labeling_step)
                second_job_input_t = replace_substring_in_string_list(second_job_input_t, "_R_CELL_", " ".join([str(_) for _ in [cell_info[0][i] for i in [0, 4, 8]]]))

                string_list_to_textfile(labeling_step_path / f"2_labeling_{padded_labeling_step}.inp", second_job_input_t)
                del second_job_input_t

            job_file_t = deepcopy(system_master_job_file[1])
            job_file_t = replace_substring_in_string_list(job_file_t, "_R_PADDEDSTEP_", padded_labeling_step)
            job_file_t = replace_substring_in_string_list(job_file_t, f"_R_{labeling_program_up}_JOBNAME_", f"{labeling_program_up}_{system_auto}_{padded_curr_iter}")
            string_list_to_textfile(labeling_step_path / f"job_{labeling_program_up}_label_{padded_labeling_step}_{machine_spec['arch_type']}_{machine}.sh", job_file_t)
            del job_file_t
            if np.any(cell_info) == None:
                cell_info = np.array([])

            write_xyz_frame(labeling_step_path / f"labeling_{padded_labeling_step}.xyz", labeling_step, num_atoms, atom_symbols, atom_coords, cell_info, comments)
            job_array_params_line = f":{system_auto}:"
            job_array_params_line += f"{padded_labeling_step}:"
            job_array_params_line += f"1_labeling_{padded_labeling_step}:"
            job_array_params_line += f"2_labeling_{padded_labeling_step}:"
            job_array_params_line += f"labeling_{padded_labeling_step}-SCF.wfn:"
            job_array_params_line += f"labeling_{padded_labeling_step}.xyz:"
            job_array_params_line += f"{system_nb_nodes}:"
            job_array_params_line += f"{system_nb_mpi_per_node}:"
            job_array_params_line += f"{system_nb_threads_per_mpi}:"
            job_array_params_line += f"{walltime_approx_s}:"
            job_array_params_file[f"{labeling_program}"].append(job_array_params_line)
            del padded_labeling_step, labeling_step_path

        del labeling_step
        del num_atoms, atom_symbols, atom_coords, cell_info, comments, pbc_info, properties_info, max_f_std_info

        # Disturbed
        xyz_file_disturbed = training_path / f"{padded_curr_iter}-exploration" / system_auto / f"candidates_{padded_curr_iter}_{system_auto}_disturbed.xyz"
        if xyz_file_disturbed.is_file():
            num_atoms, atom_symbols, atom_coords, comments, cell_info, pbc_info, properties_info, max_f_std_info = parse_xyz_trajectory_file(xyz_file)
            del xyz_file_disturbed

            if atom_coords.shape[0] != candidates_count:
                arcann_logger.error(f"The number of structures in the xyz does not match the number of candidates.")
                arcann_logger.error(f"Aborting...")
                return 1

            for labeling_step_idx, labeling_step in enumerate(range(candidates_count, candidates_count + atom_coords.shape[0])):
                padded_labeling_step = str(labeling_step).zfill(5)
                labeling_step_path = system_path / padded_labeling_step
                labeling_step_path.mkdir(exist_ok=True)

                first_job_input_t = deepcopy(system_first_job_input)
                first_job_input_t = replace_substring_in_string_list(first_job_input_t, "_R_PADDEDSTEP_", padded_labeling_step)
                first_job_input_t = replace_substring_in_string_list(first_job_input_t, "_R_CELL_", " ".join([str(_) for _ in [cell_info[0][i] for i in [0, 4, 8]]]))
                string_list_to_textfile(labeling_step_path / f"1_labeling_{padded_labeling_step}.inp", first_job_input_t)
                del first_job_input_t
                if labeling_program == "cp2k":
                    second_job_input_t = deepcopy(system_second_job_input)
                    second_job_input_t = replace_substring_in_string_list(second_job_input_t, "_R_PADDEDSTEP_", padded_labeling_step)
                    second_job_input_t = replace_substring_in_string_list(second_job_input_t, "_R_CELL_", " ".join([str(_) for _ in [cell_info[0][i] for i in [0, 4, 8]]]))
                    string_list_to_textfile(labeling_step_path / f"2_labeling_{padded_labeling_step}.inp", second_job_input_t)
                    del second_job_input_t

                job_file_t = deepcopy(system_master_job_file[1])
                job_file_t = replace_substring_in_string_list(job_file_t, "_R_PADDEDSTEP_", padded_labeling_step)
                job_file_t = replace_substring_in_string_list(job_file_t, f"_R_{labeling_program_up}_JOBNAME_", f"{labeling_program_up}_{system_auto}_{padded_curr_iter}")
                string_list_to_textfile(labeling_step_path / f"job_{labeling_program_up}_label_{padded_labeling_step}_{machine_spec['arch_type']}_{machine}.sh", job_file_t)
                del job_file_t

                if np.any(cell_info) == None:
                    cell_info = np.array([])
                write_xyz_frame(labeling_step_path / f"labeling_{padded_labeling_step}.xyz", labeling_step_idx, num_atoms, atom_symbols, atom_coords, cell_info, comments)

                job_array_params_line = f":{system_auto}:"
                job_array_params_line += f"{padded_labeling_step}:"
                job_array_params_line += f"1_labeling_{padded_labeling_step}:"
                job_array_params_line += f"2_labeling_{padded_labeling_step}:"
                job_array_params_line += f"labeling_{padded_labeling_step}-SCF.wfn:"
                job_array_params_line += f"labeling_{padded_labeling_step}.xyz:"
                job_array_params_line += f"{system_nb_nodes}:"
                job_array_params_line += f"{system_nb_mpi_per_node}:"
                job_array_params_line += f"{system_nb_threads_per_mpi}:"
                job_array_params_line += f"{walltime_approx_s}:"
                job_array_params_file[f"{labeling_program}"].append(job_array_params_line)

                del padded_labeling_step, labeling_step_path

            del labeling_step
            del num_atoms, atom_symbols, atom_coords, cell_info, comments, pbc_info, properties_info, max_f_std_info

        # Update labeling JSON
        labeling_json["systems_auto"][system_auto]["walltime_first_job_h"] = system_walltime_first_job_h
        labeling_json["systems_auto"][system_auto]["walltime_second_job_h"] = system_walltime_second_job_h
        labeling_json["systems_auto"][system_auto]["nb_nodes"] = system_nb_nodes
        labeling_json["systems_auto"][system_auto]["nb_mpi_per_node"] = system_nb_mpi_per_node
        labeling_json["systems_auto"][system_auto]["nb_threads_per_mpi"] = system_nb_threads_per_mpi
        labeling_json["systems_auto"][system_auto]["candidates_count"] = candidates_count
        labeling_json["systems_auto"][system_auto]["disturbed_candidates_count"] = disturbed_candidates_count

        # System dependent cleaning
        del system_first_job_input, system_master_job_file
        del (
            system_walltime_first_job_h,
            system_walltime_second_job_h,
            system_nb_nodes,
            system_nb_mpi_per_node,
            system_nb_threads_per_mpi,
        )
        del candidates_count, disturbed_candidates_count, labeling_count

        arcann_logger.info(f"Processed system: {system_auto} ({system_auto_index + 1}/{len(main_json['systems_auto'])})")
    del system_auto_index, system_auto
    arcann_logger.info(f"{total_to_label} structures will be labeled.")
    if (total_to_label <= machine_max_jobs) or (machine_max_jobs <= 0):
        labeling_json = {**labeling_json, "launch_all_jobs": True}
    else:
        labeling_json = {**labeling_json, "launch_all_jobs": False}

    if total_to_label != len(job_array_params_file[f"{labeling_program}"]) - 1:
        arcann_logger.error(f"The number of structures to label does not match the number of jobs.")
        arcann_logger.error(f"Aborting...")
        return 1

    string_list_to_textfile(current_path / f"job-array-params_{labeling_program_up}_label_{machine_spec['arch_type']}_{machine}.lst", job_array_params_file[f"{labeling_program}"])
    job_array_params_file_array = np.genfromtxt(current_path / f"job-array-params_{labeling_program_up}_label_{machine_spec['arch_type']}_{machine}.lst", dtype=str, delimiter=":", skip_header=1)

    # Set booleans in the exploration JSON
    labeling_json = {
        **labeling_json,
        "is_locked": True,
        "is_launched": False,
        "is_checked": False,
        "is_extracted": False,
    }

    # Dump the JSON files (main, exploration and merged input)
    write_json_file(labeling_json, (control_path / f"labeling_{padded_curr_iter}.json"), read_only=True)
    backup_and_overwrite_json_file(current_input_json, (current_path / "used_input.json"), read_only=True)

    # End
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")

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
        current_input_json,
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

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
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
