"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/08/06
"""

# Standard library modules
import logging
import sys
from pathlib import Path
from copy import deepcopy
import random
import subprocess


# Non-standard library imports
import numpy as np
import yaml

# Local imports
from arcann_training.common.check import validate_step_folder, check_atomsk
from arcann_training.exploration.utils import generate_starting_points, create_models_list, update_system_nb_steps_factor, get_system_exploration, generate_input_exploration_json
from arcann_training.common.ipi import get_temperature_from_ipi_xml
from arcann_training.common.json import backup_and_overwrite_json_file, get_key_in_dict, load_default_json_file, load_json_file, write_json_file
from arcann_training.common.list import replace_substring_in_string_list, string_list_to_textfile, textfile_to_string_list
from arcann_training.common.lammps import read_lammps_data
from arcann_training.common.machine import get_machine_keyword, get_machine_spec_for_step
from arcann_training.common.plumed import analyze_plumed_file_for_movres
from arcann_training.common.slurm import replace_in_slurm_file_general
from arcann_training.common.xml import string_list_to_xml, xml_to_string_list, read_xml_file, write_xml_file


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

    # Load the previous exploration and training JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_training_json = load_json_file(control_path / f"training_{padded_prev_iter}.json")
        if prev_iter > 0:
            previous_exploration_json = load_json_file(control_path / f"exploration_{padded_prev_iter}.json")
        else:
            previous_exploration_json = {}
    else:
        previous_training_json = {}
        previous_exploration_json = {}

    arcann_logger.debug(f"previous_training_json: {previous_training_json}")
    arcann_logger.debug(f"previous_exploration_json: {previous_exploration_json}")

    # Check if the atomsk package is installed
    atomsk_bin = check_atomsk(get_key_in_dict("atomsk_path", user_input_json, previous_exploration_json, default_input_json))
    # Update the merged input JSON
    current_input_json["atomsk_path"] = atomsk_bin
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    # Get the machine keyword (Priority: user > previous > default)
    # And update the merged input JSON
    user_machine_keyword = get_machine_keyword(user_input_json, previous_exploration_json, default_input_json, "exp")
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
        "exploration",
        fake_machine,
        user_machine_keyword,
    )
    arch_type = machine_spec["arch_type"]
    arcann_logger.debug(f"machine: {machine}")
    arcann_logger.debug(f"machine_walltime_format: {machine_walltime_format}")
    arcann_logger.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    arcann_logger.debug(f"machine_launch_command: {machine_launch_command}")
    arcann_logger.debug(f"machine_max_jobs: {machine_max_jobs}")
    arcann_logger.debug(f"machine_max_array_size: {machine_max_array_size}")
    arcann_logger.debug(f"user_machine_keyword: {user_machine_keyword}")
    arcann_logger.debug(f"machine_spec: {machine_spec}")

    current_input_json["user_machine_keyword_exp"] = user_machine_keyword
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    if fake_machine is not None:
        arcann_logger.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        arcann_logger.info(f"Machine identified: '{machine}'.")
    del fake_machine

    # Generate/update the merged input JSON
    # Priority: user > previous > default
    current_input_json = generate_input_exploration_json(user_input_json, previous_exploration_json, default_input_json, current_input_json, main_json)
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    # TODO to rewrite (generate_exploration_json ?)
    # Generate the exploration JSON
    exploration_json = {}
    exploration_json = {
        **exploration_json,
        "atomsk_path": atomsk_bin,
        "user_machine_keyword_exp": user_machine_keyword,
        "deepmd_model_version": previous_training_json["deepmd_model_version"],
        "nnp_count": main_json["nnp_count"],
    }

    # Check if the job file exists (for each exploration requested)
    exploration_types = list(set(current_input_json["exploration_type"]))
    master_job_file = {}
    master_job_array_file = {}
    walltime_approx_s = {}
    cell_info_lammps = [
        "variable v_xlo equal xlo",
        "variable v_xhi equal xhi",
        "variable v_ylo equal ylo",
        "variable v_yhi equal yhi",
        "variable v_zlo equal zlo",
        "variable v_zhi equal zhi",
        'fix extra all print _R_PRINT_FREQ_ "${v_xlo} ${v_xhi} ${v_ylo} ${v_yhi} ${v_zlo} ${v_zhi}" file cell.txt',
        "",
    ]

    for exploration_type in exploration_types:
        walltime_approx_s[exploration_type] = []

        job_file_name = f"job_{exploration_type}-deepmd_explore_{arch_type}_{machine}.sh"
        job_array_file_name = f"job-array_{exploration_type}-deepmd_explore_{arch_type}_{machine}.sh"

        if (current_path.parent / "user_files" / job_file_name).is_file():
            master_job_file[exploration_type] = textfile_to_string_list(current_path.parent / "user_files" / job_file_name)
        else:
            arcann_logger.error(f"No JOB file provided for '{current_step.capitalize()} / {current_phase.capitalize()}' for this machine.")
            arcann_logger.error(f"Aborting...")
            return 1

        if (current_path.parent / "user_files" / job_array_file_name).is_file():
            master_job_array_file[exploration_type] = textfile_to_string_list(current_path.parent / "user_files" / job_array_file_name)
        else:
            arcann_logger.warning(f"No ARRAY JOB file provided for '{current_step.capitalize()} / {current_phase.capitalize()}' for this machine.")
            arcann_logger.error(f"Aborting...")
            return 1

        arcann_logger.debug(f"master_job_file: {master_job_file[exploration_type][0:5]}, {master_job_file[exploration_type][-5:-1]}")
        arcann_logger.debug(f"master_job_array_file: {master_job_array_file[exploration_type][0:5]}, {master_job_array_file[exploration_type][-5:-1]}")

        current_input_json["job_email"] = get_key_in_dict("job_email", user_input_json, previous_exploration_json, default_input_json)

        del job_file_name
    del exploration_type

    # Preparation of the exploration
    exploration_json["systems_auto"] = {}
    nb_sim = 0

    job_array_params_file = {}
    job_array_params_file["lammps"] = ["PATH/_R_DEEPMD_VERSION_/_R_MODEL_FILES_/_R_LAMMPS_IN_FILE_/_R_DATA_FILE_/_R_RERUN_FILE_/_R_PLUMED_FILES_/"]
    job_array_params_file["i-PI"] = ["PATH/_R_DEEPMD_VERSION_/_R_MODEL_FILES_/_R_INPUT_FILE_/_R_DATA_FILE_/_R_RERUN_FILE_/_R_PLUMED_FILES_/"]
    job_array_params_file["sander_emle"] = ["PATH/_R_DEEPMD_VERSION_/_R_MODEL_FILES_/_R_SANDER_IN_FILE_/_R_TOP_FILE_/_R_COORD_FILE_/_R_EMLE_YAML_FILE_/_R_EMLE_MODEL_FILE_//_R_PLUMED_FILES_/"]

    # Loop through each system and set its exploration
    for system_auto_index, system_auto in enumerate(main_json["systems_auto"]):
        random.seed()
        exploration_json["systems_auto"][system_auto] = {}

        # Set the individual system params for exploration
        (
            system_exploration_type,
            system_traj_count,
            system_timestep_ps,
            system_temperature_K,
            system_exp_time_ps,
            system_max_exp_time_ps,
            system_job_walltime_h,
            system_print_mult,
            system_previous_start,
            system_disturbed_start,
        ) = get_system_exploration(current_input_json, system_auto_index)
        arcann_logger.debug(f"{system_exploration_type, system_traj_count, system_timestep_ps,system_temperature_K,system_exp_time_ps,system_max_exp_time_ps,system_job_walltime_h,system_print_mult,system_previous_start,system_disturbed_start}")

        plumed = [False, False, False]

        input_replace_dict = {}

        # Get the input file (.in or .xml) for the current system and check if plumed is being used
        # LAMMPS
        if system_exploration_type == "lammps":
            master_system_lammps_in = textfile_to_string_list(training_path / "user_files" / (system_auto + ".in"))

            # Add cell info to the LAMMPS input file
            index_run = next((i for i, item in enumerate(master_system_lammps_in) if item.startswith("run _R_NUMBER_OF_STEPS_")), -1)
            if index_run == -1:
                arcann_logger.error(f"No 'run _R_NUMBER_OF_STEPS_' found in the LAMMPS input file: '{training_path / 'user_files' / (system_auto + '.in')}'.")
                arcann_logger.error(f"Aborting...")
                return 1
            master_system_lammps_in = master_system_lammps_in[:index_run] + cell_info_lammps + master_system_lammps_in[index_run:]
            del index_run

            # Check if the LAMMPS input file contains any "plumed" lines
            if any("plumed" in zzz for zzz in master_system_lammps_in):
                plumed[0] = True
        # END LAMMPS

        # SANDER-EMLE
        elif system_exploration_type == "sander_emle":
            master_system_sander_emle_in = textfile_to_string_list(training_path / "user_files" / (system_auto + ".in"))
            with (training_path / "user_files" / (system_auto + ".yaml")).open() as f:
                master_system_sander_emle_yaml = yaml.load(f, Loader=yaml.FullLoader)
            # Check if the SANDER input ful contains any "plumed" lines
            if any("plumed" in zzz for zzz in master_system_sander_emle_in):
                plumed[0] = True
        # END OF SANDER-EMLE

        # i-PI
        elif system_exploration_type == "i-PI":
            master_system_ipi_xml = read_xml_file(training_path / "user_files" / (system_auto + ".xml"))
            master_system_ipi_xml_aslist = xml_to_string_list(master_system_ipi_xml)
            # Create a JSON object with placeholders for the dp-i-PI input file parameters
            master_system_ipi_json = {
                "verbose": False,
                "use_unix": False,
                "port": "_R_NB_PORT_",
                "host": "_R_ADDRESS_",
                "graph_file": "_R_GRAPH_",
                "coord_file": "_R_XYZ_",
                "atom_type": {},
            }
            # Check if the XML input file contains any "plumed" lines
            if any("plumed" in zzz for zzz in master_system_ipi_xml_aslist):
                plumed[0] = True
        # END OF i-PI

        # If plumed is being used for the current system, get the plumed input files
        if plumed[0] == 1:
            # Find all plumed files associated with the current system
            plumed_files = [plumed_file for plumed_file in (training_path / "user_files").glob(f"plumed*_{system_auto}.dat")]

            # If no plumed files are found, print an error message and exit
            if len(plumed_files) == 0:
                arcann_logger.error(f"Plumed in (LAMMPS/i-PI/SANDER-EMLE) input but no plumed files found")
                arcann_logger.error(f"Aborting...")
                sys.exit(1)

            # Read the contents of each plumed file into a dictionary
            plumed_input = {}
            for plumed_file in plumed_files:
                plumed_input[plumed_file.name] = textfile_to_string_list(plumed_file)
            # Analyze each plumed file to determine whether it contains MOVINGRESTRAINTS keyword (SMD)
            for plumed_file in plumed_files:
                plumed[1], plumed[2] = analyze_plumed_file_for_movres(plumed_input[plumed_file.name])
                if plumed[1] and plumed[2] != 0:
                    break

        # Generate the starting points (if iteration number > 1)
        # Check the iteration number
        if curr_iter == 1:
            # First iteration, so no disturbed starting points
            system_disturbed_start = False
        else:
            # Get starting points
            (
                starting_point_list,
                starting_point_list_bckp,
                system_previous_start,
                system_disturbed_start,
            ) = generate_starting_points(
                system_exploration_type,
                system_auto,
                training_path,
                padded_prev_iter,
                previous_exploration_json,
                user_input_json_present,
                system_previous_start,
                system_disturbed_start,
            )
            arcann_logger.debug(f"starting_point_list: {starting_point_list}")
            arcann_logger.debug(f"starting_point_list_bckp: {starting_point_list_bckp}")
            arcann_logger.debug(f"system_previous_start: {system_previous_start}")
            arcann_logger.debug(f"system_disturbed_start: {system_disturbed_start}")

            if not starting_point_list:
                arcann_logger.error(f"No starting points found for '{system_auto}'.")
                arcann_logger.error(f"Aborting...")
                return 1

        input_replace_dict["_R_TIMESTEP_"] = f"{system_timestep_ps}"

        # LAMMPS Input
        if system_exploration_type == "lammps":
            input_replace_dict["_R_TEMPERATURE_"] = f"{system_temperature_K}"

            # First exploration
            if curr_iter == 1:
                if "job_walltime_h" in user_input_json and system_job_walltime_h != -1:
                    system_job_walltime_h = system_job_walltime_h
                else:
                    # Default value
                    system_job_walltime_h = 1.0
                if "exp_time_ps" in user_input_json and system_exp_time_ps != -1:
                    system_exp_time_ps = system_exp_time_ps
                else:
                    # Default value
                    system_exp_time_ps = 10.0

                system_lammps_data_fn = system_auto + ".lmp"
                system_lammps_data = textfile_to_string_list(training_path / "user_files" / system_lammps_data_fn)
                input_replace_dict["_R_DATA_FILE_"] = system_lammps_data_fn

                # Default time and nb steps
                if plumed[1]:
                    system_nb_steps = plumed[2]
                else:
                    system_nb_steps = system_exp_time_ps / system_timestep_ps
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(system_nb_steps)}"

                system_walltime_approx_s = system_job_walltime_h * 3600

                current_input_json["job_walltime_h"][system_auto_index] = system_walltime_approx_s / 3600
                current_input_json["exp_time_ps"][system_auto_index] = system_nb_steps * system_timestep_ps

                walltime_approx_s[system_exploration_type].append(system_walltime_approx_s)

                # Get the cell and nb of atoms: It can be done now because the starting point is the same by NNP and by traj
                system_nb_atm, num_atom_types, box, masses, coords = read_lammps_data(system_lammps_data)
                system_cell = [box[1] - box[0], box[3] - box[2], box[5] - box[4]]

            # Subsequent ones
            else:
                # SMD wins
                if plumed[1]:
                    system_nb_steps = plumed[2]
                # User inputs
                elif "exp_time_ps" in user_input_json and system_exp_time_ps != -1:
                    system_nb_steps = system_exp_time_ps / system_timestep_ps
                # Auto value
                else:
                    system_nb_steps = update_system_nb_steps_factor(previous_exploration_json, system_auto) / system_timestep_ps
                    # Update if over Max value
                    if system_nb_steps > system_max_exp_time_ps / system_timestep_ps:
                        system_nb_steps = system_max_exp_time_ps / system_timestep_ps
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(system_nb_steps)}"

                current_input_json["exp_time_ps"][system_auto_index] = system_nb_steps * system_timestep_ps

                # Walltime
                if "job_walltime_h" in user_input_json:
                    system_walltime_approx_s = int(system_job_walltime_h * 3600)
                else:
                    # Abritary factor
                    system_walltime_approx_s = max(int((previous_exploration_json["systems_auto"][system_auto]["mean_s_per_step"] * system_nb_steps) * 1.50), 1800)
                    # Round up to the next 30min
                    system_walltime_approx_s = int(np.ceil(system_walltime_approx_s / 1800) * 1800)

                current_input_json["job_walltime_h"][system_auto_index] = system_walltime_approx_s / 3600
                walltime_approx_s[system_exploration_type].append(system_walltime_approx_s)

            # Get print freq
            system_print_every_x_steps = system_nb_steps * system_print_mult
            if int(system_print_every_x_steps) < 1:
                arcann_logger.warning(f"Print frequency is less than 1 step. Setting it to 1 step.")
                system_print_every_x_steps = 1
            input_replace_dict["_R_PRINT_FREQ_"] = f"{int(system_print_every_x_steps)}"

        # END OF LAMMPS

        # SANDER-EMLE
        elif system_exploration_type == "sander_emle":

            input_replace_dict["_R_TEMPERATURE_"] = f"{system_temperature_K}"

            # First exploration
            if curr_iter == 1:
                if "job_walltime_h" in user_input_json and system_job_walltime_h != -1:
                    system_job_walltime_h = system_job_walltime_h
                else:
                    # Default value
                    system_job_walltime_h = 1.0
                if "exp_time_ps" in user_input_json and system_exp_time_ps != -1:
                    system_exp_time_ps = system_exp_time_ps
                else:
                    # Default value
                    system_exp_time_ps = 10.0

                system_sander_emle_data_fn = system_auto + ".ncrst"
                input_replace_dict["_R_COORD_FILE_"] = system_sander_emle_data_fn
                system_sander_emle_data_path = training_path / "user_files" / system_sander_emle_data_fn

                # Default time and nb steps
                if plumed[1]:
                    system_nb_steps = plumed[2]
                else:
                    system_nb_steps = system_exp_time_ps / system_timestep_ps
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(system_nb_steps)}"

                system_walltime_approx_s = system_job_walltime_h * 3600

                current_input_json["job_walltime_h"][system_auto_index] = system_walltime_approx_s / 3600
                current_input_json["exp_time_ps"][system_auto_index] = system_nb_steps * system_timestep_ps

                walltime_approx_s[system_exploration_type].append(system_walltime_approx_s)

                # Get the cell and nb of atoms: It can be done now because the starting point is the same by NNP and by traj
                system_nb_atm, num_atom_types, box, masses, coords = read_lammps_data(training_path / "user_files" / (system_auto + ".lmp"))
                system_cell = [box[1] - box[0], box[3] - box[2], box[5] - box[4]]

            # Subsequent ones
            else:
                # SMD wins
                if plumed[1]:
                    system_nb_steps = plumed[2]
                # User inputs
                elif "exp_time_ps" in user_input_json and system_exp_time_ps != -1:
                    system_nb_steps = system_exp_time_ps / system_timestep_ps
                # Auto value
                else:
                    system_nb_steps = update_system_nb_steps_factor(previous_exploration_json, system_auto) / system_timestep_ps
                    # Update if over Max value
                    if system_nb_steps > system_max_exp_time_ps / system_timestep_ps:
                        system_nb_steps = system_max_exp_time_ps / system_timestep_ps
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(system_nb_steps)}"

                current_input_json["exp_time_ps"][system_auto_index] = system_nb_steps * system_timestep_ps

                # Walltime
                if "job_walltime_h" in user_input_json:
                    system_walltime_approx_s = int(system_job_walltime_h * 3600)
                else:
                    # Abritary factor
                    system_walltime_approx_s = max(int((previous_exploration_json["systems_auto"][system_auto]["mean_s_per_step"] * system_nb_steps) * 1.50), 1800)
                    # Round up to the next 30min
                    system_walltime_approx_s = int(np.ceil(system_walltime_approx_s / 1800) * 1800)

                current_input_json["job_walltime_h"][system_auto_index] = system_walltime_approx_s / 3600
                walltime_approx_s[system_exploration_type].append(system_walltime_approx_s)

            # Get print freq
            system_print_every_x_steps = system_nb_steps * system_print_mult
            if int(system_print_every_x_steps) < 1:
                arcann_logger.warning(f"Print frequency is less than 1 step. Setting it to 1 step.")
                system_print_every_x_steps = 1
            input_replace_dict["_R_PRINT_FREQ_"] = f"{int(system_print_every_x_steps)}"
        # END OF SANDER-EMLE

        # i-PI // UNTESTED
        elif system_exploration_type == "i-PI":
            system_temperature_K = float(get_temperature_from_ipi_xml(master_system_ipi_xml))
            if system_temperature_K == -1:
                arcann_logger.error(f"No temperature found in the xml: '{training_path / 'files' / (system_auto + '.xml')}'.")
                arcann_logger.error("Aborting...")
                sys.exit(1)

            # TODO This should be read like temp
            input_replace_dict["_R_NB_BEADS_"] = str(8)

            # First exploration
            if curr_iter == 1:
                system_lammps_data_fn = system_auto + ".lmp"
                system_lammps_data = textfile_to_string_list(training_path / "user_files" / system_lammps_data_fn)
                system_ipi_xyz_fn = system_auto + ".xyz"
                input_replace_dict["_R_DATA_FILE_"] = system_ipi_xyz_fn
                master_system_ipi_json["coord_file"] = system_ipi_xyz_fn
                # Get the XYZ file from LMP
                subprocess.run([atomsk_bin, str(Path("../") / "user_files" / system_lammps_data_fn), "xyz", str(Path("../") / "user_files" / system_auto), "-ow"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                system_ipi_xyz = textfile_to_string_list(training_path / "user_files" / system_ipi_xyz_fn)

                if "job_walltime_h" in user_input_json and system_job_walltime_h != -1:
                    system_job_walltime_h = system_job_walltime_h
                else:
                    # Default value
                    system_job_walltime_h = 1.0
                if "exp_time_ps" in user_input_json and system_exp_time_ps != -1:
                    system_exp_time_ps = system_exp_time_ps
                else:
                    # Default value
                    system_exp_time_ps = 10.0

                # Default time and nb steps
                if plumed[1]:
                    system_nb_steps = plumed[2]
                else:
                    system_nb_steps = system_exp_time_ps / system_timestep_ps
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(system_nb_steps)}"

                system_walltime_approx_s = system_job_walltime_h * 3600

                current_input_json["job_walltime_h"][system_auto_index] = system_walltime_approx_s / 3600
                walltime_approx_s[system_exploration_type].append(system_walltime_approx_s)

                # Get the cell and nb of atoms: It can be done now because the starting point is the same by NNP and by traj
                system_nb_atm, num_atom_types, box, masses, coords = read_lammps_data(system_lammps_data)
                system_cell = [box[1] - box[0], box[3] - box[2], box[5] - box[4]]
                input_replace_dict["_R_CELL_"] = f"{system_cell}"
                # Get the type_map from config (key added after first training)
                for it_zzz, zzz in enumerate(main_json["type_map"]):
                    master_system_ipi_json["atom_type"][str(zzz)] = it_zzz

            # Subsequent ones
            else:
                # SMD wins
                if plumed[1]:
                    system_nb_steps = plumed[2]
                # User inputs
                elif "system_exp_time_ps" in user_input_json:
                    system_nb_steps = system_exp_time_ps / system_timestep_ps
                # Auto value
                else:
                    system_nb_steps *= update_system_nb_steps_factor(previous_exploration_json, system_auto) / system_timestep_ps
                    # Update if over Max value
                    if system_nb_steps > (system_max_exp_time_ps / system_timestep_ps):
                        system_nb_steps = system_max_exp_time_ps / system_timestep_ps
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(system_nb_steps)}"
                # Update the new input
                current_input_json["system_exp_time_ps"] = system_nb_steps * system_timestep_ps

                # Walltime
                if user_input_json_present:
                    system_walltime_approx_s = int(system_job_walltime_h * 3600)
                else:
                    # Abritary factor
                    system_walltime_approx_s = max(int((previous_exploration_json["systems_auto"][system_auto]["mean_s_per_step"] * system_nb_steps) * 1.50), 1800)
                    # Round up to the next 30min
                    system_walltime_approx_s = int(np.ceil(system_walltime_approx_s / 1800) * 1800)

                current_input_json["job_walltime_h"] = system_walltime_approx_s / 3600
                walltime_approx_s[system_exploration_type].append(system_walltime_approx_s)

            # Get print freq
            system_print_every_x_steps = system_nb_steps * system_print_mult
            if int(system_print_every_x_steps) < 1:
                arcann_logger.warning(f"Print frequency is less than 1 step. Setting it to 1 step.")
                system_print_every_x_steps = 1
            input_replace_dict["_R_PRINT_FREQ_"] = f"{int(system_print_every_x_steps)}"
        # END OF i-PI

        # Now it is by NNP and by traj_count
        for nnp_index in range(1, main_json["nnp_count"] + 1):
            for traj_index in range(1, system_traj_count + 1):
                nb_sim += 1

                local_path = Path(".").resolve() / str(system_auto) / str(nnp_index) / (str(traj_index).zfill(5))
                local_path.mkdir(exist_ok=True, parents=True)

                # Create the model list
                models_list, models_string = create_models_list(main_json, previous_training_json, nnp_index, padded_prev_iter, training_path, local_path)
                arcann_logger.debug(f"{models_list}, {models_string}")

                # LAMMPS
                if system_exploration_type == "lammps":
                    system_lammps_in = deepcopy(master_system_lammps_in)
                    input_replace_dict["_R_SEED_VEL_"] = f"{nnp_index}{random.randrange(0, 1000)}{traj_index}{padded_curr_iter}"
                    input_replace_dict["_R_SEED_THER_"] = f"{nnp_index}{random.randrange(0, 1000)}{traj_index}{padded_curr_iter}"
                    input_replace_dict["_R_DCD_OUT_"] = f"{system_auto}_{nnp_index}_{padded_curr_iter}.dcd"
                    input_replace_dict["_R_RESTART_OUT_"] = f"{system_auto}_{nnp_index}_{padded_curr_iter}.restart"
                    input_replace_dict["_R_MODEL_FILES_"] = models_string
                    input_replace_dict["_R_DEVI_OUT_"] = f"model_devi_{system_auto}_{nnp_index}_{padded_curr_iter}.out"
                    # Get data files (starting points) if iteration is > 1
                    if curr_iter > 1:
                        if len(starting_point_list) == 0:
                            starting_point_list = deepcopy(starting_point_list_bckp)
                        system_lammps_data_fn = starting_point_list[random.randrange(0, len(starting_point_list))]
                        starting_point_list.remove(system_lammps_data_fn)
                        # Check if the file is in the starting_structures or user_files
                        if (training_path / "starting_structures" / system_lammps_data_fn).is_file():
                            system_lammps_data = textfile_to_string_list(training_path / "starting_structures" / system_lammps_data_fn)
                        elif (training_path / "user_files" / system_lammps_data_fn).is_file():
                            system_lammps_data = textfile_to_string_list(training_path / "user_files" / system_lammps_data_fn)
                        else:
                            arcann_logger.error(f"Starting point '{system_lammps_data_fn}' not found.")
                            arcann_logger.error(f"Aborting...")
                            return 1
                        input_replace_dict["_R_DATA_FILE_"] = system_lammps_data_fn
                        # Get again the system_cell and nb_atom
                        system_nb_atm, num_atom_types, box, masses, coords = read_lammps_data(system_lammps_data)
                        system_cell = [box[1] - box[0], box[3] - box[2], box[5] - box[4]]

                    # Plumed files
                    if plumed[0]:
                        input_replace_dict["_R_PLUMED_IN_"] = f"plumed_{system_auto}.dat"
                        input_replace_dict["_R_PLUMED_OUT_"] = f"plumed_{system_auto}_{nnp_index}_{padded_curr_iter}.log"
                        for it_plumed_input in plumed_input:
                            plumed_input[it_plumed_input] = replace_substring_in_string_list(plumed_input[it_plumed_input], "_R_PRINT_FREQ_", f"{int(system_print_every_x_steps)}")
                            string_list_to_textfile(local_path / it_plumed_input, plumed_input[it_plumed_input], read_only=True)

                    # Write DATA file
                    string_list_to_textfile(local_path / system_lammps_data_fn, system_lammps_data, read_only=True)

                    exploration_json["systems_auto"][system_auto]["nb_steps"] = int(system_nb_steps)
                    exploration_json["systems_auto"][system_auto]["print_every_x_steps"] = int(system_print_every_x_steps)

                    #  Write INPUT file
                    for key, value in input_replace_dict.items():
                        system_lammps_in = replace_substring_in_string_list(system_lammps_in, key, value)
                    del key, value
                    string_list_to_textfile(local_path / f"{system_auto}_{nnp_index}_{padded_curr_iter}.in", system_lammps_in, read_only=True)

                    job_array_params_line = str(system_auto) + "_" + str(nnp_index) + "_" + str(traj_index).zfill(5) + "/"
                    job_array_params_line += f"{exploration_json['deepmd_model_version']}" + "/"
                    job_array_params_line += str(models_string.replace(" ", '" "')) + "/"
                    job_array_params_line += f"{system_auto}_{nnp_index}_{padded_curr_iter}.in" + "/"
                    job_array_params_line += f"{system_lammps_data_fn}" + "/"
                    job_array_params_line += "" + "/"

                    # INDIVIDUAL JOB FILE
                    job_file = replace_in_slurm_file_general(master_job_file[system_exploration_type], machine_spec, system_walltime_approx_s, machine_walltime_format, current_input_json["job_email"])
                    # Replace the inputs/variables in the job file
                    job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_VERSION_", f"{exploration_json['deepmd_model_version']}")
                    job_file = replace_substring_in_string_list(job_file, "_R_MODEL_FILES_", str(models_string.replace(" ", '" "')))
                    job_file = replace_substring_in_string_list(job_file, "_R_LAMMPS_IN_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.in")
                    job_file = replace_substring_in_string_list(job_file, "_R_LAMMPS_LOG_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.log")
                    job_file = replace_substring_in_string_list(job_file, "_R_LAMMPS_OUT_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.out")
                    job_file = replace_substring_in_string_list(job_file, "_R_DATA_FILE_", f"{system_lammps_data_fn}")
                    job_file = replace_substring_in_string_list(job_file, ' "_R_RERUN_FILE_"', "")
                    if plumed[0] == 1:
                        for n, it_plumed_input in enumerate(plumed_input):
                            if n == 0:
                                job_file = replace_substring_in_string_list(job_file, "_R_PLUMED_FILES_", it_plumed_input)
                                job_array_params_line += it_plumed_input
                            else:
                                job_file = replace_substring_in_string_list(job_file, prev_plumed, prev_plumed + '" "' + it_plumed_input)
                                job_array_params_line = job_array_params_line.replace(prev_plumed, prev_plumed + '" "' + it_plumed_input)
                            prev_plumed = it_plumed_input
                        del n, it_plumed_input, prev_plumed
                    else:
                        job_file = replace_substring_in_string_list(job_file, ' "_R_PLUMED_FILES_"', "")
                        job_array_params_line += ""

                    job_array_params_line += "/"
                    job_array_params_file[system_exploration_type].append(job_array_params_line)

                    string_list_to_textfile(local_path / f"job_{system_exploration_type}-deepmd_explore_{arch_type}_{machine}.sh", job_file, read_only=True)

                    del system_lammps_in
                    del job_file
                    # END OF INDIVIDUAL JOB FILE
                # END OF LAMMPS

                # SANDER-EMLE
                elif system_exploration_type == "sander_emle":
                    system_sander_emle_in = deepcopy(master_system_sander_emle_in)
                    system_sander_emle_yaml = deepcopy(master_system_sander_emle_yaml)

                    input_replace_dict["_R_SEED_VEL_"] = f"{nnp_index}{random.randrange(0, 1000)}{traj_index}{padded_curr_iter}"
                    input_replace_dict["_R_NC_OUT_"] = f"{system_auto}_{nnp_index}_{padded_curr_iter}.nc"
                    input_replace_dict["_R_RESTART_OUT_"] = f"{system_auto}_{nnp_index}_{padded_curr_iter}.restrt"
                    input_replace_dict["_R_DEVI_OUT_"] = f"model_devi_{system_auto}_{nnp_index}_{padded_curr_iter}.out"

                    # Get data files (starting points) if iteration is > 1
                    if curr_iter > 1:

                        # TODO: Implement the starting points for SANDER-EMLE
                        arcann_logger.warning(f"Starting points are not implemented for SANDER-EMLE. Using the same starting point as the first exploration.")
                        system_previous_start = False
                        # if len(starting_point_list) == 0:
                        #     starting_point_list = copy.deepcopy(starting_point_list_bckp)
                        # system_sander_emle_data_fn = starting_point_list[random.randrange(0, len(starting_point_list))]
                        # starting_point_list.remove(system_sander_emle_data_fn)
                        # if system_previous_start:
                        #     system_sander_emle_data_path = training_path / "starting_structures" / system_sander_emle_data_fn
                        # else:
                        #     system_sander_emle_data_path = training_path / "user_files" / system_sander_emle_data_fn
                        # system_sander_emle_data_path = training_path / "user_files" / system_sander_emle_data_fn

                        system_sander_emle_data_fn = system_auto + ".ncrst"
                        system_sander_emle_data_path = training_path / "user_files" / system_sander_emle_data_fn
                        input_replace_dict["_R_COORD_FILE_"] = system_sander_emle_data_fn

                        system_nb_atm = main_json["systems_auto"][system_auto]["nb_atm"]
                        system_cell = main_json["systems_auto"][system_auto]["cell"]

                    # Plumed files
                    if plumed[0]:
                        input_replace_dict["_R_PLUMED_IN_"] = f"plumed_{system_auto}.dat"
                        input_replace_dict["_R_PLUMED_OUT_"] = f"plumed_{system_auto}_{nnp_index}_{padded_curr_iter}.log"
                        for it_plumed_input in plumed_input:
                            plumed_input[it_plumed_input] = replace_substring_in_string_list(plumed_input[it_plumed_input], "_R_PRINT_FREQ_", f"{int(system_print_every_x_steps)}")
                            string_list_to_textfile(local_path / it_plumed_input, plumed_input[it_plumed_input], read_only=True)

                    # Write CRD/PRMTOP/EMLE_MODEL file
                    system_emle_model_file = training_path / "user_files" / f"{system_auto}.mat"
                    system_prmtop_file = training_path / "user_files" / f"{system_auto}.prmtop"

                    subprocess.run(["cp", f"{system_emle_model_file}", local_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    subprocess.run(["cp", f"{system_prmtop_file}", local_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    subprocess.run(["cp", f"{system_sander_emle_data_path}", local_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                    # Save the nb of steps and print freq in the exploration JSON
                    exploration_json["systems_auto"][system_auto]["nb_steps"] = int(system_nb_steps)
                    exploration_json["systems_auto"][system_auto]["print_every_x_steps"] = int(system_print_every_x_steps)

                    #  Write INPUT file
                    for key, value in input_replace_dict.items():
                        system_sander_emle_in = replace_substring_in_string_list(system_sander_emle_in, key, value)
                    del key, value
                    string_list_to_textfile(
                        local_path / f"{system_auto}_{nnp_index}_{padded_curr_iter}.in",
                        system_sander_emle_in,
                        read_only=True,
                    )

                    # Write YAML file
                    system_sander_emle_yaml["deepmd_model"] = models_list
                    system_sander_emle_yaml["deepmd_deviation"] = f"model_devi_{system_auto}_{nnp_index}_{padded_curr_iter}.out"
                    system_sander_emle_yaml["energy_file"] = f"{system_auto}_{nnp_index}_{padded_curr_iter}_emle.en"
                    system_sander_emle_yaml["log_file"] = f"{system_auto}_{nnp_index}_{padded_curr_iter}_emle.log"
                    system_sander_emle_yaml["model"] = f"{system_auto}.mat"
                    system_sander_emle_yaml["qm_xyz_file"] = f"{system_auto}_{nnp_index}_{padded_curr_iter}_QM.xyz"
                    system_sander_emle_yaml["qm_xyz_frequency"] = int(system_print_every_x_steps)

                    with (local_path / f"{system_auto}_{nnp_index}_{padded_curr_iter}.yaml").open("w") as f:
                        yaml.dump(system_sander_emle_yaml, f)

                    job_array_params_line = str(system_auto) + "_" + str(nnp_index) + "_" + str(traj_index).zfill(5) + "/"
                    job_array_params_line += f"{exploration_json['deepmd_model_version']}" + "/"
                    job_array_params_line += str(models_string.replace(" ", '" "')) + "/"
                    job_array_params_line += f"{system_auto}_{nnp_index}_{padded_curr_iter}.in" + "/"
                    job_array_params_line += f"{system_auto}.prmtop" + "/"
                    job_array_params_line += f"{system_sander_emle_data_fn}" + "/"
                    job_array_params_line += f"{system_auto}_{nnp_index}_{padded_curr_iter}.yaml" + "/"
                    job_array_params_line += f"{system_auto}.mat" + "/"
                    job_array_params_line += "" + "/"

                    # INDIVIDUAL JOB FILE
                    job_file = replace_in_slurm_file_general(master_job_file[system_exploration_type], machine_spec, system_walltime_approx_s, machine_walltime_format, current_input_json["job_email"])
                    # Replace the inputs/variables in the job file
                    job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_VERSION_", f"{exploration_json['deepmd_model_version']}")
                    job_file = replace_substring_in_string_list(job_file, "_R_MODEL_FILES_", str(models_string.replace(" ", '" "')))
                    job_file = replace_substring_in_string_list(job_file, "_R_SANDER_IN_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.in")
                    job_file = replace_substring_in_string_list(job_file, "_R_EMLE_IN_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.yaml")
                    job_file = replace_substring_in_string_list(job_file, "_R_SANDER_LOG_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.log")
                    job_file = replace_substring_in_string_list(job_file, "_R_SANDER_OUT_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.out")
                    job_file = replace_substring_in_string_list(job_file, "_R_SANDER_RESTART_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.ncrst")
                    job_file = replace_substring_in_string_list(job_file, "_R_EMLE_OUT_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}_emle.out")
                    job_file = replace_substring_in_string_list(job_file, "_R_SANDER_TRAJOUT_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.nc")
                    job_file = replace_substring_in_string_list(job_file, "_R_TOP_FILE_", f"{system_auto}.prmtop")
                    job_file = replace_substring_in_string_list(job_file, "_R_SANDER_COORD_FILE_", f"{system_sander_emle_data_fn}")
                    job_file = replace_substring_in_string_list(job_file, "_R_EMLE_MODEL_FILE_", f"{system_auto}.mat")
                    if plumed[0] == 1:
                        for n, it_plumed_input in enumerate(plumed_input):
                            if n == 0:
                                job_file = replace_substring_in_string_list(job_file, "_R_PLUMED_FILES_", it_plumed_input)
                                job_array_params_line += it_plumed_input
                            else:
                                job_file = replace_substring_in_string_list(job_file, prev_plumed, prev_plumed + '" "' + it_plumed_input)
                                job_array_params_line = job_array_params_line.replace(prev_plumed, prev_plumed + '" "' + it_plumed_input)
                            prev_plumed = it_plumed_input
                        del n, it_plumed_input, prev_plumed
                    else:
                        job_file = replace_substring_in_string_list(job_file, ' "_R_PLUMED_FILES_"', "")
                        job_array_params_line += ""

                    job_array_params_line += "/"
                    job_array_params_file[system_exploration_type].append(job_array_params_line)

                    string_list_to_textfile(local_path / f"job_{system_exploration_type}-deepmd_explore_{arch_type}_{machine}.sh", job_file, read_only=True)

                    # del system_lammps_in
                    del job_file
                    # END OF INDIVIDUAL JOB FILE
                # END OF SANDER-EMLE

                # i-PI
                elif system_exploration_type == "i-PI":
                    system_ipi_json = deepcopy(master_system_ipi_json)
                    system_ipi_xml_aslist = deepcopy(master_system_ipi_xml_aslist)
                    input_replace_dict["_R_SEED_"] = f"{nnp_index}{random.randrange(0, 1000)}{traj_index}{padded_curr_iter}"
                    input_replace_dict["_R_SYS_"] = f"{system_auto}_{nnp_index}_{padded_curr_iter}"
                    # Get data files (starting points) and number of steps
                    if curr_iter > 1:
                        if len(starting_points) == 0:
                            starting_point_list = deepcopy(starting_point_list_bckp)
                        system_ipi_xyz_fn = starting_point_list[random.randrange(0, len(starting_point_list))]
                        starting_point_list.remove(system_lammps_data_fn)

                        system_ipi_xyz = textfile_to_string_list(training_path / "starting_structures" / system_ipi_xyz_fn)
                        input_replace_dict["_R_XYZ_"] = system_ipi_xyz_fn
                        system_ipi_json["coord_file"] = system_ipi_xyz_fn
                        for it_zzz, zzz in enumerate(main_json["type_map"]):
                            system_ipi_json["atom_type"][str(zzz)] = it_zzz
                        system_lammps_data = textfile_to_string_list(training_path / "starting_structures" / system_ipi_xyz_fn.replace(".xyz", ".lmp"))
                        # Get again the system_cell and nb_atom
                        (system_nb_atm, num_atom_types, box, masses, coords) = read_lammps_data(system_lammps_data)
                        system_cell = [box[1] - box[0], box[3] - box[2], box[5] - box[4]]
                        input_replace_dict["_R_CELL_"] = f"{system_cell}"

                    # Plumed files
                    if plumed[0]:
                        input_replace_dict["_R_PLUMED_IN_"] = f"plumed_{system_auto}.dat"
                        for it_plumed_input in plumed_input:
                            plumed_input[it_plumed_input] = replace_substring_in_string_list(
                                plumed_input[it_plumed_input],
                                "_R_PRINT_FREQ_",
                                f"{int(system_print_every_x_steps)}",
                            )
                            # Because of weird units of time
                            plumed_input[it_plumed_input] = replace_substring_in_string_list(plumed_input[it_plumed_input], "UNITS LENGTH", "UNITS TIME=" + str(2.4188843e-05 / system_timestep_ps) + " LENGTH")
                            string_list_to_textfile(local_path / it_plumed_input, plumed_input[it_plumed_input], read_only=True)
                        del it_plumed_input

                    system_ipi_json["graph_file"] = models_list[0]

                    #  Write INPUT files
                    for key, value in input_replace_dict.items():
                        system_ipi_xml_aslist = replace_substring_in_string_list(system_ipi_xml_aslist, key, value)
                    del key, value
                    system_ipi_xml = string_list_to_xml(system_ipi_xml_aslist)
                    write_xml_file(system_ipi_xml, local_path / f"{system_auto}_{nnp_index}_{padded_curr_iter}.xml", read_only=True)
                    write_json_file(system_ipi_json, local_path / f"{system_auto}_{nnp_index}_{padded_curr_iter}.json", read_only=True)

                    # INDIVIDUAL JOB FILE
                    job_file = replace_in_slurm_file_general(master_job_file[system_exploration_type], machine_spec, system_walltime_approx_s, machine_walltime_format, current_input_json["job_email"])
                    # Replace the inputs/variables in the job file
                    job_file = replace_substring_in_string_list(job_file, "_R_DEEPMD_VERSION_", f"{exploration_json['deepmd_model_version']}")
                    job_file = replace_substring_in_string_list(job_file, "_R_MODEL_FILES_", f"{models_list[0]}")
                    job_file = replace_substring_in_string_list(job_file, "_R_IPI_IN_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.xml")
                    job_file = replace_substring_in_string_list(job_file, "_R_DPIPI_IN_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.json")
                    job_file = replace_substring_in_string_list(job_file, "_R_IPI_OUT_FILE_", f"{system_auto}_{nnp_index}_{padded_curr_iter}.out")
                    job_file = replace_substring_in_string_list(job_file, "_R_DATA_FILE_", f"{system_ipi_xyz_fn}")

                    if plumed[0] == 1:
                        for n, it_plumed_input in enumerate(plumed_input):
                            if n == 0:
                                job_file = replace_substring_in_string_list(job_file, "_R_PLUMED_FILES_", it_plumed_input)
                            else:
                                job_file = replace_substring_in_string_list(job_file, prev_plumed, prev_plumed + '" "' + it_plumed_input)
                            prev_plumed = it_plumed_input
                        del n, it_plumed_input, prev_plumed
                    else:
                        job_file = replace_substring_in_string_list(job_file, ' "_R_PLUMED_FILES_"', "")
                    string_list_to_textfile(local_path / f"job_{system_exploration_type}-deepmd_explore_{arch_type}_{machine}.sh", job_file, read_only=True)

                    del system_ipi_xml_aslist, system_ipi_xml, system_ipi_json
                    del job_file
                    # END OF INDIVIDUAL JOB FILE
                else:
                    arcann_logger.error(f"Exploration is unknown/not set.")
                    arcann_logger.error(f"Aborting...")
                    sys.exit(1)

            del traj_index, models_list, models_string, local_path

        del nnp_index

        exploration_json["systems_auto"][system_auto]["nb_atm"] = system_nb_atm
        exploration_json["systems_auto"][system_auto]["exploration_type"] = system_exploration_type
        exploration_json["systems_auto"][system_auto]["traj_count"] = system_traj_count
        exploration_json["systems_auto"][system_auto]["temperature_K"] = system_temperature_K
        exploration_json["systems_auto"][system_auto]["timestep_ps"] = system_timestep_ps
        exploration_json["systems_auto"][system_auto]["previous_start"] = system_previous_start
        exploration_json["systems_auto"][system_auto]["disturbed_start"] = system_disturbed_start
        exploration_json["systems_auto"][system_auto]["print_interval_mult"] = system_print_mult
        exploration_json["systems_auto"][system_auto]["max_exp_time_ps"] = system_max_exp_time_ps

        main_json["systems_auto"][system_auto]["cell"] = system_cell
        main_json["systems_auto"][system_auto]["nb_atm"] = system_nb_atm

        if plumed[0] == 1:
            del plumed_input, plumed, plumed_file, plumed_files
        del input_replace_dict
        if system_exploration_type == "lammps":
            del (
                master_system_lammps_in,
                system_lammps_data_fn,
                num_atom_types,
                box,
                coords,
                masses,
            )
        if curr_iter > 1:
            del starting_point_list, starting_point_list_bckp
        del system_temperature_K, system_cell, system_exp_time_ps, system_nb_atm, system_nb_steps, system_timestep_ps, system_walltime_approx_s, system_exploration_type
        del system_traj_count, system_print_mult, system_previous_start, system_disturbed_start, system_max_exp_time_ps, system_job_walltime_h, system_print_every_x_steps

    del system_auto_index, system_auto, master_job_file

    # Set booleans in the exploration JSON
    exploration_json = {
        **exploration_json,
        "is_locked": True,
        "is_launched": False,
        "is_checked": False,
        "is_deviated": False,
        "is_extracted": False,
    }
    if "i-PI" in exploration_types:
        exploration_json = {
            **exploration_json,
            "is_unbeaded": False,
            "is_reruned": False,
            "is_rechecked": False,
        }

    for exploration_type in exploration_types:
        if len(job_array_params_file[exploration_type]) > 1:
            job_array_file = replace_in_slurm_file_general(master_job_array_file[exploration_type], machine_spec, max(walltime_approx_s[exploration_type]), machine_walltime_format, current_input_json["job_email"])

            job_array_file = replace_substring_in_string_list(job_array_file, "_R_ARRAY_START_", "0")
            job_array_file = replace_substring_in_string_list(job_array_file, "_R_ARRAY_END_", f"{nb_sim - 1}")

            string_list_to_textfile(current_path / f"job-array_{exploration_type}-deepmd_explore_{arch_type}_{machine}.sh", job_array_file, read_only=True)
            string_list_to_textfile(current_path / f"job-array-params_{exploration_type}-deepmd_explore_{arch_type}_{machine}.lst", job_array_params_file[exploration_type], read_only=True)

        del exploration_types

    exploration_json["nb_sim"] = nb_sim

    # Dump the JSON files (main, exploration and merged input)
    arcann_logger.info(f"-" * 88)
    write_json_file(main_json, (control_path / "config.json"), read_only=True)
    write_json_file(exploration_json, (control_path / f"exploration_{padded_curr_iter}.json"), read_only=True)
    backup_and_overwrite_json_file(current_input_json, (current_path / "used_input.json"), read_only=True)

    # End
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")

    # Cleaning
    del current_path, control_path, training_path
    del default_input_json, default_input_json_present, user_input_json, user_input_json_present, user_input_json_filename
    del main_json, current_input_json, exploration_json, previous_training_json, previous_exploration_json
    del user_machine_keyword
    del curr_iter, padded_curr_iter, prev_iter, padded_prev_iter
    del machine, machine_spec, machine_walltime_format, machine_launch_command, machine_job_scheduler

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "prepare",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
