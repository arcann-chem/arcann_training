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
import copy
import logging
import random
import subprocess
import sys
from pathlib import Path

# Local imports
from deepmd_iterative.common.check import validate_step_folder, check_atomsk
from deepmd_iterative.exploration.utils import (
    generate_starting_points,
    create_models_list,
    update_system_nb_steps_factor,
    get_system_exploration,
    set_input_explor_json,
)
from deepmd_iterative.common.filesystem import (
    check_file_existence,
)
from deepmd_iterative.common.ipi import get_temperature_from_ipi_xml
from deepmd_iterative.common.json import (
    backup_and_overwrite_json_file,
    load_default_json_file,
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.json_parameters import (
    get_machine_keyword,
    get_key_in_dict,
)
from deepmd_iterative.common.list import (
    replace_substring_in_string_list,
    string_list_to_textfile,
    textfile_to_string_list,
)
from deepmd_iterative.common.lammps import read_lammps_data
from deepmd_iterative.common.machine import get_machine_spec_for_step
from deepmd_iterative.common.plumed import analyze_plumed_file_for_movres
from deepmd_iterative.common.slurm import replace_in_slurm_file_general
from deepmd_iterative.common.xml import (
    string_list_to_xml,
    xml_to_string_list,
    read_xml_file,
    write_xml_file,
)


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_config_filename: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}"
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

    # Load the default config (JSON)
    default_config = load_default_json_file(
        deepmd_iterative_path / "assets" / "default_config.json"
    )[current_step]
    default_config_present = bool(default_config)
    logging.debug(f"default_config: {default_config}")
    logging.debug(f"default_config_present: {default_config_present}")

    # Load the user config (JSON)
    if (current_path / user_config_filename).is_file():
        user_config = load_json_file((current_path / user_config_filename))
    else:
        user_config = {}
    user_config_present = bool(user_config)
    logging.debug(f"user_config: {user_config}")
    logging.debug(f"user_config_present: {user_config_present}")

    # Make a deepcopy
    current_config = copy.deepcopy(user_config)

    # Get control path and load the main config (JSON)
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))

    # Get extra needed paths
    jobs_path = deepmd_iterative_path / "assets" / "jobs" / current_step

    # Load the previous training config (JSON) and the previous exploration config (JSON)
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        prev_training_config = load_json_file(
            (control_path / f"training_{padded_prev_iter}.json")
        )
        if prev_iter > 0:
            prev_exploration_config = load_json_file(
                (control_path / f"exploration_{padded_prev_iter}.json")
            )
        else:
            prev_exploration_config = {}
    else:
        prev_training_config = {}
        prev_exploration_config = {}

    # Check if the atomsk package is installed
    atomsk_bin = check_atomsk(
        get_key_in_dict(
            "atomsk_path", user_config, prev_exploration_config, default_config
        )
    )
    # Update new input
    current_config["atomsk_path"] = atomsk_bin

    # Get the machine keyword (input override previous training override default_config)
    # And update the new input
    user_machine_keyword = get_machine_keyword(
        user_config, prev_training_config, default_config
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")
    current_config["user_machine_keyword"] = user_machine_keyword
    logging.debug(f"current_config: {current_config}")
    # Set it to None if bool, because: get_machine_spec_for_step needs None
    user_machine_keyword = (
        None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    )
    logging.debug(f"user_machine_keyword: {user_machine_keyword}")

    # From the keyword (or default), get the machine spec (or for the fake one)
    (
        machine,
        machine_spec,
        machine_walltime_format,
        machine_job_scheduler,
        machine_launch_command,
    ) = get_machine_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "exploration",
        fake_machine,
        user_machine_keyword,
    )
    logging.debug(f"machine: {machine}")
    logging.debug(f"machine_spec: {machine_spec}")
    logging.debug(f"machine_walltime_format: {machine_walltime_format}")
    logging.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    logging.debug(f"machine_launch_command: {machine_launch_command}")

    if fake_machine is not None:
        logging.info(f"Pretending to be on: {fake_machine}.")
    else:
        logging.info(f"We are on: {machine}.")
    del fake_machine

    # TODO to rewrite
    # Create the exploration JSON object
    exploration_config = {}

    # Set the exploration parameters in the JSON file
    exploration_config = {
        **exploration_config,
        "deepmd_model_version": prev_training_config["deepmd_model_version"],
        "nnp_count": main_config["nnp_count"],
        "exploration_type": main_config["exploration_type"],
        "traj_count": get_key_in_dict(
            "traj_count", user_config, prev_exploration_config, default_config
        ),
    }
    # Update the new input
    current_config["traj_count"] = exploration_config["traj_count"]
    logging.debug(f"current_config: {current_config}")

    # Fill the missing values from the input. We don't do exploration because it is system dependent and single value and not list
    current_config = set_input_explor_json(
        user_config,
        prev_exploration_config,
        default_config,
        current_config,
        main_config,
    )
    logging.debug(f"current_config: {current_config}")

    # Set additional machine-related parameters in the JSON file
    exploration_config = {
        **exploration_config,
        "machine": machine,
        "project_name": machine_spec["project_name"],
        "allocation_name": machine_spec["allocation_name"],
        "arch_name": machine_spec["arch_name"],
        "arch_type": machine_spec["arch_type"],
        "launch_command": machine_launch_command,
    }

    # Check if the job file exists
    job_file_name = f"job_deepmd_{exploration_config['exploration_type']}_{exploration_config['arch_type']}_{machine}.sh"
    if (current_path.parent / "user_files" / job_file_name).is_file():
        master_job_file = textfile_to_string_list(
            current_path.parent / "user_files" / job_file_name
        )
    else:
        check_file_existence(
            jobs_path / job_file_name,
            error_msg=f"No SLURM file present for {current_step.capitalize()} / {current_phase.capitalize()} on this machine.",
        )
        master_job_file = textfile_to_string_list(
            jobs_path / job_file_name,
        )
    logging.debug(f"master_job_file: {master_job_file[0:5]}, {master_job_file[-5:-1]}")
    current_config["job_email"] = get_key_in_dict(
        "job_email", user_config, prev_exploration_config, default_config
    )
    del jobs_path, job_file_name

    # Preparation of the exploration
    exploration_config["systems_auto"] = {}

    # Loop through each system and set its exploration
    for system_auto_index, system_auto in enumerate(main_config["systems_auto"]):
        random.seed()
        exploration_config["systems_auto"][system_auto] = {}

        plumed = [False, False, False]
        exploration_type = -1

        input_replace_dict = {}

        # Get the input file (.in or .xml) for the current system and check if plumed is being used
        if exploration_config["exploration_type"] == "lammps":
            exploration_type = 0
            master_system_lammps_in = textfile_to_string_list(
                training_path / "user_files" / (system_auto + ".in")
            )
            # Check if the LAMMPS input file contains any "plumed" lines
            if any("plumed" in zzz for zzz in master_system_lammps_in):
                plumed[0] = True
        elif exploration_config["exploration_type"] == "i-PI":
            exploration_type = 1
            master_system_ipi_xml = read_xml_file(
                training_path / "user_files" / (system_auto + ".xml")
            )
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

        # If plumed is being used for the current system, get the plumed input files
        if plumed[0] == 1:
            # Find all plumed files associated with the current system
            plumed_files = [
                plumed_file
                for plumed_file in (training_path / "user_files").glob(
                    f"plumed*_{system_auto}.dat"
                )
            ]
            # If no plumed files are found, print an error message and exit
            if len(plumed_files) == 0:
                error_msg = "Plumed in (LAMMPS) input but no plumed files found."
                logging.error(f"{error_msg}\nAborting...")
                sys.exit(1)
            # Read the contents of each plumed file into a dictionary
            plumed_input = {}
            for plumed_file in plumed_files:
                plumed_input[plumed_file.name] = textfile_to_string_list(plumed_file)
            # Analyze each plumed file to determine whether it contains MOVINGRESTRAINTS keyword (SMD)
            for plumed_file in plumed_files:
                plumed[1], plumed[2] = analyze_plumed_file_for_movres(
                    plumed_input[plumed_file.name]
                )
                if plumed[1] and plumed[2] != 0:
                    break

        # Set the system params for exploration
        (
            system_timestep_ps,
            system_temperature_K,
            system_exp_time_ps,
            system_max_exp_time_ps,
            system_job_walltime_h,
            system_init_exp_time_ps,
            system_init_job_walltime_h,
            system_print_mult,
            system_disturbed_start,
        ) = get_system_exploration(current_config, system_auto_index)
        logging.debug(
            f"{system_timestep_ps,system_temperature_K,system_exp_time_ps,system_max_exp_time_ps,system_job_walltime_h,system_init_exp_time_ps,system_init_job_walltime_h,system_print_mult,system_disturbed_start}"
        )

        # Set the system params for exploration
        if curr_iter == 1:
            # No distrubed start
            system_disturbed_start = False

        else:
            # Get starting points
            (
                starting_points,
                starting_points_bckp,
                system_disturbed_start,
            ) = generate_starting_points(
                exploration_type,
                system_auto,
                training_path,
                padded_curr_iter,
                prev_exploration_config,
                user_config_present,
                system_disturbed_start,
            )

        # LAMMPS Input
        input_replace_dict["_R_TIMESTEP_"] = f"{system_timestep_ps}"

        if exploration_type == 0:
            input_replace_dict["_R_TEMPERATURE_"] = f"{system_temperature_K}"

            # First exploration
            if curr_iter == 1:
                system_lammps_data_fn = system_auto + ".lmp"
                system_lammps_data = textfile_to_string_list(
                    training_path / "user_files" / system_lammps_data_fn
                )
                input_replace_dict["_R_DATA_FILE_"] = system_lammps_data_fn

                # Default time and nb steps
                if plumed[1]:
                    system_nb_steps = plumed[2]
                else:
                    system_nb_steps = system_init_exp_time_ps / system_timestep_ps
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(system_nb_steps)}"

                system_walltime_approx_s = system_init_job_walltime_h * 3600

                # Get the cell and nb of atoms: It can be done now because the starting point is the same by NNP and by traj
                system_nb_atm, num_atom_types, box, masses, coords = read_lammps_data(
                    system_lammps_data
                )
                system_cell = [box[1] - box[0], box[3] - box[2], box[5] - box[4]]

            # Subsequent ones
            else:
                # SMD wins
                if plumed[1]:
                    system_nb_steps = plumed[2]
                # User inputs
                elif "system_exp_time_ps" in user_config:
                    system_nb_steps = system_exp_time_ps / system_timestep_ps
                # Auto value
                else:
                    system_nb_steps *= update_system_nb_steps_factor(
                        prev_exploration_config, system_auto
                    )
                    # Update if over Max value
                    if system_nb_steps > system_max_exp_time_ps / system_timestep_ps:
                        system_nb_steps = system_max_exp_time_ps / system_timestep_ps
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(system_nb_steps)}"
                # Update the new input
                current_config["system_exp_time_ps"] = (
                    system_nb_steps * system_timestep_ps
                )

                # Walltime
                if "system_job_walltime_h" in user_config:
                    system_walltime_approx_s = int(system_job_walltime_h * 3600)
                else:
                    # Abritary factor
                    system_walltime_approx_s = int(
                        (
                            prev_exploration_config["systems_auto"][system_auto][
                                "s_per_step"
                            ]
                            * system_nb_steps
                        )
                        * 1.20
                    )
                # Update the new input
                current_config["system_job_walltime_h"] = (
                    system_walltime_approx_s / 3600
                )

            # Get print freq
            system_print_every_x_steps = system_nb_steps * system_print_mult
            input_replace_dict["_R_PRINT_FREQ_"] = f"{int(system_print_every_x_steps)}"

        # i-PI // UNTESTED
        elif exploration_type == 1:
            system_temperature_K = float(
                get_temperature_from_ipi_xml(master_system_ipi_xml)
            )
            if system_temperature_K == -1:
                error_msg = f"No temperature found in the xml: {training_path / 'files' / (system_auto + '.xml')}."
                logging.error(f"{error_msg}\nAborting...")
                sys.exit(1)

            # TODO: This should be read like temp
            input_replace_dict["_R_NB_BEADS_"] = str(8)

            # First exploration
            if curr_iter == 1:
                system_lammps_data_fn = system_auto + ".lmp"
                system_lammps_data = textfile_to_string_list(
                    training_path / "user_files" / system_lammps_data_fn
                )
                system_ipi_xyz_fn = system_auto + ".xyz"
                input_replace_dict["_R_DATA_FILE_"] = system_ipi_xyz_fn
                master_system_ipi_json["coord_file"] = system_ipi_xyz_fn
                # Get the XYZ file from LMP
                subprocess.run(
                    [
                        atomsk_bin,
                        str(Path("../") / "user_files" / system_lammps_data_fn),
                        # str(training_path / "user_files" / system_lammps_data_fn),
                        "xyz",
                        str(Path("../") / "user_files" / system_auto),
                        # str(training_path / "user_files" / system_auto),
                        "-ow",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
                system_ipi_xyz = textfile_to_string_list(
                    training_path / "user_files" / system_ipi_xyz_fn
                )
                # Default time and nb steps
                if plumed[1]:
                    system_nb_steps = plumed[2]
                else:
                    system_nb_steps = system_init_exp_time_ps / system_timestep_ps
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(system_nb_steps)}"

                system_walltime_approx_s = system_init_job_walltime_h * 3600

                # Get the cell and nb of atoms: It can be done now because the starting point is the same by NNP and by traj
                system_nb_atm, num_atom_types, box, masses, coords = read_lammps_data(
                    system_lammps_data
                )
                system_cell = [box[1] - box[0], box[3] - box[2], box[5] - box[4]]
                input_replace_dict["_R_CELL_"] = f"{system_cell}"
                # Get the type_map from config (key added after first training)
                for it_zzz, zzz in enumerate(main_config["type_map"]):
                    master_system_ipi_json["atom_type"][str(zzz)] = it_zzz

            # Subsequent ones
            else:
                # SMD wins
                if plumed[1]:
                    system_nb_steps = plumed[2]
                # User inputs
                elif "system_exp_time_ps" in user_config:
                    system_nb_steps = system_exp_time_ps / system_timestep_ps
                # Auto value
                else:
                    system_nb_steps *= update_system_nb_steps_factor(
                        prev_exploration_config, system_auto
                    )
                    # Update if over Max value
                    if system_nb_steps > system_max_exp_time_ps / system_timestep_ps:
                        system_nb_steps = system_max_exp_time_ps / system_timestep_ps
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(system_nb_steps)}"
                # Update the new input
                current_config["system_exp_time_ps"] = (
                    system_nb_steps * system_timestep_ps
                )

                # Walltime
                if user_config_present:
                    system_walltime_approx_s = int(system_job_walltime_h * 3600)
                else:
                    # Abritary factor
                    system_walltime_approx_s = int(
                        (
                            prev_exploration_config["systems_auto"][system_auto][
                                "s_per_step"
                            ]
                            * system_nb_steps
                        )
                        * 1.20
                    )
                # Update the new input
                current_config["system_job_walltime_h"] = (
                    system_walltime_approx_s / 3600
                )

            # Get print freq
            system_print_every_x_steps = system_nb_steps * system_print_mult
            input_replace_dict["_R_PRINT_FREQ_"] = f"{int(system_print_every_x_steps)}"

        # Now it is by NNP and by traj_count
        for nnp_index in range(1, main_config["nnp_count"] + 1):
            for traj_index in range(1, exploration_config["traj_count"] + 1):
                local_path = (
                    Path(".").resolve()
                    / str(system_auto)
                    / str(nnp_index)
                    / (str(traj_index).zfill(5))
                )
                local_path.mkdir(exist_ok=True, parents=True)

                models_list, models_string = create_models_list(
                    main_config,
                    prev_training_config,
                    nnp_index,
                    padded_prev_iter,
                    training_path,
                    local_path,
                )

                # LAMMPS
                if exploration_type == 0:
                    system_lammps_in = copy.deepcopy(master_system_lammps_in)
                    input_replace_dict[
                        "_R_SEED_VEL_"
                    ] = f"{nnp_index}{random.randrange(0, 1000)}{traj_index}{padded_curr_iter}"
                    input_replace_dict[
                        "_R_SEED_THER_"
                    ] = f"{nnp_index}{random.randrange(0, 1000)}{traj_index}{padded_curr_iter}"
                    input_replace_dict[
                        "_R_DCD_OUT_"
                    ] = f"{system_auto}_{nnp_index}_{padded_curr_iter}.dcd"
                    input_replace_dict[
                        "_R_RESTART_OUT_"
                    ] = f"{system_auto}_{nnp_index}_{padded_curr_iter}.restart"
                    input_replace_dict["_R_MODEL_FILES_LIST_"] = models_string
                    input_replace_dict[
                        "_R_DEVI_OUT_"
                    ] = f"model_devi_{system_auto}_{nnp_index}_{padded_curr_iter}.out"
                    # Get data files (starting points) and number of steps
                    if curr_iter > 1:
                        if len(starting_points) == 0:
                            starting_point_list = copy.deepcopy(starting_points_bckp)
                        system_lammps_data_fn = starting_point_list[
                            random.randrange(0, len(starting_point_list))
                        ]
                        system_lammps_data = textfile_to_string_list(
                            training_path
                            / "starting_structures"
                            / system_lammps_data_fn
                        )
                        input_replace_dict["_R_DATA_FILE_"] = system_lammps_data_fn
                        # Get again the system_cell and nb_atom
                        (
                            system_nb_atm,
                            num_atom_types,
                            box,
                            masses,
                            coords,
                        ) = read_lammps_data(system_lammps_data)
                        system_cell = [
                            box[1] - box[0],
                            box[3] - box[2],
                            box[5] - box[4],
                        ]

                    # Plumed files
                    if plumed[0]:
                        input_replace_dict[
                            "_R_PLUMED_IN_"
                        ] = f"plumed_{system_auto}.dat"
                        input_replace_dict[
                            "_R_PLUMED_OUT_"
                        ] = f"plumed_{system_auto}_{nnp_index}_{padded_curr_iter}.log"
                        for it_plumed_input in plumed_input:
                            plumed_input[
                                it_plumed_input
                            ] = replace_substring_in_string_list(
                                plumed_input[it_plumed_input],
                                "_R_PRINT_FREQ_",
                                f"{int(system_print_every_x_steps)}",
                            )
                            string_list_to_textfile(
                                local_path / it_plumed_input,
                                plumed_input[it_plumed_input],
                            )

                    # Write DATA file
                    string_list_to_textfile(
                        local_path / system_lammps_data_fn, system_lammps_data
                    )

                    exploration_config["systems_auto"][system_auto]["nb_steps"] = int(
                        system_nb_steps
                    )
                    exploration_config["systems_auto"][system_auto][
                        "print_every_x_steps"
                    ] = int(system_print_every_x_steps)

                    #  Write INPUT file
                    for key, value in input_replace_dict.items():
                        system_lammps_in = replace_substring_in_string_list(
                            system_lammps_in, key, value
                        )
                    string_list_to_textfile(
                        local_path
                        / (f"{system_auto}_{nnp_index}_{padded_curr_iter}.in"),
                        system_lammps_in,
                    )

                    # Slurm file
                    job_file = replace_in_slurm_file_general(
                        master_job_file,
                        machine_spec,
                        system_walltime_approx_s,
                        machine_walltime_format,
                        current_config["job_email"],
                    )

                    job_file = replace_substring_in_string_list(
                        job_file,
                        "_R_DEEPMD_VERSION_",
                        f"{exploration_config['deepmd_model_version']}",
                    )
                    job_file = replace_substring_in_string_list(
                        job_file,
                        "_R_MODEL_FILES_LIST_",
                        str(models_string.replace(" ", '" "')),
                    )
                    job_file = replace_substring_in_string_list(
                        job_file,
                        "_R_INPUT_FILE_",
                        f"{system_auto}_{nnp_index}_{padded_curr_iter}",
                    )
                    job_file = replace_substring_in_string_list(
                        job_file, "_R_DATA_FILE_", f"{system_lammps_data_fn}"
                    )
                    job_file = replace_substring_in_string_list(
                        job_file, ' "_R_RERUN_FILE_"', ""
                    )
                    if plumed[0] == 1:
                        for n, it_plumed_input in enumerate(plumed_input):
                            if n == 0:
                                job_file = replace_substring_in_string_list(
                                    job_file, "_R_PLUMED_FILES_LIST_", it_plumed_input
                                )
                            else:
                                job_file = replace_substring_in_string_list(
                                    job_file,
                                    prev_plumed,
                                    prev_plumed + '" "' + it_plumed_input,
                                )
                            prev_plumed = it_plumed_input
                    else:
                        job_file = replace_substring_in_string_list(
                            job_file, ' "_R_PLUMED_FILES_LIST_"', ""
                        )
                    string_list_to_textfile(
                        local_path
                        / f"job_deepmd_{exploration_config['exploration_type']}_{machine_spec['arch_type']}_{machine}.sh",
                        job_file,
                    )
                    del system_lammps_in
                    del job_file
                # i-PI
                elif exploration_type == 1:
                    system_ipi_json = copy.deepcopy(master_system_ipi_json)
                    system_ipi_xml_aslist = copy.deepcopy(master_system_ipi_xml_aslist)
                    input_replace_dict[
                        "_R_SEED_"
                    ] = f"{nnp_index}{random.randrange(0, 1000)}{traj_index}{padded_curr_iter}"
                    input_replace_dict[
                        "_R_SYS_"
                    ] = f"{system_auto}_{nnp_index}_{padded_curr_iter}"
                    # Get data files (starting points) and number of steps
                    if curr_iter > 1:
                        if len(starting_points) == 0:
                            starting_point_list = copy.deepcopy(starting_points_bckp)
                        system_ipi_xyz_fn = starting_point_list[
                            random.randrange(0, len(starting_point_list))
                        ]
                        system_ipi_xyz = textfile_to_string_list(
                            training_path / "starting_structures" / system_ipi_xyz_fn
                        )
                        input_replace_dict["_R_XYZ_"] = system_ipi_xyz_fn
                        system_ipi_json["coord_file"] = system_ipi_xyz_fn
                        for it_zzz, zzz in enumerate(main_config["type_map"]):
                            system_ipi_json["atom_type"][str(zzz)] = it_zzz
                        system_lammps_data = textfile_to_string_list(
                            training_path
                            / "starting_structures"
                            / system_ipi_xyz_fn.replace(".xyz", ".lmp")
                        )
                        # Get again the system_cell and nb_atom
                        (
                            system_nb_atm,
                            num_atom_types,
                            box,
                            masses,
                            coords,
                        ) = read_lammps_data(system_lammps_data)
                        system_cell = [
                            box[1] - box[0],
                            box[3] - box[2],
                            box[5] - box[4],
                        ]
                        input_replace_dict["_R_CELL_"] = f"{system_cell}"

                    # Plumed files
                    if plumed[0]:
                        input_replace_dict[
                            "_R_PLUMED_IN_"
                        ] = f"plumed_{system_auto}.dat"
                        for it_plumed_input in plumed_input:
                            plumed_input[
                                it_plumed_input
                            ] = replace_substring_in_string_list(
                                plumed_input[it_plumed_input],
                                "_R_PRINT_FREQ_",
                                f"{int(system_print_every_x_steps)}",
                            )
                            # Because of weird units of time
                            plumed_input[
                                it_plumed_input
                            ] = replace_substring_in_string_list(
                                plumed_input[it_plumed_input],
                                "UNITS LENGTH",
                                "UNITS TIME="
                                + str(2.4188843e-05 / system_timestep_ps)
                                + " LENGTH",
                            )
                            string_list_to_textfile(
                                local_path / it_plumed_input,
                                plumed_input[it_plumed_input],
                            )
                        del it_plumed_input

                    system_ipi_json["graph_file"] = models_list[0]

                    #  Write INPUT files
                    for key, value in input_replace_dict.items():
                        system_ipi_xml_aslist = replace_substring_in_string_list(
                            system_ipi_xml_aslist, key, value
                        )
                    del key, value
                    system_ipi_xml = string_list_to_xml(system_ipi_xml_aslist)
                    write_xml_file(
                        system_ipi_xml,
                        local_path
                        / (f"{system_auto}_{nnp_index}_{padded_curr_iter}.xml"),
                    )
                    write_json_file(
                        system_ipi_json,
                        local_path
                        / (f"{system_auto}_{nnp_index}_{padded_curr_iter}.json"),
                    )

                    # Slurm file
                    job_file = replace_in_slurm_file_general(
                        master_job_file,
                        machine_spec,
                        system_walltime_approx_s,
                        machine_walltime_format,
                        current_config["job_email"],
                    )

                    job_file = replace_substring_in_string_list(
                        job_file,
                        "_R_DEEPMD_VERSION_",
                        f"{exploration_config['deepmd_model_version']}",
                    )
                    job_file = replace_substring_in_string_list(
                        job_file, "_R_MODEL_FILES_LIST_", f"{models_list[0]}"
                    )
                    job_file = replace_substring_in_string_list(
                        job_file,
                        "_R_INPUT_FILE_",
                        f"{system_auto}_{nnp_index}_{padded_curr_iter}",
                    )
                    job_file = replace_substring_in_string_list(
                        job_file, "_R_DATA_FILE_", f"{system_ipi_xyz_fn}"
                    )
                    if plumed[0] == 1:
                        for n, it_plumed_input in enumerate(plumed_input):
                            if n == 0:
                                job_file = replace_substring_in_string_list(
                                    job_file, "_R_PLUMED_FILES_LIST_", it_plumed_input
                                )
                            else:
                                job_file = replace_substring_in_string_list(
                                    job_file,
                                    prev_plumed,
                                    prev_plumed + '" "' + it_plumed_input,
                                )
                            prev_plumed = it_plumed_input
                        del n, it_plumed_input, prev_plumed
                    else:
                        job_file = replace_substring_in_string_list(
                            job_file, ' "_R_PLUMED_FILES_LIST_"', ""
                        )
                    string_list_to_textfile(
                        local_path
                        / f"job_deepmd_{exploration_config['exploration_type']}_{machine_spec['arch_type']}_{machine}.sh",
                        job_file,
                    )
                    del (
                        system_ipi_xml_aslist,
                        system_ipi_xml,
                        system_ipi_json,
                    )
                    del job_file
                else:
                    error_msg = f"Exploration is unknown/not set."
                    logging.error(f"{error_msg}\nAborting...")
                    sys.exit(1)

            del traj_index, models_list, models_string, local_path

        del nnp_index

        exploration_config["systems_auto"][system_auto][
            "temperature_K"
        ] = system_temperature_K
        exploration_config["systems_auto"][system_auto][
            "timestep_ps"
        ] = system_timestep_ps
        exploration_config["systems_auto"][system_auto][
            "disturbed_start"
        ] = system_disturbed_start

        main_config["systems_auto"][system_auto]["cell"] = system_cell
        main_config["systems_auto"][system_auto]["nb_atm"] = system_nb_atm

        if plumed[0] == 1:
            del plumed_input, plumed
        del system_temperature_K, system_cell, system_nb_atm, system_nb_steps
        del system_lammps_data, system_timestep_ps, system_walltime_approx_s

    del system_auto_index, system_auto, master_job_file

    exploration_config["is_locked"] = True
    exploration_config["is_launched"] = False
    exploration_config["is_checked"] = False
    if exploration_type == 1:
        exploration_config["is_unbeaded"] = False
        exploration_config["is_reruned"] = False
        exploration_config["is_rechecked"] = False
    exploration_config["is_deviated"] = False
    exploration_config["is_extracted"] = False
    del exploration_type

    write_json_file(main_config, (control_path / "config.json"))
    write_json_file(
        exploration_config, (control_path / f"exploration_{padded_curr_iter}.json")
    )
    backup_and_overwrite_json_file(
        current_config, (current_path / user_config_filename)
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "preparation",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_config_filename=sys.argv[3],
        )
    else:
        pass
