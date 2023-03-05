from pathlib import Path
import logging
import sys
import copy
import subprocess
import random

# ### Non-standard imports
import numpy as np

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    backup_and_overwrite_json_file,
    load_default_json_file,
    read_key_input_json,
)
from deepmd_iterative.common.list import replace_substring_in_list_of_strings
from deepmd_iterative.common.xml import (
    parse_xml_file,
    convert_xml_to_list_of_strings,
    convert_list_of_strings_to_xml,
    write_xml,
)
from deepmd_iterative.common.machine import get_machine_spec_for_step
from deepmd_iterative.common.file import (
    check_file_existence,
    file_to_list_of_strings,
    check_directory,
    write_list_of_strings_to_file,
)

from deepmd_iterative.common.slurm import replace_in_slurm_file_general
from deepmd_iterative.common.check import validate_step_folder, check_atomsk
from deepmd_iterative.common.plumed import analyze_plumed_file_for_movres
from deepmd_iterative.common.lammps import parse_lammps_data
from deepmd_iterative.common.exploration import (
    generate_starting_points,
    create_models_list,
    update_nb_steps_factor,
)
from deepmd_iterative.common.ipi import get_temperature_from_ipi_xml


def main(
    step_name: str,
    phase_name: str,
    deepmd_iterative_path,
    fake_machine=None,
    input_fn: str = "input.json",
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

    # ### Get default inputs json
    default_present = False
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "data" / "inputs.json"
    )

    if bool(default_input_json):
        default_present = True

    # ### Get input json (user one)
    if (current_path / input_fn).is_file():
        input_json = load_json_file((current_path / input_fn))
        input_present = True
    else:
        input_json = {}
        input_present = False
    new_input_json = copy.deepcopy(input_json)

    # ### Check if atomsk is present
    atomsk_bin = check_atomsk(
        read_key_input_json(
            input_json,
            new_input_json,
            "atomsk_path",
            default_input_json,
            step_name,
            default_present,
        )
    )

    # ### Get control path and config_json
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))
    previous_iteration_zfill = str(current_iteration - 1).zfill(3)
    prevtraining_json = load_json_file(
        (control_path / ("training_" + previous_iteration_zfill + ".json"))
    )
    if int(previous_iteration_zfill) > 0:
        prevexploration_json = load_json_file(
            (control_path / ("exploration_" + previous_iteration_zfill + ".json"))
        )

    # ### Get extra needed paths
    jobs_path = deepmd_iterative_path / "data" / "jobs" / "exploration"

    # ### Get user machine keyword
    user_machine_keyword = read_key_input_json(
        input_json,
        new_input_json,
        "user_machine_keyword",
        default_input_json,
        step_name,
        default_present,
    )
    user_machine_keyword = (
        None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    )

    # ### Read machine info
    (
        machine,
        machine_spec,
        machine_walltime_format,
        machine_launch_command,
    ) = get_machine_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "training",
        fake_machine,
        user_machine_keyword,
    )
    if fake_machine is not None:
        logging.info(f"Pretending to be on {fake_machine}")
    else:
        logging.info(f"machine is {machine}")
    del fake_machine

    # # ### Checks
    # if current_iteration > 0:
    #     labeling_json = load_json_file(
    #         (control_path / f"labeling_{current_iteration_zfill}.json"), True, True
    #     )
    #     if not labeling_json["is_extracted"]:
    #         logging.error("Lock found. Execute first: training update_iter")
    #         logging.error("Aborting...")
    #         return 1

    # ### Get/Create exploration parameters
    exploration_json = load_json_file(
        (control_path / f"exploration_{current_iteration_zfill}.json"),
        abort_on_error=False,
    )

    exploration_json["deepmd_model_version"] = prevtraining_json["deepmd_model_version"]
    exploration_json["nb_nnp"] = config_json["nb_nnp"]
    exploration_json["exploration_type"] = config_json["exploration_type"]
    exploration_json["nb_traj"] = read_key_input_json(
        input_json,
        new_input_json,
        "nb_traj",
        default_input_json,
        step_name,
        default_present,
    )

    exploration_json["machine"] = machine
    exploration_json["project_name"] = machine_spec["project_name"]
    exploration_json["allocation_name"] = machine_spec["allocation_name"]
    exploration_json["arch_name"] = machine_spec["arch_name"]
    exploration_json["arch_type"] = machine_spec["arch_type"]
    exploration_json["launch_command"] = machine_launch_command

    check_file_existence(
        jobs_path
        / (
            "job_deepmd_"
            + exploration_json["exploration_type"]
            + "_"
            + exploration_json["arch_type"]
            + "_"
            + machine
            + ".sh"
        ),
        error_msg="No SLURM file present for the exploration step on this machine.",
    )
    slurm_file_master = file_to_list_of_strings(
        jobs_path
        / (
            "job_deepmd_"
            + exploration_json["exploration_type"]
            + "_"
            + exploration_json["arch_type"]
            + "_"
            + machine
            + ".sh"
        )
    )
    del jobs_path

    ### Preparation of the exploration
    exploration_json["subsys_nr"] = {}
    for it0_subsys_nr, it_subsys_nr in enumerate(config_json["subsys_nr"]):

        random.seed()
        exploration_json["subsys_nr"][it_subsys_nr] = {}

        plumed = [False, False, False]
        exploration_type = -1

        input_replace_dict = {}

        # ### Get the input file (.in or .xml) first and check if plumed
        if exploration_json["exploration_type"] == "lammps":
            exploration_type = 0
            subsys_lammps_in = file_to_list_of_strings(
                training_path / "files" / (it_subsys_nr + ".in")
            )
            if any("plumed" in zzz for zzz in subsys_lammps_in):
                plumed[0] = True
        elif exploration_json["exploration_type"] == "i-PI":
            exploration_type = 1
            subsys_ipi_xml = parse_xml_file(
                training_path / "files" / (it_subsys_nr + ".xml")
            )
            subsys_ipi_xml_aslist = convert_xml_to_list_of_strings(subsys_ipi_xml)
            subsys_ipi_json = {
                "verbose": False,
                "use_unix": False,
                "port": "_R_NB_PORT_",
                "host": "_R_ADDRESS_",
                "graph_file": "_R_GRAPH_",
                "coord_file": "_R_XYZ_",
                "atom_type": {},
            }
            if any("plumed" in zzz for zzz in subsys_ipi_xml_aslist):
                plumed[0] = True

        # ### Get plumed files
        if plumed[0] == 1:
            plumed_files_list = [
                plumed_file
                for plumed_file in (training_path / "files").glob(
                    f"plumed*_{it_subsys_nr}.dat"
                )
            ]
            if len(plumed_files_list) == 0:
                error_msg = "Plumed in (LAMMPS) input but no plumed files found."
                logging.error(f"{error_msg}\nAborting...")
                sys.exit(1)
            plumed_input = {}
            for it_plumed_files_list in plumed_files_list:
                plumed_input[it_plumed_files_list.name] = file_to_list_of_strings(
                    it_plumed_files_list
                )
            for it_plumed_files_list in plumed_files_list:
                plumed[1], plumed[2] = analyze_plumed_file_for_movres(
                    plumed_input[it_plumed_files_list.name]
                )
                if plumed[1] and plumed[2] != 0:
                    break

        # ### Timestep
        subsys_timestep = read_key_input_json(
            input_json,
            new_input_json,
            "timestep_ps",
            default_input_json,
            step_name,
            default_present,
            subsys_index=it0_subsys_nr,
            subsys_number=len(config_json["subsys_nr"]),
            exploration_dep=exploration_type,
        )

        # ### Temperature
        subsys_temp = read_key_input_json(
            input_json,
            new_input_json,
            "temperature_K",
            default_input_json,
            step_name,
            default_present,
            subsys_index=it0_subsys_nr,
            subsys_number=len(config_json["subsys_nr"]),
            exploration_dep=exploration_type,
        )

        # ### exploration time
        subsys_exp_time_ps = read_key_input_json(
            input_json,
            new_input_json,
            "exp_time_ps",
            default_input_json,
            step_name,
            default_present,
            subsys_index=it0_subsys_nr,
            subsys_number=len(config_json["subsys_nr"]),
            exploration_dep=exploration_type,
        )

        # ### Max exploration time
        subsys_max_exp_time_ps = read_key_input_json(
            input_json,
            new_input_json,
            "max_exp_time_ps",
            default_input_json,
            step_name,
            default_present,
            subsys_index=it0_subsys_nr,
            subsys_number=len(config_json["subsys_nr"]),
            exploration_dep=exploration_type,
        )

        # ###  Wall Time
        subsys_job_walltime_h = read_key_input_json(
            input_json,
            new_input_json,
            "job_walltime_h",
            default_input_json,
            step_name,
            default_present,
            subsys_index=it0_subsys_nr,
            subsys_number=len(config_json["subsys_nr"]),
            exploration_dep=exploration_type,
        )

        # ### Print mult
        subsys_print_mult = read_key_input_json(
            input_json,
            new_input_json,
            "print_mult",
            default_input_json,
            step_name,
            default_present,
            subsys_index=it0_subsys_nr,
            subsys_number=len(config_json["subsys_nr"]),
        )

        # ### Disturbed start
        subsys_disturbed_start = read_key_input_json(
            input_json,
            new_input_json,
            "disturbed_start",
            default_input_json,
            step_name,
            default_present,
            subsys_index=it0_subsys_nr,
            subsys_number=len(config_json["subsys_nr"]),
            exploration_dep=exploration_type,
        )

        if current_iteration == 1:
            # ### Initial Exploration Time
            subsys_init_exp_time_ps = read_key_input_json(
                input_json,
                new_input_json,
                "init_exp_time_ps",
                default_input_json,
                step_name,
                default_present,
                subsys_index=it0_subsys_nr,
                subsys_number=len(config_json["subsys_nr"]),
                exploration_dep=exploration_type,
            )

            # ### Initial Wall Time
            subsys_init_job_walltime_h = read_key_input_json(
                input_json,
                new_input_json,
                "init_job_walltime_h",
                default_input_json,
                step_name,
                default_present,
                subsys_index=it0_subsys_nr,
                subsys_number=len(config_json["subsys_nr"]),
                exploration_dep=exploration_type,
            )

            # ### No distrubed start
            exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"] = False

        else:
            # ### Get starting points
            (
                starting_points,
                starting_points_bckp,
                exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"],
            ) = generate_starting_points(
                exploration_type,
                it_subsys_nr,
                training_path,
                previous_iteration_zfill,
                prevexploration_json,
                input_present,
                subsys_disturbed_start,
            )

        # ### LAMMPS Input
        input_replace_dict["_R_TIMESTEP_"] = f"{subsys_timestep}"

        if exploration_type == 0:
            input_replace_dict["_R_TEMPERATURE_"] = f"{subsys_temp}"

            # ### First exploration
            if current_iteration == 1:
                subsys_lammps_data_fn = it_subsys_nr + ".lmp"
                subsys_lammps_data = file_to_list_of_strings(
                    training_path / "files" / subsys_lammps_data_fn
                )
                input_replace_dict["_R_DATA_FILE_"] = subsys_lammps_data_fn

                # ### Default time and nb steps
                if plumed[1]:
                    subsys_nb_steps = plumed[2]
                else:
                    subsys_nb_steps = subsys_init_exp_time_ps / subsys_timestep
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(subsys_nb_steps)}"

                subsys_walltime_approx_s = subsys_init_job_walltime_h * 3600

                # ### Get the cell and nb of atoms: It can be done now because the starting point is the same by NNP and by traj
                subsys_nb_atm, num_atom_types, box, masses, coords = parse_lammps_data(
                    subsys_lammps_data
                )
                subsys_cell = [box[1] - box[0], box[3] - box[2], box[5] - box[4]]

            # ### Subsequent ones
            else:
                # ### SMD wins
                if plumed[1]:
                    subsys_nb_steps = plumed[2]
                # ### User inputs
                elif input_present:
                    subsys_nb_steps = subsys_nb_steps
                # ### Auto value
                else:
                    subsys_nb_steps *= update_nb_steps_factor(
                        prevexploration_json, it_subsys_nr
                    )
                    ### Update if over Max value
                    if subsys_nb_steps > subsys_max_exp_time_ps / subsys_timestep:
                        subsys_nb_steps = subsys_max_exp_time_ps / subsys_timestep
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(subsys_nb_steps)}"

                # ### Walltime
                if input_present:
                    subsys_walltime_approx_s = int(subsys_job_walltime_h * 3600)
                else:
                    # ### Abritary factor
                    subsys_walltime_approx_s = int(
                        (
                            prevexploration_json["subsys_nr"][it_subsys_nr]["s_per_step"]
                            * subsys_nb_steps
                        )
                        * 1.20
                    )

            # ### Get print freq
            subsys_print_every_x_steps = subsys_nb_steps * subsys_print_mult
            input_replace_dict["_R_PRINT_FREQ_"] = f"{int(subsys_print_every_x_steps)}"

        # ### i-PI // UNTESTED
        elif exploration_type == 1:
            subsys_temp = float(get_temperature_from_ipi_xml(subsys_ipi_xml))
            if subsys_temp == -1:
                error_msg = f"No temperature found in the xml: {training_path / 'files' / (it_subsys_nr + '.xml')}."
                logging.error(f"{error_msg}\nAborting...")
                sys.exit(1)

            # ### TODO: This should be read like temp
            input_replace_dict["_R_NB_BEADS_"] = str(8)

            # ### First exploration
            if current_iteration == 1:
                subsys_lammps_data_fn = it_subsys_nr + ".lmp"
                subsys_lammps_data = file_to_list_of_strings(
                    training_path / "inputs" / subsys_lammps_data_fn
                )
                subsys_ipi_xyz_fn = it_subsys_nr + ".xyz"
                input_replace_dict["_R_DATA_FILE_"] = subsys_ipi_xyz_fn
                subsys_ipi_json["coord_file"] = subsys_ipi_xyz_fn
                # ### Get the XYZ file from LMP
                subprocess.call(
                    [
                        atomsk_bin,
                        str(training_path / "files" / subsys_lammps_data_fn),
                        "xyz",
                        str(training_path / "files" / it_subsys_nr),
                        "-ow",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
                subsys_ipi_xyz = file_to_list_of_strings(
                    training_path / "files" / subsys_ipi_xyz_fn
                )
                # ### Default time and nb steps
                if plumed[1]:
                    subsys_nb_steps = plumed[2]
                else:
                    subsys_nb_steps = subsys_init_exp_time_ps / subsys_timestep
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(subsys_nb_steps)}"

                subsys_walltime_approx_s = subsys_init_job_walltime_h * 3600

                # ### Get the cell and nb of atoms: It can be done now because the starting point is the same by NNP and by traj
                subsys_nb_atm, num_atom_types, box, masses, coords = parse_lammps_data(
                    subsys_lammps_data
                )
                subsys_cell = [box[1] - box[0], box[3] - box[2], box[5] - box[4]]
                input_replace_dict["_R_CELL_"] = f"{subsys_cell}"
                # ### Get the type_map from config (key added after first training)
                for it_zzz, zzz in enumerate(config_json["type_map"]):
                    subsys_ipi_json["atom_type"][str(zzz)] = it_zzz

            # ### Subsequent ones
            else:
                # ### SMD wins
                if plumed[1]:
                    subsys_nb_steps = plumed[2]
                # ### User inputs
                elif input_present:
                    subsys_nb_steps = subsys_nb_steps
                # ### Auto value
                else:
                    subsys_nb_steps *= update_nb_steps_factor(
                        prevexploration_json, it_subsys_nr
                    )
                    ### Update if over Max value
                    if subsys_nb_steps > subsys_max_exp_time_ps / subsys_timestep:
                        subsys_nb_steps = subsys_max_exp_time_ps / subsys_timestep
                input_replace_dict["_R_NUMBER_OF_STEPS_"] = f"{int(subsys_nb_steps)}"

                # ### Walltime
                if input_present:
                    subsys_walltime_approx_s = int(subsys_job_walltime_h * 3600)
                else:
                    # ### Abritary factor
                    subsys_walltime_approx_s = int(
                        (
                            prevexploration_json["subsys_nr"][it_subsys_nr]["s_per_step"]
                            * subsys_nb_steps
                        )
                        * 1.20
                    )

            # ### Get print freq
            subsys_print_every_x_steps = subsys_nb_steps * subsys_print_mult
            input_replace_dict["_R_PRINT_FREQ_"] = f"{int(subsys_print_every_x_steps)}"

        # ### Now it is by NNP and by nb_traj
        for it_nnp in range(1, config_json["nb_nnp"] + 1):
            for it_number in range(1, exploration_json["nb_traj"] + 1):

                local_path = (
                    Path(".").resolve()
                    / str(it_subsys_nr)
                    / str(it_nnp)
                    / (str(it_number).zfill(5))
                )
                local_path.mkdir(exist_ok=True, parents=True)

                models_list, models_string = create_models_list(
                    config_json,
                    prevtraining_json,
                    it_nnp,
                    previous_iteration_zfill,
                    training_path,
                    local_path,
                )

                # ### LAMMPS
                if exploration_type == 0:
                    it_subsys_lammps_in = copy.deepcopy(subsys_lammps_in)
                    input_replace_dict[
                        "_R_SEED_VEL_"
                    ] = f"{it_nnp}{random.randrange(0, 1000)}{it_number}{previous_iteration_zfill}"
                    input_replace_dict[
                        "_R_SEED_THER_"
                    ] = f"{it_nnp}{random.randrange(0, 1000)}{it_number}{previous_iteration_zfill}"
                    input_replace_dict[
                        "_R_DCD_OUT_"
                    ] = f"{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}.dcd"
                    input_replace_dict[
                        "_R_RESTART_OUT_"
                    ] = f"{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}.restart"
                    input_replace_dict["_R_MODEL_FILES_LIST_"] = models_string
                    input_replace_dict[
                        "_R_DEVI_OUT_"
                    ] = f"model_devi_{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}.out"
                    # ### Get data files (starting points) and number of steps
                    if current_iteration > 1:
                        if len(starting_points) == 0:
                            starting_point_list = copy.deepcopy(starting_points_bckp)
                        subsys_lammps_data_fn = starting_point_list[
                            random.randrange(0, len(starting_point_list))
                        ]
                        subsys_lammps_data = file_to_list_of_strings(
                            training_path
                            / "starting_structures"
                            / subsys_lammps_data_fn
                        )
                        input_replace_dict["_R_DATA_FILE_"] = subsys_lammps_data_fn
                        # ### Get again the subsys_cell and nb_atom
                        (
                            subsys_nb_atm,
                            num_atom_types,
                            box,
                            masses,
                            coords,
                        ) = parse_lammps_data(subsys_lammps_data)
                        subsys_cell = [
                            box[1] - box[0],
                            box[3] - box[2],
                            box[5] - box[4],
                        ]

                    # ### Plumed files
                    if plumed[0]:
                        input_replace_dict[
                            "_R_PLUMED_IN_"
                        ] = f"plumed_{it_subsys_nr}.dat"
                        input_replace_dict[
                            "_R_PLUMED_OUT_"
                        ] = f"plumed_{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}.log"
                        for it_plumed_input in plumed_input:
                            plumed_input[
                                it_plumed_input
                            ] = replace_substring_in_list_of_strings(
                                plumed_input[it_plumed_input],
                                "_R_PRINT_FREQ_",
                                f"{int(subsys_print_every_x_steps)}",
                            )
                            write_list_of_strings_to_file(
                                local_path / it_plumed_input,
                                plumed_input[it_plumed_input],
                            )

                    # ### Write DATA file
                    write_list_of_strings_to_file(
                        local_path / subsys_lammps_data_fn, subsys_lammps_data
                    )

                    exploration_json["subsys_nr"][it_subsys_nr]["nb_steps"] = int(
                        subsys_nb_steps
                    )
                    exploration_json["subsys_nr"][it_subsys_nr][
                        "print_every_x_steps"
                    ] = int(subsys_print_every_x_steps)

                    # ###  Write INPUT file
                    for key, value in input_replace_dict.items():
                        it_subsys_lammps_in = replace_substring_in_list_of_strings(
                            it_subsys_lammps_in, key, value
                        )
                    write_list_of_strings_to_file(
                        local_path
                        / (f"{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}.in"),
                        it_subsys_lammps_in,
                    )

                    # ### Slurm file
                    job_email = read_key_input_json(
                        input_json,
                        new_input_json,
                        "job_email",
                        default_input_json,
                        step_name,
                        default_present,
                    )
                    slurm_file = replace_in_slurm_file_general(
                        slurm_file_master,
                        machine_spec,
                        subsys_walltime_approx_s,
                        machine_walltime_format,
                        job_email,
                    )

                    slurm_file = replace_substring_in_list_of_strings(
                        slurm_file,
                        "_R_DEEPMD_VERSION_",
                        f"{exploration_json['deepmd_model_version']}",
                    )
                    slurm_file = replace_substring_in_list_of_strings(
                        slurm_file,
                        "_R_MODEL_FILES_LIST_",
                        str(models_string.replace(" ", '" "')),
                    )
                    slurm_file = replace_substring_in_list_of_strings(
                        slurm_file,
                        "_R_INPUT_FILE_",
                        f"{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}",
                    )
                    slurm_file = replace_substring_in_list_of_strings(
                        slurm_file, "_R_DATA_FILE_", f"{subsys_lammps_data_fn}"
                    )
                    slurm_file = replace_substring_in_list_of_strings(
                        slurm_file, ' "_R_RERUN_FILE_"', ""
                    )
                    if plumed[0] == 1:
                        for n, it_plumed_input in enumerate(plumed_input):
                            if n == 0:
                                slurm_file = replace_substring_in_list_of_strings(
                                    slurm_file, "_R_PLUMED_FILES_LIST_", it_plumed_input
                                )
                            else:
                                slurm_file = replace_substring_in_list_of_strings(
                                    slurm_file,
                                    prev_plumed,
                                    prev_plumed + '" "' + it_plumed_input,
                                )
                            prev_plumed = it_plumed_input
                    else:
                        slurm_file = replace_substring_in_list_of_strings(
                            slurm_file, ' "_R_PLUMED_FILES_LIST_"', ""
                        )
                    write_list_of_strings_to_file(
                        local_path
                        / f"job_deepmd_{exploration_json['exploration_type']}_{machine_spec['arch_type']}_{machine}.sh",
                        slurm_file,
                    )
                    del it_subsys_lammps_in, job_email
                    del slurm_file
                # ### i-PI
                elif exploration_type == 1:
                    it_subsys_ipi_json = copy.deepcopy(subsys_ipi_json)
                    it_subsys_ipi_xml_aslist = copy.deepcopy(subsys_ipi_xml_aslist)
                    input_replace_dict[
                        "_R_SEED_"
                    ] = f"{it_nnp}{random.randrange(0, 1000)}{it_number}{previous_iteration_zfill}"
                    input_replace_dict[
                        "_R_SUBSYS_"
                    ] = f"{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}"
                    # ### Get data files (starting points) and number of steps
                    if current_iteration > 1:
                        if len(starting_points) == 0:
                            starting_point_list = copy.deepcopy(starting_points_bckp)
                        subsys_ipi_xyz_fn = starting_point_list[
                            random.randrange(0, len(starting_point_list))
                        ]
                        subsys_ipi_xyz = file_to_list_of_strings(
                            training_path / "starting_structures" / subsys_ipi_xyz_fn
                        )
                        input_replace_dict["_R_XYZ_"] = subsys_ipi_xyz_fn
                        it_subsys_ipi_json["coord_file"] = subsys_ipi_xyz_fn
                        for it_zzz, zzz in enumerate(config_json["type_map"]):
                            it_subsys_ipi_json["atom_type"][str(zzz)] = it_zzz
                        subsys_lammps_data = file_to_list_of_strings(
                            training_path
                            / "starting_structures"
                            / subsys_ipi_xyz_fn.replace(".xyz", ".lmp")
                        )
                        # ### Get again the subsys_cell and nb_atom
                        (
                            subsys_nb_atm,
                            num_atom_types,
                            box,
                            masses,
                            coords,
                        ) = parse_lammps_data(subsys_lammps_data)
                        subsys_cell = [
                            box[1] - box[0],
                            box[3] - box[2],
                            box[5] - box[4],
                        ]
                        input_replace_dict["_R_CELL_"] = f"{subsys_cell}"

                    # ### Plumed files
                    if plumed[0]:
                        input_replace_dict[
                            "_R_PLUMED_IN_"
                        ] = f"plumed_{it_subsys_nr}.dat"
                        for it_plumed_input in plumed_input:
                            plumed_input[
                                it_plumed_input
                            ] = replace_substring_in_list_of_strings(
                                plumed_input[it_plumed_input],
                                "_R_PRINT_FREQ_",
                                f"{int(subsys_print_every_x_steps)}",
                            )
                            # ### Because of weird units of time
                            plumed_input[
                                it_plumed_input
                            ] = replace_substring_in_list_of_strings(
                                plumed_input[it_plumed_input],
                                "UNITS LENGTH",
                                "UNITS TIME="
                                + str(2.4188843e-05 / subsys_timestep)
                                + " LENGTH",
                            )
                            write_list_of_strings_to_file(
                                local_path / it_plumed_input,
                                plumed_input[it_plumed_input],
                            )
                        del it_plumed_input

                    it_subsys_ipi_json["graph_file"] = models_list[0]

                    # ###  Write INPUT files
                    for key, value in input_replace_dict.items():
                        it_subsys_ipi_xml_aslist = replace_substring_in_list_of_strings(
                            it_subsys_ipi_xml_aslist, key, value
                        )
                    del key, value
                    it_subsys_ipi_xml = convert_list_of_strings_to_xml(
                        it_subsys_ipi_xml_aslist
                    )
                    write_xml(
                        it_subsys_ipi_xml,
                        local_path
                        / (f"{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}.xml"),
                    )
                    write_json_file(
                        it_subsys_ipi_json,
                        local_path
                        / (f"{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}.json"),
                    )

                    # ### Slurm file
                    job_email = read_key_input_json(
                        input_json,
                        new_input_json,
                        "job_email",
                        default_input_json,
                        step_name,
                        default_present,
                    )
                    slurm_file = replace_in_slurm_file_general(
                        slurm_file_master,
                        machine_spec,
                        subsys_walltime_approx_s,
                        machine_walltime_format,
                        job_email,
                    )

                    slurm_file = replace_substring_in_list_of_strings(
                        slurm_file,
                        "_R_DEEPMD_VERSION_",
                        f"{exploration_json['deepmd_model_version']}",
                    )
                    slurm_file = replace_substring_in_list_of_strings(
                        slurm_file, "_R_MODEL_FILES_LIST_", f"{models_list[0]}"
                    )
                    slurm_file = replace_substring_in_list_of_strings(
                        slurm_file,
                        "_R_INPUT_FILE_",
                        f"{it_subsys_nr}_{it_nnp}_{current_iteration_zfill}",
                    )
                    slurm_file = replace_substring_in_list_of_strings(
                        slurm_file, "_R_DATA_FILE_", f"{subsys_ipi_xyz_fn}"
                    )
                    if plumed[0] == 1:
                        for n, it_plumed_input in enumerate(plumed_input):
                            if n == 0:
                                slurm_file = replace_substring_in_list_of_strings(
                                    slurm_file, "_R_PLUMED_FILES_LIST_", it_plumed_input
                                )
                            else:
                                slurm_file = replace_substring_in_list_of_strings(
                                    slurm_file,
                                    prev_plumed,
                                    prev_plumed + '" "' + it_plumed_input,
                                )
                            prev_plumed = it_plumed_input
                        del n, it_plumed_input, prev_plumed
                    else:
                        slurm_file = replace_substring_in_list_of_strings(
                            slurm_file, ' "_R_PLUMED_FILES_LIST_"', ""
                        )
                    write_list_of_strings_to_file(
                        local_path
                        / f"job_deepmd_{exploration_json['exploration_type']}_{machine_spec['arch_type']}_{machine}.sh",
                        slurm_file,
                    )
                    del (
                        it_subsys_ipi_xml_aslist,
                        it_subsys_ipi_xml,
                        it_subsys_ipi_json,
                        job_email,
                    )
                    del slurm_file
                else:
                    error_msg = f"Exploration is unknown/not set."
                    logging.error(f"{error_msg}\nAborting...")
                    sys.exit(1)

            del it_number, models_list, models_string, local_path

        del it_nnp

        exploration_json["subsys_nr"][it_subsys_nr]["temperature_K"] = subsys_temp
        exploration_json["subsys_nr"][it_subsys_nr]["timestep_ps"] = subsys_timestep

        config_json["subsys_nr"][it_subsys_nr]["cell"] = subsys_cell
        config_json["subsys_nr"][it_subsys_nr]["nb_atm"] = subsys_nb_atm

        if plumed[0] == 1:
            del plumed_input, plumed
        del subsys_temp, subsys_cell, subsys_nb_atm, subsys_nb_steps
        del subsys_lammps_data, subsys_timestep, subsys_walltime_approx_s

    del it0_subsys_nr, it_subsys_nr, slurm_file_master

    exploration_json["is_locked"] = True
    exploration_json["is_launched"] = False
    exploration_json["is_checked"] = False
    if exploration_type == 1:
        exploration_json["is_unbeaded"] = False
        exploration_json["is_reruned"] = False
        exploration_json["is_rechecked"] = False
    exploration_json["is_deviated"] = False
    exploration_json["is_extracted"] = False
    del exploration_type

    write_json_file(config_json, (control_path / "config.json"))
    write_json_file(
        exploration_json, (control_path / f"exploration_{current_iteration_zfill}.json")
    )
    backup_and_overwrite_json_file(new_input_json, (current_path / input_fn))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "preparation",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
