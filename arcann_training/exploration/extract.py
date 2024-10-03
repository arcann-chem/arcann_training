"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/07/14
"""

# Standard library modules
import logging
import sys
from pathlib import Path
from copy import deepcopy
import subprocess

# Non-standard library imports
import numpy as np

# Local imports
from arcann_training.common.json import (
    load_json_file,
    write_json_file,
    get_key_in_dict,
    load_default_json_file,
    backup_and_overwrite_json_file,
)
from arcann_training.common.filesystem import (
    check_file_existence,
    remove_file,
    remove_files_matching_glob,
)
from arcann_training.common.list import (
    replace_substring_in_string_list,
    string_list_to_textfile,
    textfile_to_string_list,
)
from arcann_training.common.check import validate_step_folder, check_atomsk, check_vmd
from arcann_training.exploration.utils import (
    generate_input_exploration_disturbed_json,
    get_system_disturb,
)
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
    arcann_logger.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}."
    )
    arcann_logger.debug(f"Current path :{current_path}")
    arcann_logger.debug(f"Training path: {training_path}")
    arcann_logger.debug(f"Program path: {deepmd_iterative_path}")
    arcann_logger.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Load the default input JSON
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "assets" / "default_config.json"
    )[current_step]
    default_input_json_present = bool(default_input_json)
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

    # If the used input JSON is present, load it
    if (current_path / "used_input.json").is_file():
        current_input_json = load_json_file((current_path / "used_input.json"))
    else:
        arcann_logger.warning(f"No used_input.json found. Starting with empty one.")
        arcann_logger.warning(
            f"You should avoid this by not deleting the used_input.json file."
        )
        current_input_json = {}
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    # Get control path, load the main JSON and the exploration JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    exploration_json = load_json_file(
        (control_path / f"exploration_{padded_curr_iter}.json")
    )

    # Load the previous exploration JSON and training JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_training_json = load_json_file(
            (control_path / f"training_{padded_prev_iter}.json")
        )
        if prev_iter > 0:
            previous_exploration_json = load_json_file(
                (control_path / ("exploration_" + padded_prev_iter + ".json"))
            )
        else:
            previous_exploration_json = {}
    else:
        previous_training_json = {}
        previous_exploration_json = {}

    # Check if the atomsk and vmd package are installed
    if "atomsk_path" not in user_input_json:
        atomsk_bin = check_atomsk(
            get_key_in_dict(
                "atomsk_path",
                current_input_json,
                previous_exploration_json,
                default_input_json,
            )
        )
    else:
        atomsk_bin = check_atomsk(
            get_key_in_dict(
                "atomsk_path",
                user_input_json,
                previous_exploration_json,
                default_input_json,
            )
        )
        current_input_json["atomsk_path"] = atomsk_bin
    if "vmd_path" not in user_input_json:
        vmd_bin = check_vmd(
            get_key_in_dict(
                "vmd_path",
                current_input_json,
                previous_exploration_json,
                default_input_json,
            )
        )
    else:
        vmd_bin = check_vmd(
            get_key_in_dict(
                "vmd_path",
                user_input_json,
                previous_exploration_json,
                default_input_json,
            )
        )
        current_input_json["vmd_path"] = vmd_bin
    arcann_logger.debug(f"atomsk_bin: {atomsk_bin}")
    arcann_logger.debug(f"vmd_bin: {vmd_bin}")

    # Check if we can continue
    if not exploration_json["is_deviated"]:
        arcann_logger.error(f"Lock found. Execute first: exploration deviation.")
        arcann_logger.error(f"Aborting...")
        return 1

    # Update the current input JSON
    current_input_json["atomsk_path"] = atomsk_bin
    current_input_json["vmd_path"] = vmd_bin
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    # Update the current exploration JSON
    exploration_json["atomsk_path"] = atomsk_bin
    exploration_json["vmd_path"] = vmd_bin
    arcann_logger.debug(f"exploration_json: {exploration_json}")

    # Generate/update the merged input JSON
    # Priority: user > previous > default
    current_input_json = generate_input_exploration_disturbed_json(
        user_input_json,
        previous_exploration_json,
        default_input_json,
        current_input_json,
        main_json,
    )
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    check_file_existence(
        deepmd_iterative_path / "assets" / "others" / "vmd_dcd_selection_index.tcl"
    )
    master_vmd_tcl = textfile_to_string_list(
        deepmd_iterative_path / "assets" / "others" / "vmd_dcd_selection_index.tcl"
    )
    arcann_logger.debug(f"{master_vmd_tcl}")

    starting_structures_path = training_path / "starting_structures"
    starting_structures_path.mkdir(exist_ok=True)

    for system_auto_index, system_auto in enumerate(main_json["systems_auto"]):
        arcann_logger.info(
            f"Processing system: {system_auto} ({system_auto_index + 1}/{len(main_json['systems_auto'])})"
        )
        candidates_files = []
        candidates_disturbed_files = []

        print_every_x_steps = exploration_json["systems_auto"][system_auto][
            "print_every_x_steps"
        ]
        # Set the system params for disburbed selection
        (
            disturbed_start_value,
            disturbed_start_indexes,
            disturbed_candidate_value,
            disturbed_candidate_indexes,
        ) = get_system_disturb(current_input_json, system_auto_index)

        if (
            exploration_json["systems_auto"][system_auto]["exploration_type"]
            == "lammps"
        ):
            check_file_existence(training_path / "user_files" / f"{system_auto}.lmp")
            subprocess.run(
                [
                    atomsk_bin,
                    str(training_path / "user_files" / f"{system_auto}.lmp"),
                    "pdb",
                    str(training_path / "user_files" / system_auto),
                    "-ow",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            topo_file = training_path / "user_files" / f"{system_auto}.pdb"
        elif (
            exploration_json["systems_auto"][system_auto]["exploration_type"] == "i-PI"
        ):
            check_file_existence(training_path / "user_files" / f"{system_auto}.lmp")
            subprocess.run(
                [
                    atomsk_bin,
                    str(training_path / "user_files" / f"{system_auto}.lmp"),
                    "pdb",
                    str(training_path / "user_files" / system_auto),
                    "-ow",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            topo_file = training_path / "user_files" / f"{system_auto}.pdb"

        for it_nnp in range(1, main_json["nnp_count"] + 1):
            for it_number in range(
                1, exploration_json["systems_auto"][system_auto]["traj_count"] + 1
            ):
                arcann_logger.debug(f"{system_auto} / {it_nnp} / {it_number}")
                # Get the local path
                local_path = (
                    Path(".").resolve()
                    / str(system_auto)
                    / str(it_nnp)
                    / str(it_number).zfill(5)
                )
                QbC_stats = load_json_file(local_path / "QbC_stats.json", True, False)
                QbC_indexes = load_json_file(
                    local_path / "QbC_indexes.json", True, False
                )
                arcann_logger.debug(QbC_stats)

                if (local_path / "cell.txt").is_file():
                    cell_array = np.genfromtxt(local_path / "cell.txt")
                    cella = cell_array[:, 1] - cell_array[:, 0]
                    cellb = cell_array[:, 3] - cell_array[:, 2]
                    cellc = cell_array[:, 5] - cell_array[:, 4]
                    is_cell_constant = False
                    del cell_array
                else:
                    cella = main_json["systems_auto"][system_auto]["cell"][0]
                    cellb = main_json["systems_auto"][system_auto]["cell"][1]
                    cellc = main_json["systems_auto"][system_auto]["cell"][2]
                    is_cell_constant = True
                arcann_logger.debug(f"is_cell_constant: {is_cell_constant}")

                # Selection of the structure for the next iteration starting point
                if QbC_stats["minimum_index"] != -1:
                    if (
                        exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "lammps"
                    ):
                        traj_file = (
                            local_path
                            / f"{system_auto}_{it_nnp}_{padded_curr_iter}.dcd"
                        )
                        min_index = int(
                            QbC_stats["minimum_index"] / print_every_x_steps
                        )
                    elif (
                        exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "i-PI"
                    ):
                        traj_file = (
                            local_path
                            / f"{system_auto}_{it_nnp}_{padded_curr_iter}.dcd"
                        )
                        min_index = int(QbC_stats["minimum_index"])
                    elif (
                        exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "sander_emle"
                    ):
                        traj_file = (
                            local_path
                            / f"{system_auto}_{it_nnp}_{padded_curr_iter}_QM.xyz"
                        )
                        min_index = int(QbC_stats["minimum_index"])

                    (local_path / "min.vmd").write_text(f"{min_index}")
                    padded_min_index = str(min_index).zfill(5)

                    if (
                        exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "lammps"
                        or exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "i-PI"
                    ):

                        min_file_name = f"{padded_curr_iter}_{system_auto}_{it_nnp}_{str(it_number).zfill(5)}"
                        vmd_tcl = deepcopy(master_vmd_tcl)
                        vmd_tcl = replace_substring_in_string_list(
                            vmd_tcl, "_R_PDB_FILE_", str(topo_file)
                        )
                        vmd_tcl = replace_substring_in_string_list(
                            vmd_tcl, "_R_DCD_FILE_", str(traj_file)
                        )
                        vmd_tcl = replace_substring_in_string_list(
                            vmd_tcl, "_R_FRAME_INDEX_FILE_", str(local_path / "min.vmd")
                        )
                        vmd_tcl = replace_substring_in_string_list(
                            vmd_tcl,
                            "_R_XYZ_OUT_",
                            str(starting_structures_path / f"{min_file_name}"),
                        )
                        string_list_to_textfile(local_path / "vmd.tcl", vmd_tcl)
                        del vmd_tcl, traj_file

                        # VMD DCD -> XYZ
                        remove_file(
                            starting_structures_path
                            / f"{min_file_name}_{padded_min_index}.xyz"
                        )
                        subprocess.run(
                            [
                                vmd_bin,
                                "-e",
                                str(local_path / "vmd.tcl"),
                                "-dispdev",
                                "text",
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT,
                        )
                        xyz_string = textfile_to_string_list(
                            starting_structures_path
                            / f"{min_file_name}_{padded_min_index}.xyz"
                        )
                        if not is_cell_constant:
                            extended_xyz_header = f'Lattice="{cella[min_index]} 0.0000 0.0000 0.0000 {cellb[min_index]} 0.0000 0.0000 0.0000 {cellc[min_index]}" Properties=species:S:1:pos:R:3 Frame={min_index}'
                        else:
                            extended_xyz_header = f'Lattice="{cella} 0.0000 0.0000 0.0000 {cellb} 0.0000 0.0000 0.0000 {cellc}" Properties=species:S:1:pos:R:3 Frame={min_index}'
                        xyz_string = (
                            [xyz_string[0]] + [extended_xyz_header] + xyz_string[2:]
                        )
                        string_list_to_textfile(
                            starting_structures_path
                            / f"{min_file_name}_{padded_min_index}.xyz",
                            xyz_string,
                        )
                        del xyz_string, extended_xyz_header

                        remove_file((local_path / "vmd.tcl"))
                        remove_file((local_path / "min.vmd"))

                        # Atomsk XYZ -> LMP
                        remove_file(
                            starting_structures_path
                            / f"{min_file_name}_{padded_min_index}.lmp"
                        )
                        subprocess.run(
                            [
                                atomsk_bin,
                                "-ow",
                                "-properties",
                                str(Path("..") / "user_files" / "properties.txt"),
                                str(
                                    Path("..")
                                    / "starting_structures"
                                    / f"{min_file_name}_{padded_min_index}.xyz"
                                ),
                                str(
                                    Path("..")
                                    / "starting_structures"
                                    / f"{min_file_name}_{padded_min_index}.lmp"
                                ),
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT,
                        )
                        # Add a mass to any 0.0000 mass in the LMP file
                        lmp_file = textfile_to_string_list(
                            starting_structures_path
                            / f"{min_file_name}_{padded_min_index}.lmp"
                        )
                        lmp_file = replace_substring_in_string_list(
                            lmp_file,
                            "0.00000000              # XX",
                            "1.00000000              # XX",
                        )
                        string_list_to_textfile(
                            starting_structures_path
                            / f"{min_file_name}_{padded_min_index}.lmp",
                            lmp_file,
                        )
                        del lmp_file

                        # If the a minium value was set by the user or previous, enable disturbed min structures
                        if disturbed_start_value != 0:
                            # or (curr_iter > 1 and previous_exploration_json["systems_auto"][system_auto]["disturbed_start"]):

                            # Atomsk XYZ ==> XYZ_disturbed
                            remove_file(
                                (
                                    starting_structures_path
                                    / f"{min_file_name}_{padded_min_index}_disturbed.xyz"
                                )
                            )
                            (
                                starting_structures_path
                                / f"{min_file_name}_{padded_min_index}_disturbed.xyz"
                            ).write_text(
                                (
                                    starting_structures_path
                                    / f"{min_file_name}_{padded_min_index}.xyz"
                                ).read_text()
                            )

                            if not disturbed_start_indexes:
                                subprocess.run(
                                    [
                                        atomsk_bin,
                                        "-ow",
                                        str(
                                            Path("..")
                                            / "starting_structures"
                                            / f"{min_file_name}_{padded_min_index}_disturbed.xyz"
                                        ),
                                        "-disturb",
                                        str(disturbed_start_value),
                                        "exyz",
                                    ],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT,
                                )
                            else:
                                subprocess.run(
                                    [
                                        atomsk_bin,
                                        "-ow",
                                        str(
                                            Path("..")
                                            / "starting_structures"
                                            / f"{min_file_name}_{padded_min_index}_disturbed.xyz"
                                        ),
                                        "-select",
                                        ",".join(
                                            [
                                                str(idx)
                                                for idx in disturbed_start_indexes
                                            ]
                                        ),
                                        "-disturb",
                                        str(disturbed_start_value),
                                        "exyz",
                                    ],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT,
                                )

                            # Atomsk XYZ -> LMP
                            remove_file(
                                starting_structures_path
                                / f"{min_file_name}_{padded_min_index}_disturbed.lmp"
                            )
                            subprocess.run(
                                [
                                    atomsk_bin,
                                    "-ow",
                                    "-properties",
                                    str(Path("..") / "user_files" / "properties.txt"),
                                    str(
                                        Path("..")
                                        / "starting_structures"
                                        / f"{min_file_name}_{padded_min_index}_disturbed.xyz"
                                    ),
                                    str(
                                        Path("..")
                                        / "starting_structures"
                                        / f"{min_file_name}_{padded_min_index}_disturbed.lmp"
                                    ),
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT,
                            )
                            # Add a mass to any 0.0000 mass in the LMP file
                            lmp_file = textfile_to_string_list(
                                starting_structures_path
                                / f"{min_file_name}_{padded_min_index}_disturbed.lmp"
                            )
                            lmp_file = replace_substring_in_string_list(
                                lmp_file,
                                "0.00000000              # XX",
                                "1.00000000              # XX",
                            )
                            string_list_to_textfile(
                                starting_structures_path
                                / f"{min_file_name}_{padded_min_index}_disturbed.lmp",
                                lmp_file,
                            )
                            del lmp_file

                            exploration_json["systems_auto"][system_auto][
                                "disturbed_start_value"
                            ] = disturbed_start_value
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_start_indexes"
                            ] = disturbed_start_indexes
                        else:
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_start_value"
                            ] = 0
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_start_indexes"
                            ] = []

                        del min_index, padded_min_index, min_file_name

                    elif (
                        exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "sander_emle"
                    ):

                        remove_file((local_path / "min.vmd"))
                        # This part should read the nc file and convert it
                        # But for now, it is deactivated

                        if disturbed_start_value != 0:
                            arcann_logger.warning(
                                "Disturbed start value is not supported for sander_emle"
                            )
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_start_value"
                            ] = 0
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_start_indexes"
                            ] = []

                # Selection of labeling XYZ
                if QbC_stats["selected_count"] > 0:
                    candidate_indexes = np.array(QbC_indexes["selected_indexes"])

                    if (
                        exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "lammps"
                    ):
                        traj_file = (
                            local_path
                            / f"{system_auto}_{it_nnp}_{padded_curr_iter}.dcd"
                        )
                        candidate_indexes = candidate_indexes / print_every_x_steps
                    elif (
                        exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "i-PI"
                    ):
                        traj_file = (
                            local_path
                            / f"{system_auto}_{it_nnp}_{padded_curr_iter}.dcd"
                        )
                        candidate_indexes = candidate_indexes
                    elif (
                        exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "sander_emle"
                    ):
                        traj_file = (
                            local_path
                            / f"{system_auto}_{it_nnp}_{padded_curr_iter}_QM.xyz"
                        )

                    candidate_indexes = (
                        candidate_indexes.astype(int).astype(str).tolist()
                    )

                    if (
                        exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "lammps"
                        or exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "i-PI"
                    ):
                        string_list_to_textfile(
                            (local_path / "label.vmd"), candidate_indexes
                        )

                        vmd_tcl = deepcopy(master_vmd_tcl)
                        vmd_tcl = replace_substring_in_string_list(
                            vmd_tcl, "_R_PDB_FILE_", str(topo_file)
                        )
                        vmd_tcl = replace_substring_in_string_list(
                            vmd_tcl, "_R_DCD_FILE_", str(traj_file)
                        )
                        vmd_tcl = replace_substring_in_string_list(
                            vmd_tcl,
                            "_R_FRAME_INDEX_FILE_",
                            str(local_path / "label.vmd"),
                        )
                        vmd_tcl = replace_substring_in_string_list(
                            vmd_tcl, "_R_XYZ_OUT_", str(local_path / "candidates")
                        )
                        string_list_to_textfile((local_path / "vmd.tcl"), vmd_tcl)
                        del vmd_tcl, traj_file

                        # VMD DCD ->  A lot of XYZ
                        subprocess.run(
                            [
                                vmd_bin,
                                "-e",
                                str(local_path / "vmd.tcl"),
                                "-dispdev",
                                "text",
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT,
                        )
                        for xyz_files in local_path.glob("candidates_*.xyz"):
                            if "disturbed" in xyz_files.stem:
                                continue
                            index_xyz = int(xyz_files.stem.split("_")[-1])
                            xyz_string = textfile_to_string_list(xyz_files)
                            if not is_cell_constant:
                                extended_xyz_header = f'Lattice="{cella[index_xyz]} 0.0000 0.0000 0.0000 {cellb[index_xyz]} 0.0000 0.0000 0.0000 {cellc[index_xyz]}" Properties=species:S:1:pos:R:3 Frame={index_xyz}'
                            else:
                                extended_xyz_header = f'Lattice="{cella} 0.0000 0.0000 0.0000 {cellb} 0.0000 0.0000 0.0000 {cellc}" Properties=species:S:1:pos:R:3 Frame={index_xyz}'
                            xyz_string = (
                                [xyz_string[0]] + [extended_xyz_header] + xyz_string[2:]
                            )
                            string_list_to_textfile(xyz_files, xyz_string)
                            del xyz_string, extended_xyz_header, index_xyz
                        del xyz_files

                        remove_file((local_path / "label.vmd"))
                        remove_file((local_path / "vmd.tcl"))

                        candidate_indexes_padded = [
                            _.zfill(5) for _ in candidate_indexes
                        ]
                        candidates_files.extend(
                            [
                                str(
                                    Path(".")
                                    / str(system_auto)
                                    / str(it_nnp)
                                    / str(it_number).zfill(5)
                                    / ("candidates_" + _ + ".xyz")
                                )
                                for _ in candidate_indexes_padded
                            ]
                        )

                        # If the a minium value was set by the user or previous, enable disturbed min structures
                        if disturbed_candidate_value != 0:
                            remove_files_matching_glob(local_path, "_disturbed.xyz")
                            for candidate_index_padded in candidate_indexes_padded:
                                (
                                    local_path
                                    / f"candidates_{candidate_index_padded}_disturbed.xyz"
                                ).write_text(
                                    (
                                        local_path
                                        / f"candidates_{candidate_index_padded}.xyz"
                                    ).read_text()
                                )
                                if not disturbed_candidate_indexes:
                                    subprocess.run(
                                        [
                                            atomsk_bin,
                                            "-ow",
                                            str(
                                                Path(".")
                                                / str(system_auto)
                                                / str(it_nnp)
                                                / str(it_number).zfill(5)
                                                / f"candidates_{candidate_index_padded}_disturbed.xyz"
                                            ),
                                            "-disturb",
                                            str(disturbed_candidate_value),
                                            "exyz",
                                        ],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.STDOUT,
                                    )
                                else:
                                    subprocess.run(
                                        [
                                            atomsk_bin,
                                            "-ow",
                                            str(
                                                Path(".")
                                                / str(system_auto)
                                                / str(it_nnp)
                                                / str(it_number).zfill(5)
                                                / f"candidates_{candidate_index_padded}_disturbed.xyz"
                                            ),
                                            "-select",
                                            ",".join(
                                                [
                                                    str(idx)
                                                    for idx in disturbed_candidate_indexes
                                                ]
                                            ),
                                            "-disturb",
                                            str(disturbed_candidate_value),
                                            "exyz",
                                        ],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.STDOUT,
                                    )

                            candidates_disturbed_files.extend(
                                [
                                    str(
                                        Path(".")
                                        / str(system_auto)
                                        / str(it_nnp)
                                        / str(it_number).zfill(5)
                                        / ("candidates_" + _ + "_disturbed.xyz")
                                    )
                                    for _ in candidate_indexes_padded
                                ]
                            )
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_candidate_value"
                            ] = disturbed_candidate_value
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_candidate_indexes"
                            ] = disturbed_candidate_indexes
                            del candidate_index_padded

                        else:
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_candidate_value"
                            ] = 0
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_candidate_indexes"
                            ] = []
                        del candidate_indexes, candidate_indexes_padded

                    elif (
                        exploration_json["systems_auto"][system_auto][
                            "exploration_type"
                        ]
                        == "sander_emle"
                    ):

                        candidate_indexes_padded = [
                            _.zfill(5) for _ in candidate_indexes
                        ]
                        (
                            atom_counts,
                            atomic_symbols,
                            atomic_coordinates,
                            comments,
                            lattice_info,
                            pbc_info,
                            properties_info,
                            max_f_std_info,
                        ) = parse_xyz_trajectory_file(local_path / traj_file)

                        for _ in candidate_indexes_padded:
                            print(
                                f"Processing candidate: {system_auto} / {it_nnp} / {it_number} / {_}"
                            )
                            write_xyz_frame(
                                local_path / f"candidates_{_}.xyz",
                                int(_),
                                atom_counts,
                                atomic_symbols,
                                atomic_coordinates,
                                np.array([]),
                                comments,
                            )
                            candidates_files.append(
                                str(
                                    Path(".")
                                    / str(system_auto)
                                    / str(it_nnp)
                                    / str(it_number).zfill(5)
                                    / ("candidates_" + _ + ".xyz")
                                )
                            )

                        if disturbed_candidate_value != 0:
                            arcann_logger.warning(
                                "Disturbed start value is not supported for sander_emle"
                            )
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_candidate_value"
                            ] = 0
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_candidate_indexes"
                            ] = []
                        else:
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_candidate_value"
                            ] = 0
                            exploration_json["systems_auto"][system_auto][
                                "disturbed_candidate_indexes"
                            ] = []
                else:
                    exploration_json["systems_auto"][system_auto][
                        "disturbed_candidate_value"
                    ] = 0
                    exploration_json["systems_auto"][system_auto][
                        "disturbed_candidate_indexes"
                    ] = []

            del (
                cella,
                cellb,
                cellc,
                is_cell_constant,
                local_path,
                QbC_stats,
                QbC_indexes,
            )

        del it_nnp, it_number

        if candidates_files:
            candidates_xyz_file = (
                current_path
                / system_auto
                / f"candidates_{padded_curr_iter}_{system_auto}.xyz"
            )
            remove_file(candidates_xyz_file)
            with open(candidates_xyz_file, "w") as f:
                for candidate_xyz_file in candidates_files:
                    f.write((current_path / candidate_xyz_file).read_text())
                    remove_file((current_path / candidate_xyz_file))
                del candidate_xyz_file
            del candidates_xyz_file, f
        del candidates_files

        if candidates_disturbed_files:
            candidates_disturbed_xyz_file = (
                current_path
                / system_auto
                / f"candidates_{padded_curr_iter}_{system_auto}_disturbed.xyz"
            )
            remove_file(candidates_disturbed_xyz_file)
            with open(candidates_disturbed_xyz_file, "w") as f:
                for candidate_disturbed_file in candidates_disturbed_files:
                    f.write((current_path / candidate_disturbed_file).read_text())
                    remove_file((current_path / candidate_disturbed_file))
                del candidate_disturbed_file
            del candidates_disturbed_xyz_file, f
        del candidates_disturbed_files
        arcann_logger.info(
            f"Processed system: {system_auto} ({system_auto_index + 1}/{len(main_json['systems_auto'])})"
        )

    del (
        disturbed_start_value,
        disturbed_start_indexes,
        disturbed_candidate_value,
        disturbed_candidate_indexes,
        print_every_x_steps,
    )
    del system_auto_index, system_auto, master_vmd_tcl
    del starting_structures_path

    arcann_logger.info(f"-" * 88)
    # Update the booleans in the exploration JSON
    exploration_json["is_extracted"] = True

    # Dump the JSON files (exploration and merged input)
    write_json_file(
        exploration_json, (control_path / f"exploration_{padded_curr_iter}.json")
    )
    backup_and_overwrite_json_file(
        current_input_json, (current_path / "used_input.json"), read_only=True
    )

    # End
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(
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
        current_input_json,
        exploration_json,
        previous_training_json,
        previous_exploration_json,
    )
    del curr_iter, padded_curr_iter, prev_iter, padded_prev_iter
    del atomsk_bin, vmd_bin

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "extract",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
