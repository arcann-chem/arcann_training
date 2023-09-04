"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/09/04
"""
# Standard library modules
import copy
import logging
import subprocess
import sys
from pathlib import Path

# Non-standard library imports
import numpy as np

# Local imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    get_key_in_dict,
    load_default_json_file,
    backup_and_overwrite_json_file,
)
from deepmd_iterative.common.filesystem import (
    check_file_existence,
    remove_file
)
from deepmd_iterative.common.list import (
    replace_substring_in_string_list,
    string_list_to_textfile,
    textfile_to_string_list,
)
from deepmd_iterative.common.check import validate_step_folder, check_atomsk, check_vmd
from deepmd_iterative.exploration.utils import generate_input_exploration_disturbed_json, get_system_disturb

def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine = None,
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

    # Make a deepcopy of it to create the current input JSON
    merged_input_json = copy.deepcopy(user_input_json)

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
        previous_training_json = load_json_file((control_path / f"training_{padded_prev_iter}.json"))
        if prev_iter > 0:
            previous_exploration_json = load_json_file((control_path / ("exploration_" + padded_prev_iter + ".json")))
        else:
            previous_exploration_json = {}
    else:
        previous_training_json = {}
        previous_exploration_json = {}

    # Check if the atomsk and vmd package are installed
    atomsk_bin = check_atomsk(get_key_in_dict("atomsk_path", user_input_json, previous_exploration_json, default_input_json))
    vmd_bin = check_vmd(get_key_in_dict("vmd_path", user_input_json, previous_exploration_json, default_input_json))

    # Check if we can continue
    if not exploration_json["is_deviated"]:
        logging.error(f"Lock found. Execute first: exploration deviation.")
        logging.error(f"Aborting...")
        return 1

    # Update the current input JSON
    merged_input_json["atomsk_path"] = atomsk_bin
    merged_input_json["vmd_path"] = vmd_bin
    logging.debug(f"merged_input_json: {merged_input_json}")

    # Generate/update the merged input JSON
    # Priority: user > previous > default
    merged_input_json = generate_input_exploration_disturbed_json(
        user_input_json,
        previous_exploration_json,
        default_input_json,
        merged_input_json,
        main_json,
    )
    logging.debug(f"merged_input_json: {merged_input_json}")

    check_file_existence(deepmd_iterative_path / "assets" / "others" / "vmd_dcd_selection_index.tcl")
    master_vmd_tcl = textfile_to_string_list(deepmd_iterative_path / "assets" / "others" / "vmd_dcd_selection_index.tcl")
    logging.debug(f"{master_vmd_tcl}")

    starting_structures_path = training_path / "starting_structures"
    starting_structures_path.mkdir(exist_ok=True)

    for system_auto_index, system_auto in enumerate(main_json["systems_auto"]):
        logging.info(f"Processing system: {system_auto} ({system_auto_index + 1}/{len(main_json['systems_auto'])})")
        candidates_files = []

        print_every_x_steps = exploration_json["systems_auto"][system_auto]["print_every_x_steps"]
        # Set the system params for disburbed selection
        (
            disturbed_start_value,
            disturbed_start_indexes,
            disturbed_candidate_value,
            disturbed_candidate_indexes
        ) = get_system_disturb(merged_input_json, system_auto_index)


        if exploration_json["exploration_type"] == "lammps":
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
        elif exploration_json["exploration_type"] == "i-PI":
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
            for it_number in range(1, exploration_json["traj_count"] + 1):

                logging.debug(f"{system_auto} / {it_nnp} / {it_number}")
                # Get the local path
                local_path = (
                    Path(".").resolve()
                    / str(system_auto)
                    / str(it_nnp)
                    / str(it_number).zfill(5)
                )
                QbC_stats = load_json_file(local_path/"QbC_stats.json", True, False)
                QbC_indexes = load_json_file(local_path/"QbC_indexes.json", True, False)
                logging.debug(QbC_stats)

                # Selection of the structure for the next iteration starting point
                if QbC_stats["minimum_index"] != -1:
                    if exploration_json["exploration_type"] == "lammps":
                        traj_file = local_path/f"{system_auto}_{it_nnp}_{padded_curr_iter}.dcd"
                        min_index = int(QbC_stats["minimum_index"] / print_every_x_steps)
                    elif exploration_json["exploration_type"] == "i-PI":
                        traj_file = local_path/f"{system_auto}_{it_nnp}_{padded_curr_iter}.dcd"
                        min_index = int(QbC_stats["minimum_index"] )
                    (local_path/"min.vmd").write_text(f"{min_index}")
                    padded_min_index = str(min_index).zfill(5)

                    min_file_name = f"{padded_curr_iter}_{system_auto}_{it_nnp}_{str(it_number).zfill(5)}"
                    vmd_tcl = copy.deepcopy(master_vmd_tcl)
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl, "_R_PDB_FILE_",str(topo_file))
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl, "_R_DCD_FILE_",str(traj_file))
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl, "_R_FRAME_INDEX_FILE_",str(local_path/"min.vmd"))
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl ,"_R_XYZ_OUT_",str(starting_structures_path / f"{min_file_name}"))
                    string_list_to_textfile(local_path / "vmd.tcl", vmd_tcl)
                    del vmd_tcl, traj_file

                    # VMD DCD -> XYZ
                    remove_file(starting_structures_path / f"{min_file_name}_{padded_min_index}.xyz")
                    subprocess.run(
                        [
                            vmd_bin,
                            "-e",
                            str(local_path/"vmd.tcl"),
                            "-dispdev",
                            "text"
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        )
                    remove_file((local_path / "vmd.tcl"))
                    remove_file((local_path / "min.vmd"))

                    # Atomsk XYZ -> LMP
                    remove_file(starting_structures_path / f"{min_file_name}_{padded_min_index}.lmp")
                    subprocess.run(
                        [
                            atomsk_bin,
                            "-ow",
                            str(Path("..") / "starting_structures" / f"{min_file_name}_{padded_min_index}.xyz"),
                            "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][0]), "H1",
                            "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][1]), "H2",
                            "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][2]), "H3",
                            str(Path("..") / "starting_structures" / f"{min_file_name}_{padded_min_index}.lmp"),
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )

                    # If the a minium value was set by the user or previous, enable disturbed min structures
                    if (disturbed_start_value != 0): \
                        # or (curr_iter > 1 and previous_exploration_json["systems_auto"][system_auto]["disturbed_start"]):

                        # Atomsk XYZ ==> XYZ_disturbed
                        remove_file((starting_structures_path/f"{min_file_name}_{padded_min_index}_disturbed.xyz"))
                        (starting_structures_path/f"{min_file_name}_{padded_min_index}_disturbed.xyz").write_text((starting_structures_path/f"{min_file_name}_{padded_min_index}.xyz").read_text())

                        if not disturbed_start_indexes :
                            subprocess.run(
                                [
                                    atomsk_bin,
                                    "-ow",
                                    str(Path("..") / "starting_structures" / f"{min_file_name}_{padded_min_index}_disturbed.xyz"),
                                    "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][0]), "H1",
                                    "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][1]), "H2",
                                    "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][2]), "H3",
                                    "-disturb", str(disturbed_start_value),
                                    "xyz"
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT
                            )
                        else:
                            subprocess.run(
                                [
                                    atomsk_bin,
                                    "-ow",
                                    str(Path("..") / "starting_structures" / f"{min_file_name}_{padded_min_index}_disturbed.xyz"),
                                    "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][0]), "H1",
                                    "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][1]), "H2",
                                    "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][2]), "H3",
                                    "-select", ",".join([str(idx) for idx in disturbed_start_indexes]),
                                    "-disturb", str(disturbed_start_value),
                                    "xyz"
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT
                            )

                        # Atomsk XYZ -> LMP
                        remove_file(starting_structures_path / f"{min_file_name}_{padded_min_index}_disturbed.lmp")
                        subprocess.run(
                            [
                                atomsk_bin,
                                "-ow",
                                str(Path("..") / "starting_structures" / f"{min_file_name}_{padded_min_index}_disturbed.xyz"),
                                "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][0]), "H1",
                                "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][1]), "H2",
                                "-cell", "set", str(main_json["systems_auto"][system_auto]["cell"][2]), "H3",
                                str(Path("..") / "starting_structures" / f"{min_file_name}_{padded_min_index}_disturbed.lmp"),
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT
                        )

                    del min_index, padded_min_index, min_file_name

                # Selection of labeling XYZ
                if QbC_stats["kept_count"] > 0:

                    candidate_indexes = np.array(QbC_indexes["candidate_indexes"])
                    if exploration_json["exploration_type"] == "lammps":
                        traj_file = local_path/f"{system_auto}_{it_nnp}_{padded_curr_iter}.dcd"
                        candidate_indexes = candidate_indexes / print_every_x_steps
                    elif exploration_json["exploration_type"] == "i-PI":
                        traj_file = local_path/f"{system_auto}_{it_nnp}_{padded_curr_iter}.dcd"
                        candidate_indexes = candidate_indexes

                    candidate_indexes = candidate_indexes.astype(int).astype(str).tolist()
                    string_list_to_textfile((local_path/"label.vmd"), candidate_indexes)

                    vmd_tcl = copy.deepcopy(master_vmd_tcl)
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl,"_R_PDB_FILE_", str(topo_file))
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl,"_R_DCD_FILE_", str(traj_file))
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl,"_R_FRAME_INDEX_FILE_", str(local_path/"label.vmd"))
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl,"_R_XYZ_OUT_", str(local_path / ("candidates")))
                    string_list_to_textfile((local_path/"vmd.tcl"), vmd_tcl)
                    del vmd_tcl, traj_file

                    # VMD DCD ->  A lot of XYZ
                    subprocess.run(
                        [
                            vmd_bin,
                            "-e",
                            str(local_path/"vmd.tcl"),
                            "-dispdev",
                            "text"
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        )

                    remove_file((local_path / "label.vmd"))
                    remove_file((local_path / "vmd.tcl"))

                    candidate_indexes_padded = [ _.zfill(5) for _ in candidate_indexes]
                    candidates_files.extend([str( Path(".") / str(system_auto) / str(it_nnp) / str(it_number).zfill(5) / ("candidates_"+_+".xyz") ) for _ in candidate_indexes_padded])
                    
                    
                    # Here disturbed:

        string_list_to_textfile((current_path / "gather.atomsk"), candidates_files)
        subprocess.run(
            [
                atomsk_bin,
                "-ow",
                "--gather",
                str(Path(".") / "gather.atomsk"),
                str(Path(".") / system_auto / f"candidates_{padded_curr_iter}_{system_auto}.xyz")
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        remove_file((current_path / "gather.atomsk"))
        for  _ in candidates_files:
            remove_file((current_path / _ ))

        logging.info(f"Processed system: {system_auto} ({system_auto_index + 1}/{len(main_json['systems_auto'])})")

    return 0

if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "exploration",
            "extract",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
