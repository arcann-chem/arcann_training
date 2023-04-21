"""
Created: 2023/01/01
Last modified: 2023/04/17
"""
from pathlib import Path
import logging
import copy
import sys
import subprocess

# Non-standard library imports
import numpy as np

# deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
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
from deepmd_iterative.common.json_parameters import (
    get_machine_keyword,
    get_key_in_dict,
)

def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine = None,
    user_config_filename: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Load the default config (JSON)
    default_config = load_default_json_file(deepmd_iterative_path / "assets" / "default_config.json")[current_step]
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

    # Get control path, config JSON and exploration JSON
    control_path = training_path / "control"
    main_config = load_json_file((control_path / "config.json"))
    exploration_config = load_json_file(
        (control_path / f"exploration_{padded_curr_iter}.json")
    )

    # Load the previous training JSON and previous exploration JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        prev_training_config = load_json_file((control_path / f"training_{padded_prev_iter}.json"))
        if prev_iter > 0:
            prev_exploration_config = load_json_file((control_path / ("exploration_" + padded_prev_iter + ".json")))
        else:
            prev_exploration_config = {}
    else:
        prev_training_config = {}
        prev_exploration_config = {}


    # Check if the atomsk package is installed
    atomsk_bin = check_atomsk(get_key_in_dict("atomsk_path", user_config, prev_exploration_config, default_config))
    vmd_bin = check_vmd(get_key_in_dict("vmd_path", user_config, prev_exploration_config, default_config))

    # Update new input
    current_config["atomsk_path"] = atomsk_bin
    current_config["vmd_path"] = vmd_bin
    logging.debug(f"current_config: {current_config}")

    # Checks
    if not exploration_config["is_deviated"]:
        logging.error(f"Lock found. Execute first: exploration deviation.")
        logging.error(f"Aborting...")
        return 1

    check_file_existence(deepmd_iterative_path / "assets" / "others" / "vmd_dcd_selection_index.tcl")
    master_vmd_tcl = textfile_to_string_list(deepmd_iterative_path / "assets" / "others" / "vmd_dcd_selection_index.tcl")
    logging.debug(f"{master_vmd_tcl}")

    starting_structures_path = training_path / "starting_structures"
    starting_structures_path.mkdir(exist_ok=True)

    for it0_subsys_nr, it_subsys_nr in enumerate(main_config["subsys_nr"]):
        print_every_x_steps = exploration_config["subsys_nr"][it_subsys_nr]["print_every_x_steps"]
        if exploration_config["exploration_type"] == "lammps":
            check_file_existence(training_path / "files" / f"{it_subsys_nr}.lmp")
            subprocess.run(
                    [
                        atomsk_bin,
                        str(training_path / "files" / f"{it_subsys_nr}.lmp"),
                        "pdb",
                        str(training_path / "files" / it_subsys_nr),
                        "-ow",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            topo_file = training_path / "files" / f"{it_subsys_nr}.pdb"
        elif exploration_config["exploration_type"] == "i-PI":
            check_file_existence(training_path / "files" / f"{it_subsys_nr}.lmp")
            subprocess.run(
                [
                    atomsk_bin,
                    str(training_path / "files" / f"{it_subsys_nr}.lmp"),
                    "pdb",
                    str(training_path / "files" / it_subsys_nr),
                    "-ow",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                )
            topo_file = training_path / "files" / f"{it_subsys_nr}.pdb"

        for it_nnp in range(1, main_config["nnp_count"] + 1):
            for it_number in range(1, exploration_config["traj_count"] + 1):

                logging.debug(f"{it_subsys_nr} / {it_nnp} / {it_number}")
                # Get the local path
                local_path = (
                    Path(".").resolve()
                    / str(it_subsys_nr)
                    / str(it_nnp)
                    / str(it_number).zfill(5)
                )
                QbC_stats = load_json_file(local_path/"QbC_stats.json",True,False)
                QbC_indexes = load_json_file(local_path/"QbC_indexes.json",True,False)
                logging.debug(QbC_stats)
                # Selection of the min for the next iteration starting point
                if QbC_stats["minimum_index"] != -1:
                    if exploration_config["exploration_type"] == "lammps":
                        traj_file = local_path/f"{it_subsys_nr}_{it_nnp}_{padded_curr_iter}.dcd"
                        min_index = int(QbC_stats["minimum_index"] / print_every_x_steps)
                    elif exploration_config["exploration_type"] == "i-PI":
                        traj_file = local_path/f"{it_subsys_nr}_{it_nnp}_{padded_curr_iter}.dcd"
                        min_index = int(QbC_stats["minimum_index"] )
                    (local_path/"min.vmd").write_text(f"{min_index}")
                    padded_min_index = str(min_index).zfill(5)

                    min_file_name = f"{padded_curr_iter}_{it_subsys_nr}_{it_nnp}_{str(it_number).zfill(5)}"
                    vmd_tcl = copy.deepcopy(master_vmd_tcl)
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl,"_R_PDB_FILE_",str(topo_file))
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl,"_R_DCD_FILE_",str(traj_file))
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl,"_R_FRAME_INDEX_FILE_",str(local_path/"min.vmd"))
                    vmd_tcl = replace_substring_in_string_list(vmd_tcl,"_R_XYZ_OUT_",str(starting_structures_path / f"{min_file_name}"))
                    string_list_to_textfile(local_path / "vmd.tcl", vmd_tcl)
                    del vmd_tcl, traj_file

                    # VMD DCD -> XYZ
                    remove_file(starting_structures_path / f"{min_file_name}.xyz")
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
                    remove_file(starting_structures_path / f"{min_file_name}.lmp")
                    subprocess.run(
                        [
                            atomsk_bin,
                            "-ow",
                            str(Path("..") / "starting_structures" / f"{min_file_name}_{padded_min_index}.xyz"),
                            "-cell", "set", str(main_config["subsys_nr"][it_subsys_nr]["cell"][0]), "H1",\
                            "-cell", "set", str(main_config["subsys_nr"][it_subsys_nr]["cell"][1]), "H2",\
                            "-cell", "set", str(main_config["subsys_nr"][it_subsys_nr]["cell"][2]), "H3",\
                            str(Path("..") / "starting_structures" / f"{min_file_name}_{padded_min_index}.lmp"),
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )




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
