from pathlib import Path
import logging
import sys

### Non-standard imports
import numpy as np


def import_xyz(file_path: Path):
    """_summary_

    Args:
        file_path (Path): An xyz file

    Returns:
        traj_nb_atoms (np.ndarray): Array number of atoms with dim(nb_step)
        traj_atm_symbol (np.ndarray): Array atomic symbol with dim(nb_step, nb_atoms)
        traj_atm_coords (np.ndarray): Array of the number of atoms with dim(nb_step, nb_atoms, 3)
    """
    if not file_path.is_file():
        logging.critical(f"File not found: {file_path.name} not in {file_path.parent}")
        logging.critical(f"Aborting...")
        sys.exit(1)

    with file_path.open("r") as xyz:
        step_nb_atoms = 0
        step_atm_symbol = []
        step_atm_coords = []

        traj_nb_atoms = []
        traj_atm_symbol = []
        traj_atm_coords = []
        empty = []
        s = 0
        current_line = xyz.readline()
        while len(current_line) > 1:
            print(len(current_line))
            step_nb_atoms = int(current_line)
            traj_nb_atoms.append(step_nb_atoms)
            if step_nb_atoms != traj_nb_atoms[0]:
                logging.critical(
                    f"This function doesn't work on variable number of atoms xyz"
                )
                logging.critical(f"Aborting...")
                sys.exit(1)

            empty.append(xyz.readline())
            for ii in range(1, traj_nb_atoms[s] + 1):
                current_line = xyz.readline()
                atm_symbol, atm_x, atm_y, atm_z = current_line.split()
                step_atm_symbol.append(atm_symbol)
                step_atm_coords.append([float(atm_x), float(atm_y), float(atm_z)])
                if ii >= traj_nb_atoms[s]:
                    traj_atm_symbol.append(step_atm_symbol)
                    traj_atm_coords.append(step_atm_coords)
                    step_atm_symbol = []
                    step_atm_coords = []
                    s = s + 1

            current_line = xyz.readline()
        traj_nb_atoms = np.array(traj_nb_atoms)
        traj_atm_coords = np.array(traj_atm_coords)
        traj_atm_symbol = np.array(traj_atm_symbol)
        return traj_nb_atoms, traj_atm_symbol, traj_atm_coords


def write_xyz_from_index(
        file_path: Path,
        idx: int,
        traj_nb_atoms: np.ndarray,
        traj_atm_coords: np.ndarray,
        traj_atm_symbol: np.ndarray,
):
    """_summary_

    Args:
        file_path (Path): _description_
        idx (int): _description_
        traj_nb_atoms (np.ndarray): _description_
        traj_atm_coords (np.ndarray): _description_
        traj_atm_symbol (np.ndarray): _description_
    """
    with file_path.open("w") as xyz:
        xyz.write(f"{traj_nb_atoms[idx]}\n")
        xyz.write(f"Index: {idx}\n")
        x = 0
        for ii in range(traj_nb_atoms[idx]):
            xyz.write(
                f"{traj_atm_symbol[idx, ii]} {traj_atm_coords[idx, ii, 0]} {traj_atm_coords[idx, ii, 1]} {traj_atm_coords[idx, ii, 2]} \n"
            )
            x = x + 1
