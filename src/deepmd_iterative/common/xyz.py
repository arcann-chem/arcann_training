from pathlib import Path
import logging
import sys
from typing import Tuple

# Non-standard imports
import numpy as np


def read_xyz_trajectory(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read an XYZ format trajectory file and return the number of atoms, atomic symbols, and atomic coordinates.

    Args:
        file_path (Path): The path to the trajectory file.

    Returns:
        A tuple containing the following numpy arrays:
        - num_atoms (np.ndarray): Array of the number of atoms with dim(nb_step)
        - atom_symbols (np.ndarray): Array of atomic symbols with dim(nb_step, num_atoms)
        - atom_coords (np.ndarray): Array of atomic coordinates with dim(nb_step, num_atoms, 3)

    Raises:
        ValueError: If the number of atoms is not constant throughout the trajectory file.
    """
    # Initialize the output lists
    num_atoms_list = []
    atom_symbols_list = []
    atom_coords_list = []

    # Open the file and read in the lines
    with file_path.open("r") as f:
        lines = f.readlines()

        # Loop through each line in the file
        for i, line in enumerate(lines):
            # First line contains the number of atoms in the first timestep.
            if i == 0:
                num_atoms = int(line.strip())
                # Create empty numpy arrays to store atomic symbols and coordinates for the current timestep.
                step_atom_symbols = np.zeros((num_atoms,), dtype="<U2")
                step_atom_coords = np.zeros((num_atoms, 3))
                num_atoms_list.append(num_atoms)
            # Every other (n_atoms + 2) line contains the number of atoms in the current timestep.
            elif i % (num_atoms + 2) == 1:
                if int(line.strip()) != num_atoms:
                    error_msg = "Number of atoms is not constant throughout the trajectory file."
                    logging.error(f"{error_msg}\nAborting...")
                    sys.exit(1)
                    # raise ValueError(error_msg)
                num_atoms = int(line.strip())
                num_atoms_list.append(num_atoms)
                # Add the atomic symbols and coordinates for the previous timestep to their respective lists.
                atom_symbols_list.append(step_atom_symbols)
                atom_coords_list.append(step_atom_coords)
                # Create empty numpy arrays to store atomic symbols and coordinates for the current timestep.
                step_atom_symbols = np.zeros((num_atoms,), dtype="<U2")
                step_atom_coords = np.zeros((num_atoms, 3))
            # Every line that is not the first or a (n_atoms + 2) line contains the atomic symbol and coordinates.
            elif i % (num_atoms + 2) != 0:
                fields = line.split()
                # Add the atomic symbol and coordinates to their respective numpy arrays for the current timestep.
                step_atom_symbols[i % (num_atoms + 2) - 2] = fields[0]
                step_atom_coords[i % (num_atoms + 2) - 2] = fields[1:4]

    # Add the atomic symbols and coordinates for the last timestep to their respective lists.
    atom_symbols_list.append(step_atom_symbols)
    atom_coords_list.append(step_atom_coords)

    # Convert the lists to numpy arrays.
    num_atoms = np.array(num_atoms_list, dtype=int)
    atom_symbols = np.array(atom_symbols_list)
    atom_coords = np.array(atom_coords_list)

    return num_atoms, atom_symbols, atom_coords


def write_xyz_frame_to_file(
    file_path: Path,
    frame_idx: int,
    num_atoms: np.ndarray,
    atom_coords: np.ndarray,
    atom_symbols: np.ndarray,
) -> None:
    """
    Writes the XYZ coordinates of a specific frame of a trajectory to a file.

    Args:
        file_path (Path): The file path to write the XYZ coordinates to.
        frame_idx (int): The index of the frame to write the XYZ coordinates for.
        num_atoms (np.ndarray): An array containing the number of atoms in each frame of the trajectory with dim(nb_step).
        atom_coords (np.ndarray): An array containing the coordinates of each atom in each frame of the trajectory with dim(nb_step, num_atoms, 3).
        atom_symbols (np.ndarray): An array containing the atomic symbols for each atom in each frame of the trajectory with dim(nb_step, num_atoms).

    Raises:
        ValueError: If the specified frame index is out of range.

    Returns:
        None
    """
    # Check that the specified frame index is within the range of available frames
    if frame_idx >= num_atoms.size:
        error_msg = f"Frame index out of range: {frame_idx} (number of frames: {num_atoms.size})"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)
        # raise ValueError(error_msg)

    # Open the specified file in write mode
    with file_path.open("w") as xyz_file:
        # Write the number of atoms in the specified frame to the file
        xyz_file.write(f"{num_atoms[frame_idx]}\n")
        # Write a line indicating the index of the frame
        xyz_file.write(f"Frame index: {frame_idx}\n")
        # Loop over each atom in the specified frame
        for ii in range(num_atoms[frame_idx]):
            # Write the atomic symbol and Cartesian coordinates to the file in XYZ format
            xyz_file.write(
                f"{atom_symbols[frame_idx, ii]} "
                f"{atom_coords[frame_idx, ii, 0]: .6f} "
                f"{atom_coords[frame_idx, ii, 1]: .6f} "
                f"{atom_coords[frame_idx, ii, 2]: .6f}\n"
            )
    # Close the file
    xyz_file.close()
