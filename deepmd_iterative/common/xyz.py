from pathlib import Path
import logging
import sys
from typing import Tuple
import re

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
        2: FileNotFoundError: If the specified file does not exist.
        1: ValueError: If the number of atoms is not constant throughout the trajectory file.
    """

    # Check if the file exists
    if not file_path.is_file():
        # If the file does not exist, log an error message and abort
        error_msg = f"File not found {file_path.name} not in {file_path.parent}"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(2)

    # Initialize the output lists
    num_atoms_list = []
    atom_symbols_list = []
    atom_coords_list = []

    # Open the file and read in the lines
    with file_path.open("r") as f:
        lines = f.readlines()

        # Loop through each line in the file
        i = 0
        while i < len(lines):
            # First line contains the total number of atoms in the molecule
            num_atoms_str = lines[i].strip()
            if not re.match(r"^\d+$", num_atoms_str):
                error_msg = "Incorrect file format: number of atoms must be an integer."
                logging.error(f"{error_msg}\nAborting...")
                sys.exit(1)
            num_atoms = int(num_atoms_str)
            num_atoms_list.append(num_atoms)

            # Second line contains the molecule name or comment (optional)
            molecule_name = lines[i + 1].strip()

            # Initialize arrays to store the symbols and coordinates for the current timestep
            step_atom_symbols = np.zeros((num_atoms,), dtype="<U2")
            step_atom_coords = np.zeros((num_atoms, 3))

            # Loop through the lines for the current timestep
            for j in range(num_atoms):
                # Parse the line to get the symbol and coordinates
                try:
                    fields = lines[i + j + 2].split()
                except IndexError:
                    error_msg = (
                        "Incorrect file format: end of file reached prematurely."
                    )
                    logging.error(f"{error_msg}\nAborting...")
                    sys.exit(1)

                if len(fields) != 4:
                    error_msg = "Incorrect file format: each line after the first two must contain an atomic symbol and three floating point numbers."
                    logging.error(f"{error_msg}\nAborting...")
                    sys.exit(1)

                symbol = fields[0]
                if not re.match(r"^[A-Za-z]{1,2}$", symbol):
                    error_msg = f"Incorrect file format: invalid atomic symbol '{symbol}' on line {i+j+2}."
                    logging.error(f"{error_msg}\nAborting...")
                    sys.exit(1)
                try:
                    x, y, z = map(float, fields[1:4])
                except ValueError:
                    error_msg = f"Incorrect file format: could not parse coordinates on line {i+j+2}."
                    logging.error(f"{error_msg}\nAborting...")
                    sys.exit(1)

                # Add the symbol and coordinates to the arrays
                step_atom_symbols[j] = symbol
                step_atom_coords[j] = [x, y, z]

            # Add the arrays for the current timestep to the output lists
            atom_symbols_list.append(step_atom_symbols)
            atom_coords_list.append(step_atom_coords)

            # Increment the line index by num_atoms + 2 (to skip the two lines for the current timestep)
            i += num_atoms + 2

    # Check if the number of atoms is constant throughout the trajectory file.
    if len(set(num_atoms_list)) > 1:
        error_msg = "Number of atoms is not constant throughout the trajectory file."
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)

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
        1: ValueError: If the specified frame index is out of range.

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
                f"{atom_coords[frame_idx, ii, 0]:.6f} "
                f"{atom_coords[frame_idx, ii, 1]:.6f} "
                f"{atom_coords[frame_idx, ii, 2]:.6f}\n"
            )
    # Close the file
    xyz_file.close()
