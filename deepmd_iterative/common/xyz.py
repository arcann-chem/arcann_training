"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/09/06

The xyz module provides functions to manipulate XYZ data (as np.ndarray).

Functions
---------
parse_cell_from_comment(comment_line: str) -> np.ndarray
    A function to parse the cell informations from a comment line.

read_xyz_trajectory(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    A function to read an XYZ format trajectory file and return the number of atoms, atomic symbols, and atomic coordinates.

write_xyz_frame(file_path: Path, frame_idx: int, num_atoms: np.ndarray, atom_coords: np.ndarray, atom_symbols: np.ndarray) -> None
    A function to write the XYZ coordinates of a specific frame of a trajectory to a file.
"""

# Standard library modules
import re
from pathlib import Path
from typing import Tuple

# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


# Unittested
@catch_errors_decorator
def parse_cell_from_comment(comment_line: str) -> np.ndarray:
    """
    Parses the cell informations from a comment line.

    This function attempts to match the input comment line with specific patterns to extract cell information.
    It first tries to match the pattern with 9 numbers and then with 3 numbers.

    Parameters
    ----------
    comment_line : str
        The comment line containing cell information.

    Returns
    -------
    np.ndarray or False
        If the comment line matches any pattern, returns an array representing the cell;
        otherwise, returns False.

    Examples
    --------
    >>> parse_cell_from_comment("ABC = 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0")
    array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    >>> parse_cell_from_comment("ABC = 1.0 2.0 3.0")
    array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])

    >>> parse_cell_from_comment("Invalid format")
    False
    """
    nine_numbers_match = re.match(
        r"ABC\s*=\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$",
        comment_line,
    )
    if nine_numbers_match:
        return np.array([float(num) for num in nine_numbers_match.groups()])

    three_numbers_match = re.match(r"ABC\s*=\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$", comment_line)
    if three_numbers_match:
        three_cell = np.array([float(num) for num in three_numbers_match.groups()])
        cell = np.zeros(9)
        for i, f in enumerate([0, 4, 8]):
            cell[f] = three_cell[i]
        return cell

    return False


# Unittested
@catch_errors_decorator
def read_xyz_trajectory(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read an XYZ format trajectory file and return the number of atoms, atomic symbols, atomic coordinates, and cell.

    Parameters
    ----------
    file_path : Path
        The path to the trajectory file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the following numpy arrays:
        - num_atoms (np.ndarray): Array of the number of atoms for each step with dim(nb_step)
        - atom_symbols (np.ndarray): Array of atomic symbols with dim(nb_step, num_atoms)
        - atom_coords (np.ndarray): Array of atomic coordinates with dim(nb_step, num_atoms, 3)
        - cell_info (np.ndarray): Array of cell parameters with dim(nb_step, 9) or False if not present.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    TypeError
        If the number of atoms is not an integer.
    ValueError
        If the number of atoms is not constant throughout the trajectory file.
        If the file format is incorrect.
        If the comment line format is inconsistent.
    """
    # Check if the file exists
    if not file_path.is_file():
        # If the file does not exist, log an error message and abort
        error_msg = f"File not found '{file_path.name}' not in '{file_path.parent}'."
        raise FileNotFoundError(error_msg)

    # Initialize the output lists
    num_atoms_list = []
    atom_symbols_list = []
    atom_coords_list = []
    cell_info_list = []

    # Open the file and read in the lines
    with file_path.open("r") as f:
        lines = f.readlines()

        # Loop through each line in the file
        i = 0
        while i < len(lines):
            # First line contains the total number of atoms in the molecule
            num_atoms_str = lines[i].strip()
            if not re.match(r"^\d+$", num_atoms_str):
                error_msg = f"Incorrect file format: number of atoms must be an '{type(1)}'."
                raise TypeError(error_msg)
            num_atoms = int(num_atoms_str)
            num_atoms_list.append(num_atoms)

            # Second line contains ABC = xx xy xz yx yy yz zx zy zz or ABC = a b c or comments
            comment_line = lines[i + 1].strip()
            step_cell = parse_cell_from_comment(comment_line)
            cell_info_list.append(step_cell)

            # Initialize arrays to store the symbols and coordinates for the current timestep
            step_atom_symbols = np.zeros((num_atoms,), dtype="<U2")
            step_atom_coords = np.zeros((num_atoms, 3))

            # Loop through the lines for the current timestep
            for j in range(num_atoms):
                # Parse the line to get the symbol and coordinates
                try:
                    fields = lines[i + j + 2].split()
                except IndexError:
                    error_msg = f"Incorrect file format: end of file reached prematurely."
                    raise IndexError(error_msg)

                if len(fields) != 4:
                    error_msg = f"Incorrect file format: each line after the first two must contain an atomic symbol and three floating point numbers."
                    raise ValueError(error_msg)

                symbol = fields[0]
                if not re.match(r"^[A-Za-z]{1,2}$", symbol):
                    error_msg = f"Incorrect file format: invalid atomic symbol '{symbol}' on line {i+j+2}."
                    raise ValueError(error_msg)
                try:
                    x, y, z = map(float, fields[1:4])
                except ValueError:
                    error_msg = f"Incorrect file format: could not parse coordinates on line {i+j+2}."
                    raise ValueError(error_msg)

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
        error_msg = f"Number of atoms is not constant throughout the trajectory file."
        raise ValueError(error_msg)

    valid_cell_types = {np.ndarray, bool}
    cell_types = {type(cell) for cell in cell_info_list}
    if len(cell_types) > 1 or not (cell_types <= valid_cell_types):
        error_msg = f"The comment line format is inconsistent."
        raise TypeError(error_msg)

    # Convert the lists to numpy arrays.
    num_atoms = np.array(num_atoms_list, dtype=int)
    atom_symbols = np.array(atom_symbols_list)
    atom_coords = np.array(atom_coords_list)
    cell_info = np.array(cell_info_list)

    return num_atoms, atom_symbols, atom_coords, cell_info


# Unittested
@catch_errors_decorator
def write_xyz_frame(
    file_path: Path,
    frame_idx: int,
    num_atoms: np.ndarray,
    atom_symbols: np.ndarray,
    atom_coords: np.ndarray,
    cell_info: np.ndarray,
) -> None:
    """
    Write the XYZ coordinates of a specific frame of a trajectory to a file.

    Parameters
    ----------
    file_path : Path
        The file path to write the XYZ coordinates to.
    frame_idx : int
        The index of the frame to write the XYZ coordinates for.
    num_atoms : np.ndarray
        An array containing the number of atoms in each frame of the trajectory with shape (nb_step).
    atom_symbols : np.ndarray
        An array containing the atomic symbols for each atom in each frame of the trajectory with shape (nb_step, num_atoms).
    atom_coords : np.ndarray
        An array containing the coordinates of each atom in each frame of the trajectory with shape (nb_step, num_atoms, 3).
    atom_coords : np.ndarray
        An array containing the coordinates of each atom in each frame of the trajectory with shape (nb_step, num_atoms, 3).
    cell_info : np.ndarray
        An array containing the cell information for each frame of the trajectory with shape (nb_step, 9).

    Returns
    -------
    None

    Raises
    ------
    IndexError
        If the specified frame index is out of range.
    """
    # Check that the specified frame index is within the range of available frames
    if frame_idx >= num_atoms.size:
        error_msg = f"Frame index out of range: {frame_idx} (number of frames: {num_atoms.size})."
        raise IndexError(error_msg)
    # Open the specified file in write mode
    with file_path.open("w") as xyz_file:
        # Write the number of atoms in the specified frame to the file
        xyz_file.write(f"{num_atoms[frame_idx]}\n")
        # Write a line indicating the index of the frame or if the cell is defined
        if isinstance(cell_info[frame_idx], np.bool_) and not cell_info[frame_idx]:
            xyz_file.write(f"Frame index: {frame_idx}\n")
        elif isinstance(cell_info, bool):
            xyz_file.write(f"Frame index: {frame_idx}\n")
        else:
            print(cell_info[frame_idx])
            cell = " ".join([f"{value:.4f}" for value in cell_info[frame_idx]])
            xyz_file.write(f"ABC = {cell}\n")

        # Loop over each atom in the specified frame
        for ii in range(num_atoms[frame_idx]):
            # Write the atomic symbol and Cartesian coordinates to the file in XYZ format
            xyz_file.write(f"{atom_symbols[frame_idx, ii]} " f"{atom_coords[frame_idx, ii, 0]:.6f} " f"{atom_coords[frame_idx, ii, 1]:.6f} " f"{atom_coords[frame_idx, ii, 2]:.6f}\n")
    # Close the file
    xyz_file.close()
