"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/30

The xyz module provides functions to manipulate XYZ data (as np.ndarray).

Functions
---------
parse_xyz_trajectory_file(trajectory_file_path: Path, is_extended: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]
    A function to parse an extended XYZ format trajectory file, returning information about the atomic structure throughout the trajectory.

parse_extended_format(comment_line: str) -> Tuple[List[float], bool]
    A function to parse the comment line of an extended XYZ file for lattice and properties information.

write_xyz_frame(trajectory_file_path: Path, frame_idx: int, atom_counts: np.ndarray, atomic_symbols: np.ndarray, atomic_coordinates: np.ndarray, cell_info: np.ndarray, comments: List[str]) -> None
    A function to write the XYZ coordinates of a specific frame from a trajectory to a file, including extended format lattice information if provided.
"""
# TODO: Homogenize the docstrings for this module

# Standard library modules
import re
from pathlib import Path
from typing import Tuple, List

# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator

# TODO: Add tests for this function
@catch_errors_decorator
def parse_xyz_trajectory_file(trajectory_file_path: Path, is_extended: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Parses an XYZ format trajectory file, returning information about the atomic structure throughout the trajectory.

    Parameters
    ----------
    trajectory_file_path : Path
        The path to the trajectory file.
    is_extended : bool, optional
        Indicates if the file is in extended XYZ format which includes lattice information and properties.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]
        A tuple containing the following:
        - atom_counts: An array of the number of atoms for each frame.
        - atomic_symbols: An array of atomic symbols for each frame, with each symbol up to 3 characters.
        - atomic_coordinates: An array of atomic coordinates for each frame.
        - cell_info: An array of cell lattice information for each frame (only if is_extended is True and valid).
        - comments: A list of comments for each frame.

    Raises
    ------
    FileNotFoundError
        If the trajectory file does not exist.
    TypeError
        If the number of atoms is not an integer or extended format is incorrect.
    ValueError
        If the number of atoms changes throughout the file or if the file format is incorrect.
    """
    # Check if the file exists
    if not trajectory_file_path.is_file():
        # If the file does not exist, log an error message and abort
        error_msg = f"File not found {trajectory_file_path.name} not in {trajectory_file_path.parent}"
        raise FileNotFoundError(error_msg)

    # Initialize the output lists
    atom_counts, atomic_symbols, atomic_coordinates, cell_info, comments = [], [], [], [], []

    # Open the file and read in the file_lines
    with trajectory_file_path.open("r") as f:
        lines  = f.readlines()

        # Loop through each line in the file
        i = 0
        while i < len(lines):
            # First line contains the total number of atoms in the molecule
            atom_count_str = lines[i].strip()
            if not re.match(r"^\d+$", atom_count_str):
                error_msg = "Incorrect file format: number of atoms must be an integer."
                raise TypeError(error_msg)
            atom_count  = int(atom_count_str)
            atom_counts.append(atom_count)

            # Second line is the comment line (optional)
            comment = lines[i + 1].strip()
            comments.append(comment)

            if is_extended:
                lattice, properties = parse_extended_format(comment)
                cell_info.append(lattice)

            # Initialize arrays to store the symbols and coordinates for the current timeframe
            symbols_frame = np.zeros((atom_count,), dtype="<U3")
            coordinates_frame = np.zeros((atom_count, 3))

            for j in range(atom_count):
                line = lines[i + j + 2].split()

                if len(line) != 4:
                    raise ValueError("Incorrect file format: expected an atomic symbol followed by three coordinates.")

                symbol, x, y, z = line[0], float(line[1]), float(line[2]), float(line[3])
                symbols_frame[j] = symbol
                coordinates_frame[j] = [x, y, z]

            atomic_symbols.append(symbols_frame)
            atomic_coordinates.append(coordinates_frame)

            i += atom_count + 2

    if len(set(atom_counts)) > 1:
        raise ValueError("Number of atoms is not constant throughout the trajectory file.")

    return (np.array(atom_counts), np.array(atomic_symbols), np.array(atomic_coordinates), np.array(cell_info) if is_extended else np.array([]), comments)


# TODO: Add tests for this function
@catch_errors_decorator
def parse_extended_format(comment_line: str) -> Tuple[List[float], bool]:
    """
    Parses the comment line of an extended XYZ file for lattice and properties information.

    Parameters
    ----------
    comment_line : str
        The comment line containing lattice and properties information.

    Returns
    -------
    Tuple[List[float], bool]
        Lattice information as a list of floats and a boolean indicating if properties are correctly formatted.
    """
    lattice_regex = r"Lattice=\"((?:[-\d\.]+\s+){8}[-\d\.]+)\""
    properties_regex = r"Properties=species:S:1:pos:R:3"

    lattice_match = re.search(lattice_regex, comment_line)
    properties_match = re.search(properties_regex, comment_line)

    if lattice_match and properties_match:
        lattice_floats = [float(value) for value in lattice_match.group(1).split()]
        return lattice_floats, True
    else:
        error_msg = f"Incorrect extended format: comment '{comment_line}'."
        raise ValueError(error_msg)


# TODO: Add tests for this function
@catch_errors_decorator
def write_xyz_frame(trajectory_file_path: Path, frame_idx: int, atom_counts: np.ndarray, atomic_symbols: np.ndarray, atomic_coordinates: np.ndarray, cell_info: np.ndarray, comments: List[str]) -> None:
    """
    Writes the XYZ coordinates of a specific frame from a trajectory to a file, including extended format lattice information if provided.

    Parameters
    ----------
    trajectory_file_path : Path
        The file path where the XYZ coordinates will be written.
    frame_idx : int
        The index of the frame to write the XYZ coordinates for.
    atom_counts : np.ndarray
        An array containing the number of atoms in each frame of the trajectory.
    atomic_symbols : np.ndarray
        An array containing the atomic symbols for each atom in each frame of the trajectory.
    atomic_coordinates : np.ndarray
        An array containing the coordinates of each atom in each frame of the trajectory.
    cell_info : np.ndarray
        An array containing the cell lattice information for each frame.
    comments : List[str]
        A list of comments for each frame.

    Raises
    ------
    IndexError
        If the specified frame index is out of range.
    """

    # Validate frame index
    if frame_idx >= atom_counts.shape[0]:
        raise IndexError(f"Frame index out of range: {frame_idx} (total frames: {atom_counts.shape[0]})")

    # Start writing to the file
    with trajectory_file_path.open("w") as file:
        # Write the atom count for the frame
        file.write(f"{atom_counts[frame_idx]}\n")

        # Write the comment line with cell information if available
        if cell_info.size > 0:
            cell_line = " ".join(map(str, cell_info[frame_idx]))
            comment_line = f'Lattice="{cell_line}" Properties=species:S:1:pos:R:3'
        else:
            comment_line = comments[frame_idx] if frame_idx < len(comments) else ""
        file.write(f"{comment_line}\n")

        # Write atomic symbols and coordinates for the frame
        for atom_index in range(atom_counts[frame_idx]):
            symbol = atomic_symbols[frame_idx, atom_index]
            coords = atomic_coordinates[frame_idx, atom_index]
            coords_line = " ".join(f"{coord:.6f}" for coord in coords)
            file.write(f"{symbol} {coords_line}\n")
