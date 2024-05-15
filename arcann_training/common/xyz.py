"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15

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
from typing import Tuple, List, Optional

# Third-party modules
import numpy as np

# Local imports
from arcann_training.common.utils import catch_errors_decorator


# TODO: Add tests for this function
@catch_errors_decorator
def parse_xyz_trajectory_file(trajectory_file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Optional[List[bool]], Optional[bool], Optional[float]]:
    """
    Parses an XYZ format trajectory file, extracting atomic structure and optional extended properties such as lattice information,
    periodic boundary conditions (PBC), and additional properties if they are provided in the comments.

    Parameters
    ----------
    trajectory_file_path : Path
        The path to the trajectory file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Optional[List[bool]], Optional[bool], Optional[float]]
        - atom_counts: An array of the number of atoms for each frame.
        - atomic_symbols: An array of atomic symbols for each frame, each symbol up to 3 characters.
        - atomic_coordinates: An array of atomic coordinates for each frame.
        - comments: A list of comments for each frame.
        - lattice: Optional array of lattice parameters for each frame if provided.
        - pbc: Optional list of booleans indicating periodic boundary conditions if provided and valid (True/False for each axis).
        - properties: Optional boolean indicating if specific properties like 'species:S:1:pos:R:3' are present.
        - max_f_std: Optional float representing maximum force standard deviation if provided.

    Raises
    ------
    FileNotFoundError
        If the trajectory file does not exist.
    TypeError
        If the number of atoms is not an integer.
    ValueError
        If the number of atoms changes throughout the file or if the file format is incorrect.
    """
    if not trajectory_file_path.is_file():
        raise FileNotFoundError(f"File not found: {trajectory_file_path}")

    atom_counts, atomic_symbols, atomic_coordinates, comments, lattice_info, pbc_info, properties_info, max_f_std_info = [], [], [], [], [], [], [], []

    with trajectory_file_path.open("r") as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        atom_count_str = lines[i].strip()
        if not atom_count_str.isdigit():
            raise TypeError("Incorrect file format: number of atoms must be an integer.")
        atom_count = int(atom_count_str)
        atom_counts.append(atom_count)

        comment = lines[i + 1].strip()
        comments.append(comment)

        lattice, properties, pbc, max_f_std = parse_extended_format(comment)

        lattice_info.append(np.array([float(x) for x in lattice]) if lattice else None)
        if pbc:
            if len(pbc) == 3:
                pbc_info.append(pbc)
            else:
                raise ValueError("PBC data must consist of three boolean values (True or False for each axis).")
        else:
            pbc_info.append(None)

        properties_info.append(properties if properties else None)
        max_f_std_info.append(max_f_std if max_f_std else None)

        symbols_frame = np.zeros(atom_count, dtype='<U3')
        coordinates_frame = np.zeros((atom_count, 3))

        for j in range(atom_count):
            line_elements = lines[i + 2 + j].split()
            if len(line_elements) != 4:
                raise ValueError("Incorrect file format: expected an atomic symbol followed by three coordinates.")
            symbol, x, y, z = line_elements
            symbols_frame[j] = symbol
            coordinates_frame[j] = [float(x), float(y), float(z)]

        atomic_symbols.append(symbols_frame)
        atomic_coordinates.append(coordinates_frame)

        i += atom_count + 2

    if len(set(atom_counts)) > 1:
        raise ValueError("Number of atoms is not constant throughout the trajectory file.")

    return np.array(atom_counts), np.array(atomic_symbols), np.array(atomic_coordinates), comments, lattice_info, pbc_info, properties_info, max_f_std_info


# TODO: Add tests for this function
@catch_errors_decorator
def parse_extended_format(comment_line: str) -> Tuple[Optional[List[float]], bool, Optional[List[bool]], Optional[float]]:
    """
    Parses the comment line of an extended XYZ file for lattice, properties, periodic boundary conditions (PBC),
    and max force standard deviation.

    Parameters
    ----------
    comment_line : str
        The comment line containing lattice and properties information.

    Returns
    -------
    Tuple[Optional[List[float]], bool, Optional[List[bool]], Optional[float]]
        - Lattice information as an optional list of floats.
        - Boolean indicating if properties are correctly formatted.
        - PBC as an optional list of booleans.
        - Max force standard deviation as an optional float.
    """
    lattice_regex = r"Lattice=\"((?:[-\d\.]+\s+){8}[-\d\.]+)\""
    properties_regex = r"Properties=species:S:1:pos:R:3"
    pbc_regex = r'pbc="([^"]*)"'
    max_f_std_regex = r"max_f_std=([\d.]+)"

    lattice_match = re.search(lattice_regex, comment_line)
    properties_match = re.search(properties_regex, comment_line)
    pbc_match = re.search(pbc_regex, comment_line)
    max_f_std_match = re.search(max_f_std_regex, comment_line)

    lattice_values = [float(value) for value in lattice_match.group(1).split()] if lattice_match else None
    properties_present = bool(properties_match)
    pbc_values = [v.lower() in ['true', 't'] for v in pbc_match.group(1).split()] if pbc_match else None
    max_f_std_value = float(max_f_std_match.group(1)) if max_f_std_match else None

    return lattice_values, properties_present, pbc_values, max_f_std_value

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
        if len(cell_info) > 0:
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
