"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/09/05

The lammps module provides functions to manipulate LAMMPS data (as list of strings).

Functions
---------
read_lammps_data(lines: List[str],) -> Tuple(int, int, np.ndarray, Dict[int], np.ndarray)
    Read LAMMPS data file and extract required information.
"""
# Standard library modules
from typing import Dict, List, Tuple

# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


# Unittested
@catch_errors_decorator
def read_lammps_data(
    lines: List[str],
) -> Tuple[int, int, np.ndarray, Dict[int, float], np.ndarray]:
    """
    Read LAMMPS data file and extract required information.

    Parameters
    ----------
    lines : List[str]
        List of strings, where each string represents a line from the LAMMPS data file.

    Returns
    -------
    Tuple[int, int, np.ndarray, Dict[int, float], np.ndarray]
        A tuple containing the number of atoms, number of atom types, simulation box boundaries,
        atom masses, and atom coordinates, respectively. The simulation box boundaries are stored
        as a numpy array with the following format: [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz].
        The atom masses are stored in a dictionary with atom type as key and mass as value.
        The atom coordinates are stored as a numpy array with shape (num_atoms, 3).

    Raises
    ------
    ValueError
        If any required information is missing or inconsistent in the input data.
    """
    # Input validation
    if not lines or not isinstance(lines, list):
        raise ValueError("Input 'lines' must be a non-empty list of strings.")

    # Initialize variables
    num_atoms = None
    num_atom_types = None
    xlo, xhi = None, None
    ylo, yhi = None, None
    zlo, zhi = None, None
    xy, xz, yz = None, None, None
    masses = {}
    atoms = []
    in_atoms_section = False
    in_masses_section = False

    # Parse input file
    for line in lines:
        if len(line) == 0:
            continue
        if "Atoms" in line:
            in_atoms_section = True
            in_masses_section = False
            continue
        if in_atoms_section:
            atoms.append(line.split()[2:6])
            continue
        if "Masses" in line:
            in_masses_section = True
            in_atoms_section = False
            continue
        if in_masses_section:
            fields = line.split()
            masses[int(fields[0])] = float(fields[1])
            continue
        if "xlo" in line:
            xlo, xhi = map(float, line.split()[0:2])
            continue
        if "ylo" in line:
            ylo, yhi = map(float, line.split()[0:2])
            continue
        if "zlo" in line:
            zlo, zhi = map(float, line.split()[0:2])
            continue
        if "xy" in line:
            xy, xz, yz = map(float, line.split()[0:3])
            continue
        if "atom types" in line:
            num_atom_types = int(line.split()[0])
            continue
        if "atoms" in line:
            num_atoms = int(line.split()[0])
            continue
    # Convert lists to NumPy arrays
    atoms = np.array(atoms, dtype=float)
    if num_atoms == None:
        error_msg = "The number of atoms was not found."
        raise ValueError(error_msg)
    if num_atom_types == None:
        error_msg = "The number of atom types was not found."
        raise ValueError(error_msg)
    if (
        xlo == None
        or xhi == None
        or ylo == None
        or yhi == None
        or zlo == None
        or zhi == None
    ):
        error_msg = f"Invalid box coordinates."
        raise ValueError(error_msg)
    if len(masses) == 0:
        error_msg = f"Masses not found"
        raise ValueError(error_msg)
    if len(masses) != num_atom_types:
        error_msg = f"Number of masses ('{len(masses)}') does not match the number of atom types ('{num_atom_types}')."
        raise ValueError(error_msg)
    if len(atoms) == 0:
        error_msg = f"Coordinates not found."
        raise ValueError(error_msg)
    if len(atoms) != num_atoms:
        error_msg = f"Number of coordinates ('{len(atoms)}') does not match the number of atoms ('{num_atoms}')."
        raise ValueError(error_msg)

    # Return results
    return (
        num_atoms,
        num_atom_types,
        np.array([xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz]),
        masses,
        atoms,
    )
