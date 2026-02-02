"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2026 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2026/01/31

The lammps module provides functions to manipulate LAMMPS data (as list of strings).

Functions
---------
read_lammps_data(lines: List[str],) -> Tuple(int, int, np.ndarray, Dict[int], np.ndarray)
    Read LAMMPS data file and extract required information.
"""

# TODO: Homogenize the docstrings for this module

# Standard library modules
from pathlib import Path
from typing import Dict, List, Union, Tuple

# Third-party modules
import numpy as np

# Local imports
from arcann_training.common.utils import catch_errors_decorator
from arcann_training.common.list import textfile_to_string_list


# Unittested
@catch_errors_decorator
def read_lammps_data(
    data_file: Union[Path, List[str]],
) -> Tuple[int, int, np.ndarray, Dict[int, float], np.ndarray]:
    """
    Read LAMMPS data file and extract required information.

    Parameters
    ----------
    data_file : Path
        Path to the LAMMPS data file or list of lines from a LAMMPS data file.

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
    if type(data_file) == type(Path(".")):
        lines = textfile_to_string_list(data_file)
    else:
        lines = data_file

    # Basic validation of required sections
    validate_lammps_sections(lines, data_file)

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
    atoms_type_list = []
    in_atoms_section = False
    in_masses_section = False

    # Parse input file
    for line in lines:
        if len(line) == 0:
            continue
        if "Atoms" in line and "Atomsk" not in line:
            in_atoms_section = True
            in_masses_section = False
            continue
        if in_atoms_section:
            atoms.append(line.split()[2:6])
            atoms_type_list.append(line.split()[1:2])
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

    atoms_type_list = np.array(atoms_type_list, dtype=int)
    atoms_type_list = np.unique(atoms_type_list)

    for atoms_type in atoms_type_list:
        if atoms_type not in masses:
            if type(data_file) == type(Path(".")):
                error_msg = f"Atom type '{atoms_type}' present in the coordinates section but not found in masses. Problem with your LMP file: {data_file}"
            else:
                error_msg = f"Atom type '{atoms_type}' present in the coordinates section but not found in masses. Problem with your LMP file"
            raise ValueError(error_msg)

    # Return results
    return (
        num_atoms,
        num_atom_types,
        np.array([xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz]),
        masses,
        atoms,
    )


@catch_errors_decorator
def validate_lammps_sections(
    data_file: Union[Path, List[str]], source: Union[Path, str, None] = None
) -> bool:
    """
    Check that a LAMMPS data file contains the minimum required sections.

    Required:
    - A line specifying the number of atoms (contains 'atoms')
    - A line specifying the number of atom types (contains 'atom types')
    - A 'Masses' section
    - An 'Atoms' section header

    Raises
    ------
    ValueError
        With an explicit message indicating which section is missing.
    """
    # Read lines
    if type(data_file) == type(Path(".")):
        lines = textfile_to_string_list(data_file)
    else:
        lines = data_file

    if not lines or not isinstance(lines, list):
        raise ValueError("LAMMPS input is empty or invalid.")

    lowered = [l.lower() for l in lines]

    # atoms count line
    has_atoms_count = any("atoms" in l for l in lowered)
    if not has_atoms_count:
        src = f": {source}" if isinstance(source, Path) else ""
        raise ValueError(f"Missing 'N atoms' line in LAMMPS data file{src}.")

    # atom types line
    has_atom_types = any("atom types" in l for l in lowered)
    if not has_atom_types:
        src = f": {source}" if isinstance(source, Path) else ""
        raise ValueError(f"Missing 'N atom types' line in LAMMPS data file{src}.")

    # Masses section
    has_masses = any("masses" in l for l in lowered)
    if not has_masses:
        src = f": {source}" if isinstance(source, Path) else ""
        raise ValueError(f"Missing 'Masses' section in LAMMPS data file{src}.")

    # Atoms section header (avoid Atomsk)
    has_atoms_section = any(
        l.strip().lower().startswith("atoms")
        for l in lines
        if "atomsk" not in l.lower()
    )
    if not has_atoms_section:
        src = f": {source}" if isinstance(source, Path) else ""
        raise ValueError(f"Missing 'Atoms' section header in LAMMPS data file{src}.")

    return True


@catch_errors_decorator
def get_lammps_atom_types(data_file: Union[Path, List[str]]) -> np.ndarray:
    """
    Extract per-atom type indices from a LAMMPS data file (zero-based).

    Parameters
    ----------
    data_file : Path or List[str]
        Path to the LAMMPS data file or a list of lines from it.

    Returns
    -------
    np.ndarray
        Integer array with the atom type for each atom (zero-based).

    Raises
    ------
    ValueError
        If the atom types cannot be parsed or the file is inconsistent.
    """
    # Read lines
    if type(data_file) == type(Path(".")):
        lines = textfile_to_string_list(data_file)
    else:
        lines = data_file

    # Basic validation of required sections
    validate_lammps_sections(lines, data_file)

    # Determine the number of atoms by scanning for the first line containing 'atoms'
    num_atoms = None
    for l in lines:
        if "atoms" in l.lower():
            parts = l.split()
            try:
                num_atoms = int(parts[0])
                break
            except Exception:
                continue
    if num_atoms is None:
        raise ValueError("The number of atoms was not found.")

    # Find candidate Atoms section headers (avoid Atomsk), case-insensitive
    indexes = [
        idx
        for idx, s in enumerate(lines)
        if s.strip().lower().startswith("atoms") and "atomsk" not in s.lower()
    ]
    if not indexes:
        raise ValueError("'Atoms' section not found in LAMMPS data file.")

    # Try each candidate Atoms block until one yields the expected number of types
    successful_types = None
    for idx in indexes:
        try:
            # Collect atom lines by scanning forward, skipping blank/comment lines,
            # until we have collected `num_atoms` entries.
            types = []
            scan_idx = idx + 1
            while scan_idx < len(lines) and len(types) < int(num_atoms):
                ln = lines[scan_idx].strip()
                scan_idx += 1
                if not ln:
                    continue
                if ln.startswith("#"):
                    continue
                fields = ln.split()
                if len(fields) == 0:
                    continue

                # Try to parse 'id type ...' pattern
                parsed = False
                if len(fields) >= 2:
                    try:
                        _ = int(fields[0])
                        atom_type = int(fields[1])
                        types.append(atom_type - 1)
                        parsed = True
                    except Exception:
                        parsed = False

                if parsed:
                    continue

                # Try 'type x y z' (no id)
                try:
                    atom_type = int(fields[0])
                    types.append(atom_type - 1)
                    continue
                except Exception:
                    # Not parseable for this candidate; give up and try next
                    raise ValueError

            if len(types) == int(num_atoms):
                successful_types = np.asarray(types, dtype=np.int64)
                break
        except Exception:
            continue

    if successful_types is None:
        raise ValueError("Not enough atom lines found in Atoms section.")

    return successful_types
