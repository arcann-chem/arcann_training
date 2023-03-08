from typing import Dict, List, Tuple
# Non-standard library imports
import numpy as np


def parse_lammps_data(lines: List[str]) -> Tuple(int, int, np.ndarray, Dict[int], np.ndarray):
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
            continue
        if in_atoms_section:
            atoms.append(line.split()[3:6])
        if "Masses" in line:
            in_masses_section = True
            continue
        if in_masses_section:
            fields = line.split()
            masses[int(fields[0])] = float(fields[1])
        if "xlo" in line:
            xlo, xhi = map(float, line.split()[0:2])
        if "ylo" in line:
            ylo, yhi = map(float, line.split()[0:2])
        if "zlo" in line:
            zlo, zhi = map(float, line.split()[0:2])
        if "xy" in line:
            xy, xz, yz = map(float, line.split()[0:3])
        if "atome type" in line:
            num_atom_types = int(line.split()[0])
        if "atoms" in line:
            num_atoms = int(line.split()[0])
    # Convert lists to NumPy arrays
    atoms = np.array(atoms, dtype=float)

    # Return results
    return (
        num_atoms,
        num_atom_types,
        np.array([xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz]),
        masses,
        atoms,
    )
