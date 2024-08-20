# ArcaNN Requirements #

## Installation Requirements ##

To install and run the software, ensure the following dependencies are installed:

- **Python**: `>= 3.10`
- **Pip**: `>= 21.3`
- **Setuptools**: `>= 60.0`
- **Wheel**: `>= 0.37`
- **NumPy**: `>= 1.22`

## External Programs for Trajectories/Structures Manipulation ##

ArcaNN requires the following external programs for manipulating trajectories and structures:

- **VMD**: `>= 1.9.3`
- **Atomsk**: `>= b0.12.2`

## Supported Programs by Workflow Step ##

Different steps in the workflow are supported by specific programs:

- **DeePMD-kit**: `>= 2.0` (Used in **training** and **testing** steps)
  - **LAMMPS**: Must be compatible with DeePMD-kit (Used in **exploration**)
  - **i-PI**: Must be compatible with DeePMD-kit (Used in **exploration**)
  - **PLUMED**: Must be compatible with DeePMD-kit (Used in **exploration**)
- **CP2K**: `>= 6.1` (Used in the **labeling** step)
