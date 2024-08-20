# Prerequisites #

For installation (supported):

- python >= 3.10
- pip >= 21.3
- setuptools >= 60.0
- wheel >= 0.37
- numpy >= 1.22

External programs needed by ArcaNN for trajectories/structures manipulations:

- VMD >= 1.9.3
- Atomsk >= b0.12.2

Supported programs used for each **step**:

- DeePMD-kit >= 2.0 (**training**, **test**)
  - LAMMPS version adequate with DeePMD-kit (**exploration**)
  - i-PI version adequate with DeePMD-kit (**exploration**)
  - plumed version adequate with DeePMD-kit (**exploration**)
- CP2K >= 6.1 (**labeling**)
