# About The Project #

ArcaNN proposes an automated enhanced sampling generation of training sets for chemically reactive machine learning interatomic potentials.
In its current version, it aims to simplify and to automate the iterative training process of a [DeePMD-kit](https://doi.org/10.1063/5.0155600) neural network potential for a user-chosen system, but the core concepts of the training procedure could be extended to other network architectures.
The main advantages of this code are its modularity, the ability to finely tune the training process to adapt to your system and workflow, and great traceability, as the code records every parameter set during the procedure.
During the iterative training process, you will iteratively train neural network potentials, use them as reactive force fields for molecular dynamics simulations (to explore the phase space), select and label some configurations based on a query by committee approach, and then train neural network potentials again with an improved training set, and so forth.
This workflow, sometimes referred to as active or concurrent learning, was heavily inspired by [DP-GEN](https://doi.org/10.1016/j.cpc.2020.107206), and we use their naming scheme for the steps of the iterative procedure.
Namely, each iteration or cycle will consist of the **training**, **exploration**, **labeling**, and (optional) **test** steps.
Make sure that you understand the meaning of each step before using the code.

This repository contains several folders:

- The `examples/` folder contains all the necessary files (in template form) to set up the iterative training procedure for your system:
  - The `inputs/` folder with five JSON files, one per `step`.
These files contain all the keywords used to control each step of an iteration (namely **initialization**, **exploration**, **labeling**, **training** and optionally **test**), including their type and the default values taken by the code if a keyword isn't provided by the user.
If the default is a list containing a single value it means that this value will be repeated and used for every **system** (see below).
For the **exploration** step some keywords have two default values, the first one will be used if the exploration is conducted with classical nuclei MD (*i.e.* LAMMPS) and the second one will be used with quantum nuclei MD (*i.e.* i-PI).
  - The `user_files/` folder with:
    - A `machine.json` which contains all the information about your cluster that the code needs (see [HPC Configuration](./getting-started/hpc_conf.md)).
    - A input folder for each `step`, where skeleton files are provided as templates for writing your own inputs for the respective external programs.
For example, in the `exploration_lammps/`, `labeling_cp2k/` and `training_deepmd/`, you can find the necessecary files to perform the **exploration** with LAMMPS, the **labeling** with CP2K and the **training** with DeePMD-kit (see [Exploration](./usage/exploration), [Labeling](./usage/labeling) and [Training](./usage/training) for a detailed description of the **tunneable** keywords).
    - A job folder for each step, where skeleton submission files are provided as template that **ArcaNN** use to launch the different phases in each `step` when they require a HPC machine.
For example, in `job_exploration_lammps_slurm/`, `job_labeling_CP2K_slurm`, `job_training_deepmd_slurm` and the optional `step` `job_test_deepmd_slurm`, you can find basic `Slurm` submission files.

You **must**  adapt these files to ensure they work on your machine (see [Usage](./usage/iter_prerequisites.md)), but be careful not to modify the **replaceable** keywords (every word starting with `_R_` and ending with `_`) that Arcann will replace with user-defined or auto-generated values (e.g., the wall time for labeling calculations, the cluster partition to be used, etc.).

- The `tools/` folder which contains helper scripts and files.
- The `arcann_training/` folder contains all the files that comprise the `ArcaNN Training` code. **We strongly advise against modifying its contents**.
