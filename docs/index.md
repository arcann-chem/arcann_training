<div style="text-align: center;">
<img src="./docs/arcann_logo.svg" alt="ArcaNN logo" style="width: 25%; height: auto;" />
</div>

---

[![GNU AGPL v3.0 License](https://img.shields.io/github/license/arcann-chem/arcann_training.svg)](https://github.com/arcann-chem/arcann_training/blob/main/LICENSE)
[![Unit Tests Requirements](https://github.com/arcann-chem/arcann/actions/workflows/unittests_requirements.yml/badge.svg)](https://github.com/arcann-chem/arcann/actions/workflows/unittests_requirements.yml)
[![Unit Tests Matrix](https://github.com/arcann-chem/arcann/actions/workflows/unittests_matrix.yml/badge.svg?branch=main)](https://github.com/arcann-chem/arcann/actions/workflows/unittests_matrix.yml)
[![Docs](https://github.com/arcann-chem/arcann/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/arcann-chem/arcann/actions/workflows/docs.yml)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2407.07751-blue)](https://doi.org/10.48550/arXiv.2407.07751)

---

# ArcaNN #

ArcaNN proposes an automated enhanced sampling generation of training sets for chemically reactive machine learning interatomic potentials.
In its current version, it aims to simplify and automate the iterative training process of a [DeePMD-kit](https://doi.org/10.1063/5.0155600) neural network potential for a user-chosen system.
The core concepts of this training procedure could be extended to other network architectures.

## Key Advantages ##

- **Modularity**: The code is designed with modularity in mind, allowing users to finely tune the training process to fit their specific system and workflow.
- **Traceability**: Every parameter set during the procedure is recorded, ensuring great traceability.

## Iterative Training Process ##

During the iterative training process, you will:

1. Train neural network potentials.

2. Use them as reactive force fields in molecular dynamics simulations to explore the phase space.

3. Select and label configurations based on a query by committee approach.

4. Train a new generation of neural network potentials again with the improved training set.

This workflow, often referred to as active or concurrent learning, is inspired by [DP-GEN](https://doi.org/10.1016/j.cpc.2020.107206).
We adopt their naming scheme for the steps in the iterative procedure. Each iteration, or cycle, consists of the following steps:

- **Training**
- **Exploration**
- **Labeling**
- (Optional) **Testing**

Ensure you understand the meaning of each step before using the code.

## GitHub Repository Structure ##

You will find in our [GitHub repository](https://github.com/arcann-chem/arcann_training/) everything you need to set up the ArcaNN software, as well as example files that you can use as an example. The repository contains several folders:

- The `tools/` folder contains helper scripts and files.
- The `arcann_training/` folder contains the `ArcaNN Training` code. **We strongly advise against modifying its contents**. See [ArcaNN Installation Guide](./getting-started/installation.md) for the installation. 
- The `examples/` folder contains template files to set up the iterative training procedure for your system. Within this folder you can find : 
  - The `inputs/` folder with five JSON files, one per `step`.
These files contain all the keywords used to control each step of an iteration (namely **initialization**, **exploration**, **labeling**, **training** and optionally **test**), including their type and the default values taken by the code if a keyword isn't provided by the user.
If the default is a list containing a single value it means that this value will be repeated and used for every **system** (see the corresponding section fir each `step`).
Note : For the **exploration** step some keywords have two default values, the first one will be used if the exploration is conducted with classical nuclei MD (*i.e.* LAMMPS) and the second one will be used with quantum nuclei MD (*i.e.* i-PI). 
  - The `user_files/` folder with:
    - A `machine.json` template file where all the information about your cluster should be provided (see [HPC Configuration](./getting-started/hpc_conf.md)).
    - A input folder for each `step`, where skeleton files are provided as templates for writing your own inputs for the respective external programs.
    For example, in the `exploration_lammps/`, `labeling_cp2k/` and `training_deepmd/`, you can find the necessecary files to perform the **exploration** with LAMMPS, the **labeling** with CP2K and the **training** with DeePMD-kit (see [Exploration](./usage/exploration), [Labeling](./usage/labeling) and [Training](./usage/training) for a detailed description of the **tunneable** keywords).
    - A job folder for each step, where skeleton submission files are provided as template that **ArcaNN** use to launch the different phases in each `step` when they require a HPC machine.
    For example, in `job_exploration_lammps_slurm/`, `job_labeling_CP2K_slurm`, `job_training_deepmd_slurm` and the optional `step` `job_test_deepmd_slurm`, you can find basic `Slurm` submission files.

You **must** adapt these files to ensure they work on your machine (see [Usage](./usage/iter_prerequisites.md)), but **be careful not to modify the replaceable keywords** (every word starting with `_R_` and ending with `_`) that Arcann will replace with user-defined or auto-generated values (e.g., the wall time for labeling calculations, the cluster partition to be used, etc.).


