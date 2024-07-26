<div id="top"></div>

<!-- PROJECT SHIELDS -->

[![GNU AGPL v3.0 License][license-shield]][license-url]
[![Unit Tests Requirements](https://github.com/arcann-chem/arcann/actions/workflows/unittests_requirements.yml/badge.svg)](https://github.com/arcann-chem/arcann/actions/workflows/unittests_requirements.yml)
[![Unit Tests Matrix](https://github.com/arcann-chem/arcann/actions/workflows/unittests_matrix.yml/badge.svg?branch=main)](https://github.com/arcann-chem/arcann/actions/workflows/unittests_matrix.yml)

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#fundings">Fundings</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
<div id="about"></div>

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
    - A `machine.json` which contains all the information about your cluster that the code needs (see [Cluster setup](#cluster-setup) below).
    - A input folder for each `step`, where skeleton files are provided as templates for writing your own inputs for the respective external programs.
For example, in the `exploration_lammps/`, `labeling_cp2k/` and `training_deepmd/`, you can find the necessecary files to perform the **exploration** with LAMMPS, the **labeling** with CP2K and the **training** with DeePMD-kit (see [Exploration](#exploration), [Labeling](#usage) and [Training](#training) for a detailed description of the **tunneable** keywords).
    - A job folder for each step, where skeleton submission files are provided as template that **ArcaNN** use to launch the different phases in each `step` when they require a HPC machine.
For example, in `job_exploration_lammps_slurm/`, `job_labeling_CP2K_slurm`, `job_training_deepmd_slurm` and the optional `step` `job_test_deepmd_slurm`, you can find basic `Slurm` submission files.

You **must**  adapt these files to ensure they work on your machine (see [Usage](#usage-req)), but be careful not to modify the **replaceable** keywords (every word starting with `_R_` and ending with `_`) that Arcann will replace with user-defined or auto-generated values (e.g., the wall time for labeling calculations, the cluster partition to be used, etc.).

- The `tools/` folder which contains helper scripts and files.
- The `arcann_training/` folder contains all the files that comprise the `ArcaNN Training` code. **We strongly advise against modifying its contents**.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
<div id="getting-started"></div>

## Getting Started ##

<div id="prerequisites"></div>

### Prerequisites ###

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

<div id="installation"></div>

## Installation ##

### If your machine has access to the internet ###

To use `ArcaNN`, you first need to clone or download this repository using the green `Code` button in the top right corner of the repository's main page.
We recommend keeping a local copy of this repository on each computer that will be used to prepare, run, or analyze any part of the iterative training process, though it is not mandatory.

After the download is complete, go to the repository's main folder.
Create a Python environment with all the required packages indicated in the `tools/arcann_conda_linux-64_env.txt` file using the following command:

```bash
conda create --name <ENVNAME> --file tools/arcann_conda_linux-64_env.txt
```

Load this environment with `conda activate <ENVNAME>` and run:

```bash
pip install .
```

That's it! `ARCANN` has now been installed as a module of your `<ENVNAME>` Python environment.

To verify the installation, you can run:

```bash
python -m arcann_training --help
```

This command should print the basic usage message of the code.

After installation, the repository folder can be deleted if you wish.

**Note:** You can also install the program with:

```bash
pip install -e .
```

This way, any modifications to the source files will immediately take effect during program execution.
This method is **only** recommended if you plan to modify the source files, and it requires you to **keep** the repository folder on your machine.

### If your machine does not have access to the internet ###

Download the repository onto a machine that has access to the internet and to your offline working computer.
In a `tmp/` folder (outside of the repository), copy the `arcann_training/tools/download_arcann_environment.sh` and `arcann_training/tools/arcann_conda_linux-64_env.txt.txt` files, then run

```bash
chmod +x download_arcann_environment.sh ; ./download_arcann_environment.sh
```

This script will download all the required Python packages into a `arcann_conda_linux-64_env_offline_files/` folder and create a `arcann_conda_linux-64_env_offline.txt` file.
Then upload this to the working computer along with the `arcann_training` repository:

```bash
rsync -rvu tmp/* USER@WORKMACHINE:/PATH/TO/INSTALLATION/FOLDER/.
rsync -rvu arcann_training USER@WORKMACHINE:/PATH/TO/INSTALLATION/FOLDER/.
```

Now you can simply create the required Python environment:

```bash
conda create --name <ENVNAME> --file arcann_conda_linux-64_env_offline.txt
```

and then install `ArcaNN` as a Python module as detailed above (with pip).

<div id="machine"></div>

## Cluster setup ##

ArcaNN was designed for use on one or several HPC machines whose specificities must be indicated by the user through a `machine.json` file.
You can find a general example file in `arcann_training/examples/user_files/machine.json` that you should modify to adapt it to your setup and that you will need to copy to the `user_files/` folder of your working directory (see [Usage](#Usage) below).
This file is organized as a `JSON` dictionary with one or several keys that designate the different HPC machines, the typical structure looks like this:

```json
{
    "myHPCkeyword1":
        {ENTRIES THAT DESCRIBE HPC 1},
    "myHPCkeyword2":
        {ENTRIES THAT DESCRIBE HPC 2},
    etc.
}
```

Each key of the JSON file is a short string designating the name of the machine (here `"myHPCkeyword1"`, `"myHPCkeyword2"`, etc.).
The associated entry is also a dictionary whose keys are keywords (or further dictionaries of keywords) associated with information needed to run jobs in the corresponding HPC machine.
Let's have a look at the first few entries of the `"myHPCkeyword1"` machine for a SLURM job scheduler:

```json
{
    "myHPCkeyword1":
    {
        "hostname": "myHPC1",
        "walltime_format": "hours",
        "job_scheduler": "slurm",
        "launch_command": "sbatch",
        "max_jobs" : 200,
        "max_array_size" : 500,
        "mykeyword1": {
            "project_name": "myproject",
            "allocation_name": "myallocationgpu1",
            "arch_name": "a100",
            "arch_type": "gpu",
            "partition": "mypartitiongpu1",
            "subpartition": "mysubpartitiongpu1",
            "qos": {"myqosgpu1": 72000, "myqosgpu2": 360000},
            "valid_for":  ["training"],
            "default": ["training"]
        },
        "mykeyword2": { etc. },
        etc.
    }
    etc.
}
```

- `"hostname"` is a substring contained in the output of the following command: `python -c "import socket ; print(socket.gethostname())"`, which should indicate your machine's name.
- `"walltime_format"` is the time unit used to specify the wall time to the cluster.
- `"job_scheduler"` is the name of the job scheduler used in your HPC machine. The code has been extensively used with `Slurm` and has been tested with other schedulers.
- `"launch_command"` is the bash command used for submitting jobs to your cluster (typically `sbatch` in normal `Slurm` setups, but you can adapt it to match your cluster requirements, such as `qsub` for machines running `PBS/Torque`).
- `"max_jobs"` is the maximum number of jobs per user allowed by the job scheduler of your HPC machine.
You can also use this to set a safety limit if your scheduler does not impose one by default.
- `"max_array_size"` is the maximum number of jobs that can be submitted in a single job array (in `Slurm`, the preferred usage of the `ArcaNN` suite relies heavily on arrays to submit jobs).
- The next keyword is the key name of a partition.
It should contain all the information needed to run a job in that partition of your cluster.
The keyword names are self-explanatory.
The keyword `"valid_for"` indicates the steps that can be performed in this partition (possible options include `["training", "freezing", "compressing", "exploration", "test", "labeling"]`).
The `"default"` keyword indicates that this partition of the machine is the default one used (if not explicitly indicated by the user) for the specified steps.

You can add as many partition keywords as you need.
In the above example, `"mykeyword1"` is a GPU partition that uses `A100` `GPU` nodes.
We will use this for every iteration, unless the user explicitly specifies a different partition for the **training** step.
Note that this example assumes that the HPC machine is divided into projects with allocated time (indicated in `"project_name"` and `"allocation_name"`), as is typical in large HPC facilities used by different groups.
If this does not apply to your HPC machine, you don't need to provide these keywords.
Likewise, if there are no partitions or subpartitions, the corresponding keywords need not be provided.
Finally, to use your HPC machine, you will need to provide example submission files tailored to your machine.
These should follow the style of the `examples/user_files/job*/*.sh` files, **keeping the replaceable strings** indicated by a `_R_` prefix and a `_` suffix.
Place these files in the `$WORK_DIR/user_files/` folder, which you must create to use `ArcaNN` for a particular system (see [Usage](#usage)).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
<div id="usage"></div>

# Usage #

At this stage, `arcann_training` is available on your machine, and you have made the necessary changes to adapt it (see [Cluster setup](#cluster-setup)).
You can now start the procedure.
Create an empty directory anywhere you like that will be your iterative training working directory.
We will refer to this directory by the variable name `$WORK_DIR`.

We will describe the **prerequisites**, and then the **initialization**, **training**, **exploration**, **labeling** steps and, the optional **test**.
At the end of each step description, we include an example

<div id="usage-req"></div>

## Iterative procedure prerequisites ##

In `$WORK_DIR`, you should create two folders: `user_files/` and `data/`.

When training a neural network potential (NNP) for a chemical system (or several systems that you want to describe with the same NNP), you will often want to explore the chemical space as diversely as possible.
In `ArcaNN`, this is made possible by the use of **systems**.
A **system** corresponds to a particular way of exploring the chemical space that interests you and will be represented by specific *datasets* within the total training set of the NNP.
A *dataset* corresponds to an ensemble of structures (*e.g.*, atomic positions, types of atoms, box size, etc.) and corresponding labels (*e.g.*, energy, forces, virial).

**Examples**:

- When building an NNP to study liquid water and ice, you will need to include both liquid and solid configurations in your training set.
This can be done by defining two **systems** (*e.g.*, `ice` and `liquid`).
At every iteration, you will perform explorations with each **system**, thus obtaining (after candidate selection and labeling) corresponding `ice` and `liquid` datasets at every iteration.
If you also want to include configurations with self-dissociated molecules, you might explore adding biases using the PLUMED software, for which you need to have a corresponding **system**.

- If you want train an NNP to study a reaction, such as an SN2 reaction, you would like to include in the training set configurations representing the reactant, product and transition states.
In this case, we would start by defining two **systems** (`reactant` and `product`) and generating structures in both bassins by performing several iterations (exploring the chemical space, labeling the generated structures and training the NNP on the extented *dataset*).
Next, you would also want to generate transition structures between the `reactant` and the `product`.
For that, you would need to performed biased explorations with the PLUMED software (see [Exploration](#exploration)) within different **systems**.

You get the idea: you need a subsystem for every kind of chemical composition, physical state (temperature, density, pressure, cell size, etc.), biased reactive pathway, and more that you wish to include in your final training *dataset*.

**Attention**, **systems** are defined once and for all in the [Initialization](#initialization) of the procedure.
Because of this, every time you want to include a new subsystem (such as self-dissociated structures in the first example or transition state structures in the SN2 example), you will need to initialize the procedure again.
This is very simple—you only need to create a new `$WORK_DIR` and include the necessary files in `user_files/` for each extra **system** you want to add.

To initiate the iterative training procedure, you will need to prepare several files.
You must create a `$WORK_DIR/user_files/` folder and store all the files there (**not** in the `arcann_training/examples/user_files/` folder if you kept it!).
You can start with the examples given in `arcann_training/examples/user_files/`:

- The LAMMPS (or i-PI) and CP2K files used for carrying out the exploration and labeling phases of each **system** should also be prepared before initialization and follow the required naming scheme (`SYSNAME.in` for the LAMMPS input file, `SYSNAME.xml` for i-PI, and `[1-2]_SYSNAME_labeling_XXXXX_[cluster].inp` for the two CP2K files required per **system**, where `[cluster]` refers to the short string selected for the labeling cluster in the `machine.json`; see [Labeling](#labeling)).
We **strongly** advise you to create these files starting from those given in the `arcann_training/examples/user_files/` folder, as they must contain replaceable strings for the key parameters that will be updated by the procedure.

- If a **system** requires the use of PLUMED for the explorations, you will need: a `plumed_SYSNAME.dat` where `SYSNAME` refers to the **system** name (additional PLUMED files can be used as `plumed_*_SYSNAME.dat`, which will also be taken into account for explorations).

- A DeePMD-kit JSON file for training also needs to be prepared and named `dptrain_VERSION.json`, where `VERSION` is the DeePMD-kit version that you will use (*e.g.*, `2.1`; currently supported versions are `2.0`, `2.1`, and `2.2`).

- A representative file in the [LAMMPS Data Format](https://docs.lammps.org/2001/data_format.html) format for each **system**, named `SYSNAME.lmp`, where `SYSNAME` refers to the **system** name, and we will refer them as `LMP` files.
This file represent the configuration of the **system**, with the number of atoms and the number of types of atoms, the simulation cell dimension, the atomic masses for each type and the atomic geometry of your **system**.
Theses will be used as starting point for the first exploration.
You can see the expected format for a water molecule below.

```Data
 # WATER

           3  atoms
           2  atom types

      0.000000000000       2.610000000000  xlo xhi
      0.000000000000       2.000000000000  ylo yhi
      0.000000000000       2.000000000000  zlo zhi

Masses

            1   15.99900000             # O
            2   1.00800000              # H

Atoms

         1    1          0.000000000000       0.000000000000       0.000000000000
         2    2          0.979000000000       0.000000000000       0.000000000000
         3    2         -0.326000000000       0.923000000000       0.000000000000
```

**Please note** that the order of the atoms in the `LMP` files **must** be identical for every **system** and **must** match the order indicated in the `"type_map"` keyword of the DeePMD-kit `dptrain_VERSION.json` training file.
Regarding the order of atoms, if we take the water `LMP` exemple: an oxygen atom will be type 1 (in the `LMP` file) and type 0 with the `"type_map": ["O", "H"]'`, in DeePMD-kit.
If we want to add NaCl in another **system**, the Na and Cl types can't be type 1 or 2.
If you are preparing these files with [atomsk](https://atomsk.univ-lille.fr/) from `.xyz` files, you can set the correct simulation cell and atom ordering by providing them in a [properties file](https://atomsk.univ-lille.fr/tutorial_properties.php).

- A `properties` file must be provided and named `properties.txt`.
This is file will be used by `ArcaNN` with `atomsk`.
In the case of your **system** not having the same chemical composition (*e.g.*, NaCl in water and NaBr in water), the propertie file must account for all types to ensure consistant type mapping.

Here an example:

```Data
type
O 1
H 2
Cl 3
Br 4
Na 5

masses
O 15.99900000
H 1.00800000
Cl 35.45000000
Br 79.90400000
Na 22.98900000
```

And your corresponding `LMP` files will have `type 1 2 3 5` for NaCl/H2O, `type 1 2 4 5` for NaBr/H2O.
In the dptrain JSON: `"type_map"` should be `["O", "H", "Cl", "Br", "Na"]`.

- Finally, you also need to prepare at least one initial *dataset* which will be used for your first neural networks training.
This follows DeePMD-kit standards and should contain a `type.raw` file and `set.000/` folder with `box.npy`, `coord.npy`, `energy.npy` and `force.npy` (see [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/master/)).
You can prepare as many initial *datasets* as you wish and they should all be stored in the `$WORK_DIR/data/` folder with a folder name starting with `init_`.

<div id="usage-steps"></div>

## Iterations, Steps and Phases of the Iterative Procedure ##

As described in more detail below, training the NNP proceeds in iterations consisting of three **steps**: exploration, labeling, and training.
Each **step** is broken down into elementary tasks, which we call *phases*.
Each iteration will have three folders: XXX-exploration, XXX-labeling, and XXX-training (*e.g.*, `XXX` is `003` for the 3rd iteration).
Each **step** is executed in its corresponding folder by running, in order, the relevant *phases* with the following command:

```bash
python -m arcann_training STEP_NAME PHASE_NAME 
```

where `STEP_NAME` refers to the current **step** (`initialization`, `exploration`, `labeling`, `training`, or `test`) and `PHASE_NAME` is the specific task that needs to be performed within that **step**.
This will become clearer with examples in the sections below, where each **step** is explained.
The following tables provide a brief description of the *phases* in each **step**, in the correct order.
Since `initialization` has only a single `start` *phase*, which is self-explanatory, it is detailed in the example below.

### Exploration ###

| Phase | Description |
| --- | --- |
| `prepare` | Prepares the folders for running the exploration MDs of all **systems** (automatically generating input files required for each simulation). |
| `launch` | Submits the MD simulation to the specified partition of the cluster, usually with a SLURM array. |
| `check` | Verifies whether the exploration simulations have completed successfully. If any simulations ended abruptly, it indicates which ones, allowing the user to `skip` or `force` them (see [Exploration](#exploration)). |
| `deviate` | Reads the model deviation (maximum deviation between atomic forces predicted by the committee of NN) along the trajectories of each system and identifies configurations that are candidates (deviations within specified boundaries; see [Exploration](#exploration)). |
| `extract` | Extracts a user-defined number of candidate configurations per **system**, saving them to a `SYSNAME/candidates_SYSNAME.xyz` file for labeling and addition to the NNP training set. |
| `clean` | Removes files that are no longer required (optional). |

### Labeling ###

| Phase | Description |
| --- | --- |
| `prepare` | Prepares folders and files to run electronic structure calculations on identified candidates of each **system**, obtaining the energies and forces required to train the NNP. |
| `launch` | Submits the calculations with one SLURM array per **system**. |
| `check` | Verifies that calculations have completed successfully. If any calculations finished abruptly, it writes their index to a text file in the corresponding `SYSNAME/` folder. The user must decide whether to `skip` or resubmit manually each failed calculation before proceeding. |
| `extract` | Extracts necessary information from the CP2K outputs and builds DeePMD-kit "systems"/datasets for each **system** (stored in the `$WORK_DIR/data/` folder). |
| `clean` | Removes files that are no longer required and compresses the calculation outputs into an archive (optional). |

### Training ###

| Phase | Description |
| --- | --- |
| `prepare` | Prepares folders and files for training and the user-defined number of independent NNPs to be used in the next iteration. |
| `launch` | Submits the training calculations using the `dp train` code from DeePMD-kit. |
| `check` | Verifies whether the training has completed successfully. If any traoining ended abruptly, they need to be resubmitted manually to ensure the training finishes correctly. |
| `freeze` | Freezes the NN parameters into a binary file (`.pb` extension for TensorFlow back-end) usable with LAMMPS and Python. This is done with the `dp freeze` code from DeePMD-kit. |
| `check_freeze` | Verifies that the calculations completed successfully. If any calculations finished abruptly, they must be resubmitted manually to ensure freezing completes correctly. |
| `compress` | [Compresses](https://doi.org/10.1021/acs.jctc.2c00102) the NNP by modifying the `.pb` file to enhance performance with minimal loss of accuracy. Uses the `dp compress` code from DeePMD-kit (optional). |
| `check_compress` | Verifies that the calculations completed successfully. If any calculations finished abruptly, they must be resubmitted manually to ensure compressing completes correctly. |
| `increment` | Changes the iteration number in `control` and creates new `exploration`, `labeling`, and `training` folders for the next iteration. |
| `clean` | Removes files that are no longer required (optional). |

### Test ###

| Phase | Description |
| --- | --- |
| `prepare` | Prepares folders and files for testing the performance of the current iteration's NNP on each *dataset* included in the training set.
| `launch` | Submits the testing calculations using the `dp test` code from DeePMD-kit. If you need ["detail files"](https://docs.deepmodeling.com/projects/deepmd/en/r2/test/test.html) generated by `dp test`, include this directly in the `job_test_deepmd` file. |
| `check` | Verifies whether the calculations have completed successfully. |
| `clean` | Removes files that are no longer required (optional). If "detail files" weren't requested, the `XXX-test/` folder will be removed, as all the step information is consolidated in the `control/test_XXX.json` file. Otherwise, the "detail files" will be compressed into .npy format and stored in `XXX-test/`. |

Parameters will need to be defined for most *phases* of each **step** (*e.g.*, length of MD simulations, temperature, number of CPU tasks for labeling calculations, etc.).
This is done via input files in the JSON format.
Executing *phase* without an input file will use all the default values (see the `exploration.json` file in `examples/inputs` for all `exploration` phases) and write them to a `default_json.json` file.
This file serves as a reminder of what the default values are.
After successfully executing a *phase*, a `used_input.json` file will be created, indicating which parameters `ArcaNN` used for that *phase*.
It will be appended with additional parameters after a subsequent *phase* (*e.g.*, after `exploration prepare`, a `used_input.json` is created and appended after `exploration deviate` with parameters specific to the `deviate` phase).
If you want to override the default values for a phase, simply create an `input.json` file with the parameters you want to change.
For example, to override the number of picoseconds for the `exploration prepare` phase, add an `input.json` file like this:

```JSON
{
    "exp_time_ps": 100
}
```

And run or rerun  `exploration prepare`: you will see within the  `used_input.json` that the requested parameters have been read, and that dependent parameters (*e.g.*, walltime) have been adjusted accordingly.

The parameters indicated in an `input.json` file will **always override** the default or auto-calculated ones, and for some of them, the values **will persist**.
For instance, if you provided an override for `max_candidates` in iteration `003`, it will be maintained in iteration `004` without requiring another `input.json`.

We will now describe each **step** of the concurrent learning procedure in detail.

<div id="usage-initialization"></div>

## Initialization ##

Now that you have decided the subsystems that you want to train your NNP on and prepared all the required files you can initialize the `arcann_training_py` procedure by running (from the $WORK_DIR folder):

```bash
python -m arcann_training initialization start 
```

Now it should have generated your first `000-training` directory. In `$WORK_DIR` you will also find a `default_input.json` file that lools like this :

```JSON
{
    "step_name": "initialization",
    "systems_auto": ["SYSNAME1", "SYSNAME2", "SYSNAME3"],
    "nnp_count": 3
}
```

The `"systems_auto"` keyword contains the name of all the subsystems that were found in your `$WORK_DIR/user_files/` (i.e. all files lmp files) directory and `"nnp_count"` is the number of NNP that is used by default in the committee.

The initialization will create several folders. The most important one is the `control/` folder, in which essential data files will be stored throughout the iterative procedure. These files will be written in `.json` format and should NOT be modified. Right after initialization the only file in `control/` is `config.json`, which contains the essential information about your initialization choices (or defaults), such as your subsystem names and options. Finally the `000-training` empty folder should also have been created by the execution of the python script, where you will perform the first iteration of [training](#training).

If at this point you want to modify the datasets used for the first training you simply have to create an `input.json` from the `default_input.json` file and remove or add the system names to the list. You could also change the number of NNP if you wish. Then you only have have to execute the command of the initialization phase again and your `000-training` directory will be updated. 

### EXAMPLE ###

Let's use the above example of a NNP for water and ice that is able to describe water self-dissociation. Suppose that you want 3 subsystems (ice, un-dissociated liquid water, water with a dissociated pair) your `defaut_input.json` file might look like this:

```JSON
{
    "step_name": "initialization",
    "systems_auto": ["ice", "water", "water-reactive"],
    "nnp_count": 3
}
```

Before executing this phase, you will have prepared a data set for each subsystem (not compulsory, but recommended), stored in the data directory: `data/init_ice`, `data/init_water` and `data/init_water-reactive`. In the `user_files/` folder you will have the following scripts:

- `dp_train_2.1.json` for the DeePMD-kit trainings (or any other version with the corresponding name)
- `machine.json` file containing the cluster parameters 
- `ice.in`, `water.in` and `water-reactive.in` LAMMPS inputs
- `ice.lmp`, `water.lmp` and `water-reactive.lmp` starting configurations
- `plumed_water-reactive.dat` plumed file used for biasing only in the reactive system
- `1_ice_labeling_XXXXX_[cluster].inp`, `2_ice_labeling_XXXXX_[cluster].inp`, `1_water_labeling_XXXXX_[cluster].inp`, `2_water_labeling_XXXXX_[cluster].inp`, `1_water-reactive_labeling_XXXXX_[cluster].inp` and `2_water-reactive_labeling_XXXXX_[cluster].inp` CP2K files (there are 2 input files per subsystem, see details in [labeling](#labeling)), where "[cluster]" is the machine keyword indicated in the `machine.json` file.
- `job_lammps-deepmd_explore_gpu_myHPCkeyword1.sh` and `job-array_lammps-deepmd_explore_gpu_myHPCkeyword1.sh` job scripts for exploration, `job_CP2K_label_cpu_myHPCkeyword1.sh` and `job-array_CP2K_label_cpu_myHPCkeyword1.sh` job scripts for labeling, `job_deepmd_compress_gpu_myHPCkeyword1.sh`, `job_deepmd_freeze_gpu_myHPCkeyword1.sh` and `job_deepmd_train_gpu_myHPCkeyword1.sh` job scripts for training
- `dptrain_2.1.json` input for DeePMD /!\ il s'appelle training_2.1.json dans examples/user_files/training_deepmd

<div id="usage-training"></div>

## Training ##

During the training procedure you will use DeePMD-kit to train neural networks on the data sets that you have thus far generated (or on the initial ones only for the 000 iteration). In order to do this go to the current iteration training folder `XXX-training`. 
There are 9 phases (see [Steps](#usage-steps) above) that you must now execute in order after having optionally modified the `input.json` file to define the relevant parameters (in case you want something different from the defaults, which are written to `default_input.json` in the `prepare` phase). The input keywords that you should check the most carefully are those related to the first phase `prepare`, as this sets all the important parameters for the training. Some phases will simply submit `Slurm` jobs (model training, freezing and compressing). You must wait for the jobs to finish before executing the next phase (generally this will be a check phase that will tell you that jobs have failed or are currently running). Once you have executed the first 8 phases the training iteration is done! Executing the 9-th phases is optional, as this will only remove intermediary files.

### EXAMPLE ###

Suppose that you just ran the `initialization` step described in the previous example. You must now perform the first training phase. Update (or copy for the first time) the full `$WORK_DIR` from your local machine to your HPC machine (where you must have also a copy of this repository and an environment in which it is installed):

```bash
rsync -rvu $WORK_DIR USER@HPC-MACHINE:/PATH/TO/WORK_DIR
```

Now go to the empty `000-training` folder created by the script execute the `prepare` phase:

```bash
python -m arcann_training training prepare
```

This will create three folders `1/`, `2/` and `3/` and a copy of your `data/` folder, as well as a `default_input.json` file containing the default training parameters. If you want to modify some of the default values you can create a `input.json` file from the `default_input.json` file that looks like this:

```JSON
{
    "step_name": "training",
    "user_machine_keyword_train": "mykeyword1",
    "user_machine_keyword_freeze": "mykeyword2",
    "user_machine_keyword_compress": "mykeyword2",
    "slurm_email": "",
    "use_initial_datasets": true,
    "use_extra_datasets": false,
    "deepmd_model_version": 2.2,
    "job_walltime_train_h": 4,
    "mean_s_per_step": -1,
    "start_lr": 0.001,
    "stop_lr": 1e-06,
    "decay_rate": 0.9172759353897796,
    "decay_steps": 5000,
    "decay_steps_fixed": false,
    "numb_steps": 400000,
    "numb_test": 0,  
}
```

Here the `"user_machine_keyword"` should match the `"myHPCkeyword1"` keyword in the `machine.json` (see [Cluster Setup](#machine) above). Note that the more performant GPUs should ideally be used for training, while the other steps could be alllocated to less performant GPUs or even to CPUs. Here we used a user chosen walltime of 4 h (instead of the default indicated by `-1`, which will calculate the job walltime automatically based on your previous trainings).
The followiing keywords are the DeePMD training parameters, that you can eventually modify or keep the default values. 
 We can then execute all the other phases in order (waiting for `Slurm` jobs to finish!). That's it! Now you just need to update the local folder:

```bash
rsync -rvu USER@HPC-MACHINE.fr:/PATH/TO/WORK_DIR $WORK_DIR
```

and you are ready to move on to the exploration phase!

**Notes:**

- At some point during the iterative procedure we might want to get rid of our initial data sets, we would only need to set the `use_initial_datasets` variable to `False`
- We might also have generated some data independently from the iterative procedure that we might want to start using, this can be done by copying the corresponding DeePMD-kit systems to `data/`, prefixing their names by `extra_` and setting the `use_extra_datasets` variable to `True`
- At the end of the step the last phase `increment` will create the folders needed for the next iteration, save the current NNPs (stored as graph files `graph_[nnp_count]_XXX[_compressed].pb`) into the `$WORK_DIR/NNP` folder and write a `control/training_XXX.json` file with all parameters used during training.

<div id="usage-exploration"></div>

## Exploration ##

In the exploration phase we will generate new configurations (referred to as **candidates**) to include in the training set. For this we will perform MD simulations with either the LAMMPS (classical nuclei) or i-PI (quantum nuclei) softwares. Go to the current iteration exploration folder `XXX-exploration` created at the end of the previous training phase and execute the `prepare` phase to initialize the exploration. As for the Initialization and Training steps, you could change the default parameters of the exploration by creating a `input.json` file from the `default_input.json` file, modifying the desired values and running the `prepare` phase again. If you want to keep the default values you only need to run the `prepare` phase once. Some of the phases are slightly different if you use LAMMPS or i-PI, both phase workflows are detailed below.

### LAMMPS: classical nuclei simulations ###

Once you are satisfied with your exploration parameters (see example below) you can execute the next exploration phases : `launch` to run MD trajectories with each subsystem  and `check` (once the `Slurm` MD jobs are done!). If the `check` phase is succesfull, you can move on to the `deviate` phase, where you can set important parameters for candidate selection. Once again you can modify these keywors by the creation (or modification if you already created one for a previous phase) of a `default_input.json` file and re-executing the `deviate` phase. In the `extract` phase an important choice can be made: whether to include "disturbed" candidates in the training set or not. This is done by changing the `disturbed_start_value` and `disturbed_candidate_value` variables from the defaults (0.0) and will include a set of candidates generated by applying a random perturbation to those obtained in the MD trajectories (this will multiply by 2 the number of selected candidates, make sure that the `disturbed_start_value` that you choose will still give physically meaningful configurations, otherwise you will deteriorate your NNP!). Once you execute this phase a `candidates_SUBSYS_XXX.xyz` file will be created in each subsystem directory containing the candidate configurations that will be added to the training set (you might want to check that they make sense!). You can also disturb only some atoms in the configuration in which case you will need to write their (zero-based) atomic indices in the `disturbed_candidate_indexes` variable. The `clean` phase can be executed to clean up all the temporary files. A `control/exploration_XXX.json` file will be written recording all the exploration parameters.

### EXAMPLE ###

After the first training phase of your ice & water NNP you now have starting models that can be used to propagate reactive MD. For this go to the `$WORK_DIR/001-exploration` folder (in your HPC machine!) and execute the `prepare` phase to obtain an `defauld_input.json` file with default values. For the first iteration we might be satisfied with the defaults (2 simulations per NNP and per subsystem, 10 ps simulations with the LAMMPS time-step of 0.5 fs, etc.) so we might directly run exploration phases 2 and 3 right away (waiting for the `Slurm` jobs to finish as always). These will have created 3 directories (one per subsystem) `ice/`, `water/` and `water-reactive/`, in which there will be 3 subdirectories (one per trained NNP) `1/`, `2/` and `3/`, in which again there will be 2 subdirectories (default) `0001/` and `0002/`. This means that a total of 18 MD trajectories will be performed for this first iteration (180 ps total simulation time). Be careful, the total exploration time can quickly become huge, especially if you have many subsystems.

Since this is the first exploration phase we might want to generate only a few candidate configurations to check whether our initial NNP are stable enough to give physically meaningful configurations, we might as well want to use a relatively strict error criterion for candidate selection. All this can be done by modifying the default values written to `input.json` at the `deviate` phase and re-running this phase. In the end, your input file might look like this:

```JSON
{
    "step_name": "exploration",
    "user_machine_keyword_exp": "mykeyword1",
    "slurm_email": "",
    "atomsk_path": "PATH_TO_THE_ATOMSK_BINARY",
    "vmd_path": "PATH_TO_THE_VMD_BINARY",
    "exploration_type": ["lammps", "lammps", "lammps"],
    "traj_count": [2, 2, 2],
    "temperature_K": [273.0, 300.0, 300.0],
    "timestep_ps": [0.0005, 0.0005, 0.0005],
    "previous_start": [true, true, true],
    "disturbed_start": [false, false, false],
    "print_interval_mult": [0.01, 0.01, 0.01],
    "job_walltime_h": [-1, -1, -1],
    "exp_time_ps": [10, 10, 10],
    "max_exp_time_ps": [400, 400, 400],
    "max_candidates": [50, 50, 100],
    "sigma_low": [0.1, 0.1, 0.1],
    "sigma_high": [0.8, 0.8, 0.8],
    "sigma_high_limit": [1.5, 1.5, 1.5],
    "ignore_first_x_ps": [0.5, 0.5, 0.5],
    "init_exp_time_ps": [-1, -1, -1],
    "init_job_walltime_h": [-1, -1, -1],
    "disturbed_candidate_value": [0.5, 0, 0],
    "disturbed_start_value": [0.0, 0.0, 0.0],  
    "disturbed_start_indexes": [[], [], []], 
    "disturbed_candidate_indexes": [[], [], []]    
}
```

We have indicated the path to the `Atomsk` code used for creating the disturbed geometries at the beginning of the input file. We allow for slightly larger deviations (`"sigma_high"` keyword set to 0.8 eV/Ang) and collect a larger number of candidates (`"max_candidates"` set to 100) for the more complex third system (reactive water).
At this stage we should decide wether we want to include disturbed candidates in the training set. Here we might want to do so only for the ice system, since explorations at lower temperature explore a more reduced zone of the phase space and it is easier to be trapped in meta-stable states. This can be done by setting `disturbed_start_value` to `0.5`. The values in `disturbed_start_value` are used to disturb the starting structures for the next iteration. For the 2 other systems `disturbed_start_value` and `disturbed_candidate_value` are set to `0.0` in order to avoid disturbance. A non-zero value sets the maximal amplitude of the random translation vector that will be applied to each atom (a different vector for each atom) in Å.  

**Note:** we have indicated the path to a `VMD` executable, this is not needed if `vmd` is inmediately available in our path when executing the `extract` phase (loaded as a module for example). Similarly, we can remove `atomsk_path` if `atomsk` is already in the path.

We can finally clean up the working folder by running the `clean` phase and move on to the labeling phase! (Don't forget to keep your local folder updated so that you can analyze all these results)

### i-PI quantum nuclei simulations ###

Simulations explicitly including nuclear quantum effects by path-integral molecular dynamics with i-PI are quite similar to classical nuclei simulations with LAMMPS. Although the i-PI input files are different (see [i-PI](https://ipi-code.org/)), the `prepare`, `launch` and `check` phases can be done exactly as previously (see [LAMMPS classical nuclei simulations](#lammps-classical-nuclei-simulations) above). Then, before executing the `deviate` and `extract` phases, you must run `select_beads` and `rerun` in this order. These phases do not have special parameters that need to be tuned but require `VMD` and `Atomsk`. After that, you can run the rest of the step as for LAMMPS MD simulations.

<div id="usage-labeling"></div>

## Labeling ##

In the labeling phase we will use the `CP2K` code to compute the electronic energies, atomic forces and (sometimes) the stress tensor of the candidate configurations obtained in the exploration phase. For this we need to go to the `XXX-labeling` folder and as usual run the `prepare` phase. It is very important to have a look at the `default_input.json` of the `prepare` phase to choose the computational resources to be used in the electronic structure calculations (number of nodes and MPI/OpenMP tasks). Note that the default values are insufficient for most condensed systems, so you should have previously determined the resources required by your specific system(s). Once you have executed this phase, folders will have been created for each subsystem within which there will be as many folders as candidate configurations (maximum number of 99999 per iteration), containing all required files to run CP2K. Make sure that you have prepared (and correctly named!) template `Slurm` submission files for your machine in the `$WORK_DIR/user_files` folder ([Initialization](#usage-initialization)). You can then submit the calculations by executing the `launch` phase. Once these are finished you can check the results with  the `check` phase. Since candidate configurations are not always very stable (or even physically meaningful if you were too generous with deviation thresholds) some DFT calculations might not have converged, this will be indicated by the code. You can either perform manually the calculations with a different setup until the result is satisfactory or skip the problematic configurations by creating empty `skip` files in the folders that should be ignored. Keep running `check` until you get a "Success!" message. Use the `extract` phase to set up everything for the training phase and eventually run the `clean` phase to clean up your folder. CP2K wavefunctions might be stored in an archive with a command given by the code that must be executed manually (if one wishes to keep these files as, for example, starting points for higher level calculations). You can also delete all files but the archives created by the code if you want. 

### EXAMPLE ###

After the first exploration phase we recovered 47, 50 and 92 candidates for our `ice`, `water` and `water-reactive` systems for which we must now compute the electronic structure at our chosen reference level of theory (for example revPBE0-D3). We will have prepared (during initialization) 2 `CP2K` scripts for each system, a first quick calculation at a lower level of theory (for example PBE) and then that at our reference level. We will first copy all this data to the HPC machine were we will perform the labeling (where we must have another copy of this repo as well, with a python environment in which the module was installed):

```bash
rsync -rvu $WORK_DIR USER@OTHER_HPC_MACHINE:PATH_TO_WORKDIR
```

 If we are using a larger number of atoms for the reactive system to ensure proper solvation and separation of the ion pair we might need to use more resources for those calculations. In this example we are using 128 CPU nodes of a `"mykeyword1"` partition and the `input.json` file might look something like this:

```JSON
{
    "step_name": "labeling",
    "user_machine_keyword_label": "mykeyword1",
    "job_email": "",
    "walltime_first_job_h": [0.5, 0.5, 0.5],
    "walltime_second_job_h": [1.0, 1.0, 1.5],
    "nb_nodes": [1, 1, 1],
    "nb_mpi_per_node": [32, 32, 64],
    "nb_threads_per_mpi": [2, 2, 2],
}
```

Here the reactive water calculations use full nodes and have a higher wall time of 1h30min. The wall times should be set for the first iteration but can be guessed automatically later using the average time per CP2K calculation measured in the previous iteration. We can now run the first 2 phases and wait for the electronic structure calculations to finish. When running the check phase there could be a message telling us that there are failed configurations in the `water-reactive` folder! We can see which calculations did not converge in the `water-reactive/water-reactive_step2_not_converged.txt` file. Suppose there were 2 failed jobs, the 13-th and the 54-th. We might just do `touch water-reactive/00013/skip` and `touch water-reactive/00054/skip` and run the `check` phase again. This time it will inform us that some configurations will be skipped, but the final message should be that check phase is a success. All that is left to do now is run the `extract` phase, clean up with the `clean` phase, store wavefunctions and remove all unwanted data and finally update our local folder. We have now augmented our total training set and might do a new training iteration and keep iterating until convergence is reached!

<div id="usage-test"></div>

## Test (optional) ##

It is possible to perform tests at every iteration of the learning procedure (the code will create `XXX-test/` folders at every `increment` phase of a `training` step). However, doing this at every iteration is rather time consuming and is not really necessary (although you should obviously test your converged NNP thoroughly). Therefore, documentation on how to test at every iteration within the `arcann_training` procedure is still not ready, sorry!

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
<div id="license"></div>

## License ##

Distributed under the GNU Affero General Public License v3.0. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- FUNDINGS -->
<div id="fundings"></div>

## Fundings & HPC Allocations ##

- Idex ANR-10-IDEX-0001-02PSL
- ERC Grant Agreement No. 757111
- GENCI Grant 2023-A0130707156

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
<div id="acknowledgments"></div>

## Acknowledgments & Sources ##

- [Stackoverflow](https://stackoverflow.com/)

### Beta-testers ###

- Olaia Anton, Zakarya Benayad, Miguel de la Puente, Axel Gomez
- Oscar Gayraud, Pierre Girard, Anne Milet

### Atomsk ###

- Hirel, P. Atomsk: A Tool for Manipulating and Converting Atomic Data Files. Comput. Phys. Commun. 2015, 197, 212–219. [https://doi.org/10.1016/j.cpc.2015.07.012](https://doi.org/10.1016/j.cpc.2015.07.012).

### VMD ###

- Humphrey, W.; Dalke, A.; Schulten, K. VMD: Visual Molecular Dynamics. J. Mol. Graph. 1996, 14 (1), 33–38. [https://doi.org/10.1016/0263-7855(96)00018-5](https://doi.org/10.1016/0263-7855(96)00018-5).

### DeePMD-kit ###

- Zeng, J.; Zhang, D.; Lu, D.; Mo, P.; Li, Z.; Chen, Y.; Rynik, M.; Huang, L.; Li, Z.; Shi, S.; Wang, Y.; Ye, H.; Tuo, P.; Yang, J.; Ding, Y.; Li, Y.; Tisi, D.; Zeng, Q.; Bao, H.; Xia, Y.; Huang, J.; Muraoka, K.; Wang, Y.; Chang, J.; Yuan, F.; Bore, S. L.; Cai, C.; Lin, Y.; Wang, B.; Xu, J.; Zhu, J.-X.; Luo, C.; Zhang, Y.; Goodall, R. E. A.; Liang, W.; Singh, A. K.; Yao, S.; Zhang, J.; Wentzcovitch, R.; Han, J.; Liu, J.; Jia, W.; York, D. M.; E, W.; Car, R.; Zhang, L.; Wang, H. DeePMD-Kit v2: A Software Package for Deep Potential Models. J. Chem. Phys. 2023, 159 (5), 054801. [https://doi.org/10.1103/PhysRevMaterials.3.023804](https://doi.org/10.1063/5.0155600).
- Wang, H.; Zhang, L.; Han, J.; E, W. DeePMD-Kit: A Deep Learning Package for Many-Body Potential Energy Representation and Molecular Dynamics. Comput. Phys. Commun. 2018, 228, 178–184. [https://doi.org/10.1016/j.cpc.2018.03.016](https://doi.org/10.1016/j.cpc.2018.03.016).

### DP-Compress ###

- Lu, D.; Jiang, W.; Chen, Y.; Zhang, L.; Jia, W.; Wang, H.; Chen, M. DP Compress: A Model Compression Scheme for Generating Efficient Deep Potential Models. J. Chem. Theory Comput. 2022, 18 (9), 5559–5567. [https://doi.org/10.1021/acs.jctc.2c00102](https://doi.org/10.1021/acs.jctc.2c00102).

### Concurrent Learning ###

- Zhang, L.; Lin, D.-Y.; Wang, H.; Car, R.; E, W. Active Learning of Uniformly Accurate Interatomic Potentials for Materials Simulation. Phys. Rev. Materials 2019, 3 (2), 023804. [https://doi.org/10.1103/PhysRevMaterials.3.023804](https://doi.org/10.1103/PhysRevMaterials.3.023804)
- Zhang, Y.; Wang, H.; Chen, W.; Zeng, J.; Zhang, L.; Wang, H.; E, W. DP-GEN: A Concurrent Learning Platform for the Generation of Reliable Deep Learning Based Potential Energy Models. Comput. Phys. Commun. 2020, 253, 107206. [https://doi.org/10.1016/j.cpc.2020.107206](https://doi.org/10.1016/j.cpc.2020.107206).

### LAMMPS ###

- Thompson, A. P.; Aktulga, H. M.; Berger, R.; Bolintineanu, D. S.; Brown, W. M.; Crozier, P. S.; In ’T Veld, P. J.; Kohlmeyer, A.; Moore, S. G.; Nguyen, T. D.; Shan, R.; Stevens, M. J.; Tranchida, J.; Trott, C.; Plimpton, S. J. LAMMPS - a Flexible Simulation Tool for Particle-Based Materials Modeling at the Atomic, Meso, and Continuum Scales. Comput. Phys. Commun. 2022, 271, 108171. [https://doi.org/10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171).

### i-PI ###

- Kapil, V.; Rossi, M.; Marsalek, O.; Petraglia, R.; Litman, Y.; Spura, T.; Cheng, B.; Cuzzocrea, A.; Meißner, R. H.; Wilkins, D. M.; Helfrecht, B. A.; Juda, P.; Bienvenue, S. P.; Fang, W.; Kessler, J.; Poltavsky, I.; Vandenbrande, S.; Wieme, J.; Corminboeuf, C.; Kühne, T. D.; Manolopoulos, D. E.; Markland, T. E.; Richardson, J. O.; Tkatchenko, A.; Tribello, G. A.; Van Speybroeck, V.; Ceriotti, M. I-PI 2.0: A Universal Force Engine for Advanced Molecular Simulations. Comput. Phys. Commun. 2019, 236, 214–223. [https://doi.org/10.1016/j.cpc.2018.09.020](https://doi.org/10.1016/j.cpc.2018.09.020).

### CP2K ###

- Kühne, T. D.; Iannuzzi, M.; Del Ben, M.; Rybkin, V. V.; Seewald, P.; Stein, F.; Laino, T.; Khaliullin, R. Z.; Schütt, O.; Schiffmann, F.; Golze, D.; Wilhelm, J.; Chulkov, S.; Bani-Hashemian, M. H.; Weber, V.; Borštnik, U.; Taillefumier, M.; Jakobovits, A. S.; Lazzaro, A.; Pabst, H.; Müller, T.; Schade, R.; Guidon, M.; Andermatt, S.; Holmberg, N.; Schenter, G. K.; Hehn, A.; Bussy, A.; Belleflamme, F.; Tabacchi, G.; Glöß, A.; Lass, M.; Bethune, I.; Mundy, C. J.; Plessl, C.; Watkins, M.; VandeVondele, J.; Krack, M.; Hutter, J. CP2K: An Electronic Structure and Molecular Dynamics Software Package - Quickstep: Efficient and Accurate Electronic Structure Calculations. J. Chem. Phys. 2020, 152 (19), 194103. [https://doi.org/10.1063/5.0007045](https://doi.org/10.1063/5.0007045).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/arcann-chem/arcann_training.svg?style=for-the-badge
[license-url]: https://github.com/arcann-chem/arcann_training/blob/main/LICENSE
