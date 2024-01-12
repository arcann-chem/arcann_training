<div id="top"></div>

<!-- PROJECT SHIELDS -->

[![GNU AGPL v3.0 License][license-shield]][license-url]

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
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
<div id="about"></div>

# About The Project

"Very fancy DeePMD-based semi-automatic highly-customizable iterative training procedure" 
would definitely be the best definition of this repository. It aims at simplifying and automatizing the iterative training process of a [DeePMD-kit](https://doi.org/10.1016/j.cpc.2018.03.016) neural network potential for a user-chosen system. The main advantages of this code are its modularity, the ability to finely tune the training process to adapt to your system and workflow and a great traceability as the code records every parameter set during the procedure. During the iterative training process, you will successively train neural network potentials, use them as reactive force-fields for molecular dynamics simulations (explore the phase space), select and label some configurations based on a *query by committee* approach, and train again neural network potentials with an improved training set, etc. This workflow, sometimes referred to as *active* or *concurrent* learning was heavily inspired by the [DP-GEN scheme](https://doi.org/10.1016/j.cpc.2020.107206) and we use their naming scheme for the "steps" of the iterative procedure. Namely, each iteration will consist of "training", "exploration", "labeling" and (optionally) "testing" phases. Make sure that you understand the meaning of each phase before using the code.

This repository contains several folders:
- `examples/` contains:
  - an `inputs/` folder with 3 json files. These files contain all the keywords that can be given to each of the "steps" followed in an iteration (namely **initialization**, **exploration**, **labeling** and **training**), as well as their type and the default values taken by the code in case the keyword is not provided by the user. If the default is a list containing a single value it means that this value will be used for every **system** (see below). For the exploration step some keywords have 2 default values, the first one will be used if the exploration is conducted with LAMMPS and the second one will be used with i-PI.  
  - a `user_files/` folder with:
    - a `machine.json` file containing all the information about your cluster that the code will need (see [Cluster setup](#cluster-setup) below) 
    - a `jobs/` folder with example `Slurm` submission files that will be used by the code to perform the different steps. You **must** to adapt these files so that they work in your machine, but careful not to modify the **replaceable** keywords (every word starting by `_R_`) that the different codes will replace by the user defined values (ex: wall time of labeling calculation, cluster partition to be used, etc.). 
    - **TO DO:** a `configs/` folder with a typical example of a LAMMPS-compatible configuration file `SYSTEM.lmp` with **replaceable** keywords (see [Usage](#usage))
    - **TO DO:** a `MD_inputs/` folder with typical examples of LAMMPS and i-PI input files with **replaceable** keywords (see [Usage](#usage))
    - **TO DO:** a `CP2K_inputs/` folder with typical examples of CP2K input files with **replaceable** keywords (see [Usage](#usage))
    
- `tools/` contains different scripts needed by the code. We recommend that you do not modify its contents.
- `deepmd_iterative/`: contains all the scripts that make the `deepmd_iterative` code. We recommend that you do not modify its contents.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
<div id="getting-started"></div>

## Getting Started

<div id="prerequisites"></div>

### Prerequisites

<!-- TODO: Prerequisites  -->
For installation:
* python >= 3.6 (all steps)
* numpy>=1.17.3
* setuptools>=40.8.0
* wheel>=0.33.1
* pip>=19.0.3

External requirements for usage:
* DeePMD-kit 2.0 (at least)
* CP2K 7.1 (at least)
* Slurm >= ? (cluster requirement)
* VMD >= 1.9.3 (exploration5_extract)
* Atomsk >= beta-0.11.2 (exploration5_extract)


<div id="installation"></div>

## Installation

To use `deepmd_iterative_py` you first need to clone or download this repository using the `Code` green button on top right corner of the main page of this repository. We recommend that you keep a local copy of this repository on each computer that will be used to prepare, run or analyze any part of the iterative training process. This repository contains important files that are required at different stages of the process and should remain available at all time.

After the download is complete, go to the main folder of the repository. Create a `python` environment (version 3.7.3 at least) containing `pip` with the following command:
```bash
conda create -n ENVNAME python=3.7.3 pip
```
Load this environment with `conda activate ENVNAME` and run:
```bash
pip install .
```
That's it, `deepmd_iterative` has now been installed as a module of your `ENVNAME` python environment ! To verify the installation you can run:
```bash
python -m deepmd_iterative --help
```
which should print the basic usage message of the code.

**Note:** you can also install the program with:
```bash
pip install -e .
```
so that any modifications of the source files will be immediately effective on the execution of the program. This is **only** recommended if you plan to modify the source files.

<div id="machine"></div>

## Cluster setup

This repository was designed for use on Jean Zay (mainly) and Irene Rome, two national French calculators. For a different computer/supercomputer, only some changes need to be made as long as it runs on `Slurm` (if it does not, good luck...):
- generate a `machine_file.json` for your cluster with the various parameters required to submit a `Slurm` job. This file will be placed in your iterative training working directory within the `user_files/` folder (NOT IN THE REPO's `user_files/` FOLDER ! see [Usage](#Usage)). To create this `machine_file.json` file you can copy the one installed with the github repo in your working directory. Let's have a look at a typical machine entry of a `machine_file.json` file:
```json
    "ir":
    {
        "hostname": "irene",
        "walltime_format": "seconds",
        "launch_command": "ccc_msub",
        "cpu_gen7156": {
            "project_name": "gen7156",
            "allocation_name": "rome",
            "arch_name": "cpu",
            "arch_type": "cpu",
            "partition": null,
            "subpartition": null,
            "qos": {"normal": 86400, "long": 259200},
            "valid_for": ["labeling"],
            "default": ["labeling"]
        }
    },
```
Each entry of the `.json` file is a short string designating the name of the machine (here "ir" for Irene-Rome). The associated entry contains several keywords:
  - `"hostname"` is a substring contained in the output of the following command `python -c "import socket ; print(socket.gethostname())"` which should be indicative of your machine's name.
  - `"walltime_format"` is the time unit in which the wall time must be indicated to the cluster.
  - `"launch_command"` is the `bash` command used for submitting jobs to your cluster (typically `sbatch` in normal `Slurm` setups, but as you see in the example you can adapt it to match your cluster requirements)
  - The next keyword is the key name of a partition, it should contain all the information needed to run a job in that partition of your cluster (the names of the keywords are quite self explanatory). The keyword `"valid_for"` indicates the steps of the procedures that can be performed in this partition (possible options are: `["training","freezing","compressing","exploration","test","test_graph","labeling"]`). The `"default"` keyword indicates that this partition of the machine is the default to be used (if not indicated by the user) for the indicated steps. You can add as many partition keywords as you want.

Finally, in order to use your cluster you will need to provide example submission files adequate for your machine (in the same style as those provided in `examples/user_files/jobs/` and **keeping the replaceable strings**) in the `user_files/` folder that you will need to create to use `deepmd_iterative` for a given system (see [Usage](#Usage).) 


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
<div id="usage"></div>

# Usage

At this stage, `deepmd_iterative_py` is available on your computer and you made the necessary changes for your computer (see [Cluster setup](#cluster-setup)). You can now start the procedure. Create an empty directory anywhere you like that will be your iterative training working directory. We will refer to this directory by the variable name `$WORK_DIR`. 

We will describe the prerequisites and then the initialization, training, exploration and labeling phases. At the end of each phase description we include an example.

<div id="usage-req"></div>

## Iterative procedure prerequisites

In `$WORK_DIR` you should create two folders `user_files/` and `data/`

When training a neural network potential (NNP) for a chemical system (or several systems that you want to describe with the same NNP) you will often want to explore the chemical space as diversely as possible. In `deepmd_iterative_py` this is made possible by the use of **subsystems**. A subsystem will correspond to a given way of exploring the chemical space that interests you and will be ultimately associated with specific data sets within the total training set of the NNP. 

**Example**: when building an NNP to study liquid water and ice you will need to include both liquid and solid configurations into your training set, this can be done by defining two subsystems (`ice` and `liquid` for example), at every iteration you will perform explorations with each subsystem, thus getting (after candidate selection and labeling) corresponding `ice` and `liquid` data sets at every iteration (**Note** a "data set" corresponds to what DeePMD people call a "system"). If you also want to add configurations with self-dissociated molecules you might want to explore by adding biases with the PLUMED software, for which you need to have a corresponding subsystem. You get the idea, you need a subsystem for every kind of chemical composition, physical state (temperature, density, pressure, cell size, etc.), biased reactive pathway, etc. that you wish to include in your final training set. 

**Attention**, subsystems are defined once and for all in the [Initialization](#initialization) of the procedure. Therefore, before starting you will need to have prepared the following files:
* a representative **configuration** for each subsystem (which you will name `SYSNAME.lmp`, where `SYSNAME` refers to the subsystem name). A configuration of the subsystem in `.lmp` contains a given atomic geometry of your subsystem (it will be used as starting point for the first exploration), the number of atoms, the simulation cell dimensions and the atomic masses, all in a LAMMPS compatible format (see, for example, `examples/user_files/configs/SYSTEM1.lmp`). 
* If a subsystem will require the use of PLUMED for the explorations you will also need to have prepared all the PLUMED files, that you will name as `plumed_SYSNAME.dat` where `SYSNAME` refers to the subsystem name (additional PLUMED files can be used as `plumed_*_SYSNAME.dat` that will also be taken into account for explorations). 
* The LAMMPS (or i-PI) and CP2K files used for carrying out the exploration and labeling phases of each subsystem should also be prepared before the initialization and follow the required naming scheme (`SYSNAME.in` for the LAMMPS input file, `SYSNAME.xml` for i-PI and `[1-2]_SYSNAME_labeling_XXXXX_[cluster].inp` for the 2 CP2K files required per subsystem, where `[cluster]` refers to the short string selected for the labeling cluster in the `machine_file.json`, see [Labeling](#labeling)). We **strongly** advise you to create these files starting from the ones given in the `examples/user_files/` folder, since they must contain replaceable strings for the key parameters that will be updated by the procedure. 
* A DeePMD-kit `.json` file for training needs also to be prepared and named as `dptrain_VERSION_DESCRIPTOR.json` where `VERSION` is the DeePMD-kit version that you will use (ex: `2.1`, currently supported versions are `2.0`, `2.1` and `2.2`) and `DESCRIPTOR` is the [smooth-edition](https://papers.nips.cc/paper_files/paper/2018/hash/e2ad76f2326fbc6b56a45a56c59fafdb-Abstract.html) strategy used for creating the atomic configuration descriptor (ex: `se2_a` for two-body descriptors with radial and angular information, currently supported descriptors are `se_a`, `se_ar` and `se_e2_a`). All these files must be stored in the `$WORK_DIR/user_files/` folder that you created (**not** in the `examples/user_files/` folder of the repo !)

Finally, you also need to prepare at least one initial training dataset which will be used for your neural networks training. This follows DeePMD-kit standards and should contain a `type.raw` file and `set.000/` folder with `box.npy`, `coord.npy`, `energy.npy` and `force.npy` (see [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/master/)) You can prepare as many initial sets as you wish and they should all be stored in the `$WORK_DIR/data/` folder with a folder name starting with `init_`.


<div id="usage-steps"></div>

## Iterations, Steps and Phases of the Iterative Procedure

As will be described in more detail below, training the NNP proceeds by iterations composed of 3 steps (exploration, labeling and training). Here we decomposed each step into elementary tasks, which we call "phases". Every iteration will be associated with three folders: `XXX-exploration`, `XXX-labeling` and `XXX-training` (ex: `XXX` is `003` for the 3rd iteration). Each step is performed in its corresponding folder by executing, **in order** the corresponding phases with the following command:
```
python -m deepmd_iterative STEP_NAME PHASE_NAME 
```
where `STEP_NAME` is the name of the step that you are currently undergoing (`initialization`, `exploration`, `labeling` and `training`) and `PHASE_NAME` is the task that needs to be performed at this point of the step (it will be clearer with some examples, see the sections corresponding to each step below). In the following tables we briefly describe the phases available in each step in the order in which they must be performed:

### Exploration

| Phase | Description |
| --- | --- |
| `prepare` | Prepare the folders for running the exploration MDs of every systems (it will automatically prepare all the input files for all the simulations that must be run for each system) | 
| `launch` | Submit the MD simulation to the specified partition of the cluster. This is usually done with a slurm array |
| `check` | Verify whether the exploration simulations have run and ended correctly. If some simulations finished abruptly for some reason it indicates which ones. The user can then `skip` or `force` those simulations (see [Exploration](#exploration)) |
| `deviate` | Read the model deviation (maximal deviation between the atomic forces predicted by the committee of NN) along the trajectories of each system and identify which configurations are **candidates** (deviations within the user specified boundaries, see [Exploration](#exploration)) |
| `extract` | Extract a user-defined number of candidate configurations per system (which will be written to a `SYSNAME/candidates_SYSNAME.xyz` file) to be labeled and added to the training set of the NNP |
| `clean` | Remove files that will no longer be necessary (optional) |

### Labeling

| Phase | Description |
| --- | --- |
| `prepare` | Prepare the folders and files for running the electronic structure calculations on the identified candidates of each system to get the energies and forces needed to train the NNP|
| `launch` | Submit the calculations with one slurm array per system |
| `check` | Verify whether the calculations have ended correctly. If some calculations have finished abruptly for some reason it writes their index to a text file in the corresponding `SYSNAME/` folder. The user must decide to either `skip` or manually resubmit each failed calculation before moving forward |
| `extract` | Extract the necessary information from the CP2K outputs and build DeePMD-kit "systems"/data sets for each subsystem (stored in the `$WORK_DIR/data/` folder) |
| `clean` | Remove files that will no longer be necessary and compress the calculation outputs into an archive (optional) |

### Training

| Phase | Description |
| --- | --- |
| `prepare` | Prepare the folders and files for training the user-defined number of independent NNP that will be used in the next iteration |
| `launch` | Submit the training calculations. Uses the `dp train` code of DeePMD-kit |
| `check` | Verify whether the calculations have ended correctly. If some calculations have finished abruptly for some reason you will need to resubmit them manually and ensure that the training finishes correctly |
| `freeze` | Freeze the NN parameters to a binary file that can be used with LAMMPS and Python (`.pb` extension file). Uses the `dp freeze` code of DeePMD-kit |
| `check_freeze` | Verify whether the calculations have ended correctly. If some calculations have finished abruptly for some reason you will need to resubmit them manually and ensure that the freezing finishes correctly |
| `compress` | Compress the NNP by modifying the `.pb` file in order to enhance performances with minimal loss of accuracy. Uses the `dp compress` code of DeePMD-kit. (optional) |
| `check_compress` | Verify whether the calculations have ended correctly. (optional) |
| `increment` | Change the iteration number in `control` and create new `exploration`, `labeling` and `training` folders for the next iteration|
| `clean` | Remove files that will no longer be necessary (optional) |

Parameters will need to be defined for most phases of each step (ex: length of MD simulations, temperature, number of cpu tasks for labeling calculations, etc.). This is done via input files in the `.json` format. Executing the `prepare` phase of each step (except initialization) without an input file will use all the default values (see the `exploration.json` file in `examples/inputs`) and write them to an `input.json` file. You can then modify this file and repeat the `prepare` phase.

We will now describe in detail each of the steps of the active learning procedure.

<div id="usage-initialization"></div>

## Initialization

Now that you have decided the subsystems that you want to train your NNP on and prepared all the files and DeePMD systems required you can initialize the `deepmd_iterative_py` procedure. For this go to `$WORK_DIR` and prepare an input `input.json` file of the form:
```json
{
    "systems_auto": ["SYSNAME1", "SYSNAME2", "SYSNAME3"],
    "nnp_count": 3
}
```
where in `"systems_auto"` you indicate the name of all the subsystems that you want to employ and `"nnp_count"` is the number of NNP that should be used in the committee. Now we only need to execute the only phase of the `initialization` procedure:
```
python -m deepmd_iterative initialization start
```
which will create several folders. The most important one is the `control/` folder, in which essential data files will be stored throughout the iterative procedure. These files will be written in `.json` format and should NOT be modified. Right after initialization the only file in `control/` is `config.json`, which contains the essential information about your initialization choices (or defaults), such as your subsystem names and options.

Finally the `000-training` empty folder should also have been created by the execution of the python script, where you will perform the first iteration of [training](#training).

### EXAMPLE

Let's use the above example of a NNP for water and ice that is able to describe water self-dissociation. Suppose that you want 3 subsystems (ice, un-dissociated liquid water, water with a dissociated pair) ypur `input.json` file might look like this:

```json
{
    "systems_auto": ["ice", "water", "water-reactive"],
    "nnp_count": 3
}
```
Before executing this phase, you will have prepared a data set for each subsystem (not compulsory, but recommended), stored in the data directory: `data/init_ice`, `data/init_water` and `data/init_water-reactive`. In the `user_files/` folder you will have the following scripts:
- `dp_train_2.1_se2_a.json` for the DeePMD-kit trainings (or any other version/descriptor with the corresponding name)
- `ice.in`, `water.in` and `water-reactive.in` LAMMPS inputs
- `ice.lmp`, `water.lmp` and `water-reactive.lmp` starting configurations
- `1_ice_labeling_XXXXX_ir.inp`, `2_ice_labeling_XXXXX_ir.inp`, `1_water_labeling_XXXXX_ir.inp`, `2_water_labeling_XXXXX_ir.inp`, `1_water-reactive_labeling_XXXXX_ir.inp` and `2_water-reactive_labeling_XXXXX_ir.inp` CP2K files (there are 2 input files per subsystem, see details in [labeling](#labeling), here we assume that labeling is performed in a machine indicated with the keyword "ir" in the `machine.json` file)
- `plumed_water-reactive.dat` plumed file used for biasing only in the reactive system


<div id="usage-training"></div>

## Training

During the training procedure you will use DeePMD-kit to train neural networks on the data sets that you have thus far generated (or on the initial ones only for the 000 iteration). In order to do this go to the current iteration training folder `XXX-training`. 
There are 9 phases (see [Steps](#usage-steps`) above) that you must now execute in order after having optionally modified the `input.json` file to define the relevant parameters (in case you want something different from the defaults, which are written to `input.json` in the `prepare` phase). The input keywords that you should check the most carefully are those related to the first phase `prepare`, as this sets all the important parameters for the training. Some phases will simply submit `Slurm` jobs (model training, freezing and compressing). You must wait for the jobs to finish before executing the next phase (generally this will be a check phase that will tell you that jobs have failed or are currently running). Once you have executed the first 8 phases the training iteration is done! Executing the 9-th phases is optional, as this will only remove intermediary files.

### EXAMPLE

Suppose that you just ran the `initialization` step described in the previous example. You must now perform the first training phase in Jean-Zay. Update (or copy for the first time) the full `$WORK_DIR` from your local machine to Jean-Zay (where you must have also a copy of this repository and an environment in which it is installed):
```
rsync -rvu $WORK_DIR USER@jean-zay.idris.fr:/PATH/TO/JZ/WORK_DIR
```
Now go to the empty `000-training` folder created by the script execute the `prepare` phase:
```bash
python -m deepmd_iterative training prepare
```
This will create three folders `1/`, `2/` and `3/` and a copy of your `data/` folder. You might want to modify some of the default values and re-execute this command. For example you might want to use the following input file:
```json
{
    "user_machine_keyword_train": "a100_nvs",
    "user_machine_keyword_freeze": "v100_nvs",
    "user_machine_keyword_compress": "v100_nvs",
    "job_email": "",
    "use_initial_datasets": true,
    "use_extra_datasets": false,
    "deepmd_model_version": 2.2,
    "deepmd_model_type_descriptor": "se_e2_a",
    "start_lr": 0.001,
    "stop_lr": 1e-06,
    "decay_rate": 0.9172759353897796,
    "decay_steps": 5000,
    "decay_steps_fixed": false,
    "numb_steps": 400000,
    "numb_test": 0,
    "job_walltime_train_h": 4,
    "mean_s_per_step": -1
}
```
Here we changed the GPU partition to be used for training (default is `v100`, indicated in the `machine.json` file) and used a user chosen walltime of 4 h (instead of the default indicated by `-1`). We can then execute all the other phases in order (waiting for `Slurm` jobs to finish!). That's it! Now you just need to update the local folder:
```
rsync -rvu USER@jean-zay.idris.fr:/PATH/TO/JZ/WORK_DIR $WORK_DIR
```
and you are ready to move on to the exploration phase!

**Notes:**
- At some point during the iterative procedure we might want to get rid of our initial data sets, we would only need to set the `use_initial_datasets` variable to `False`
- We might also have generated some data independently from the iterative procedure that we might want to start using, this can be done by copying the corresponding DeePMD-kit systems to `data/`, prefixing their names by `extra_` and setting the `use_extra_datasets` variable to `True`
- At the end of the step the last phase `increment` will create the folders needed for the next iteration, save the current NNPs (stored as graph files `graph_?_XXX[_compressed].pb`) into the `$WORK_DIR/NNP` folder and write a `control/training_XXX.json` file with all parameters used during training.

<div id="usage-exploration"></div>

## Exploration

In the exploration phase we will generate new configurations (referred to as **candidates**) to include in the training set. For this we will perform MD simulations with either the LAMMPS (classical nuclei) or i-PI (quantum nuclei) softwares. Go to the current iteration exploration folder `XXX-exploration` created at the end of the previous training phase. Copy the `input.json` file of a previous exploration step and modify it if necessary or execute the `prepare` phase without input to use the defaults (which you can then modify if you want). The phase is slightly different if you use LAMMPS or i-PI, both phase workflows are detailed below.

### LAMMPS classical nuclei simulations

Once you are satisfied with your exploration parameters (see example below) you can execute the exploration phases 1 to 3 (once the `Slurm` MD jobs are done!) to run MD trajectories with each subsystem. In the `deviate` phase you can set important parameters for candidate selection, once you are satisfied you may execute it. See the example below for the important keywords, these are not written at the `prepare` phase but if you run the `deviate` phase with the defaults they will be written to `input.json` (you can then modify them and re-execute the phase). In the `extract` phase an important choice can be made: whether to include "disturbed" candidates in the training set or not. This is done by changing the `disturbed_start_value` and `disturbed_candidate_value` variables from the defaults (0.0) and will include a set of candidates generated by applying a random perturbation to those obtained in the MD trajectories (this will multiply by 2 the number of selected candidates, make sure that the `disturbed_start_value` that you choose will still give physically meaningful configurations, otherwise you will deteriorate your NNP!). Once you execute this script a `candidates_SUBSYS_XXX.xyz` file will be created in each subsystem directory containing the candidate configurations that will be added to the training set (you might want to check that they make sense!). You can also only disturb some atoms in the configuration in which case you will need to write their (zero-based) atomic indices in the `disturbed_candidate_indexes` variable. The `clean` phase can be executed to clean up all the temporary files. A `control/exploration_XXX.json` file will be written recording all the exploration parameters.

### EXAMPLE

After the first training phase of your ice&water NNP you now have starting models that can be used to propagate reactive MD. For this go to the `$WORK_DIR/001-exploration` folder (in Jean-Zay!) and execute the `prepare` phase to obtain an `input.json` file with default values. For the first iteration we might be satisfied with the defaults (2 simulations per NNP and per subsystem, 10 ps simulations with the LAMMPS time-step of 0.5 fs, etc.) so we might directly run exploration phases 2 and 3 right away (waiting for the `Slurm` jobs to finish as always). These will have created 3 directories (one per subsystem) `ice/`, `water/` and `water-reactive/`, in which there will be 3 subdirectories (one per trained NNP) `1/`, `2/` and `3/`, in which again there will be 2 subdirectories (default) `0001/` and `0002/`. This means that a total of 18 MD trajectories will be performed for this first iteration (180 ps total simulation time). Be careful, the total exploration time can quickly become huge, especially if you have many subsystems.

Since this is the first exploration phase we might want to generate only a few candidate configurations to check whether our initial NNP are stable enough to give physically meaningful configurations, we might as well want to use a relatively strict error criterion for candidate selection. All this can be done by modifying the default values written to `input.json` at the `deviate` phase and re-running this phase. In the end, your input file might look like this:
```json
{
    "atomsk_path": "/gpfsdswork/projects/rech/nvs/commun/programs/apps/atomsk/0.11.2/atomsk",
    "user_machine_keyword_exp": "v100_nvs",
    "exploration_type": ["lammps", "lammps", "lammps"],
    "traj_count": [2, 2, 2],
    "timestep_ps": [0.0005, 0.0005, 0.0005],
    "temperature_K": [273.0, 300.0, 300.0],
    "exp_time_ps": [10, 10, 10],
    "max_exp_time_ps": [400, 400, 400],
    "job_walltime_h": [-1, -1, -1],
    "init_exp_time_ps": [-1, -1, -1],
    "init_job_walltime_h": [-1, -1, -1],
    "print_interval_mult": [0.01, 0.01, 0.01],
    "previous_start": [true, true, true],
    "disturbed_start": [false, false, false],
    "job_email": "",
    "max_candidates": [50, 50, 100],
    "sigma_low": [0.1, 0.1, 0.1],
    "sigma_high": [0.8, 0.8, 0.8],
    "sigma_high_limit": [1.5, 1.5, 1.5],
    "ignore_first_x_ps": [0.5, 0.5, 0.5],
    "vmd_path": "/gpfs7kro/gpfslocalsup/prod/vmd/1.9.4a43/bin/vmd_LINUXAMD64",
```
Note that here we allow slightly larger deviations (0.8 eV/Ang) and collect a larger number of candidates (max 100) for the more complex third system (reactive water). Once we have performed this phase all that is left is to decide wether we want to include disturbed candidates in the training set. Here we might want to do so only for the ice system, since explorations at lower temperature explore a more reduced zone of the phase space and it is easier to be trapped in meta-stable states. This can be done by adding the following keywords at the end of the input file before the `extract` phase:
```json
    "disturbed_candidate_value": [0.5, 0, 0],
    "disturbed_start_value": [0.0, 0.0, 0.0],  # this is used to disturb next starting config, not used here
    "disturbed_start_indexes": [[], [], []], # if empty, all indices will be disturbed
    "disturbed_candidate_indexes": [[], [], []]
```
where we had indicated the path to the `Atomsk` code used for creating the disturbed geometries at the beginning of the input file. `disturbed_start_value` and `disturbed_candidate_value` can be set to `0.0` to avoid disturbance. A non-zero value sets the maximal amplitude of the random translation vector that will be applied to each atom (a different vector for each atom) in Å. The values in `disturbed_start_value` are used to disturb the starting structures for the next iteration. The values in `disturbed_candidate_value` are applied to each candidate.

**Note:** we have indicated the path to a `VMD` executable, this is not needed if `vmd` is inmediately available in our path when executing the `extract` phase (loaded as a module for example). Similarly, we can remove `atomsk_path` if `atomsk` is already in the path.

We can finally clean up the working folder by running the `clean` phase and move on to the labeling phase! (Don't forget to keep your local folder updated so that you can analyze all these results)

### i-PI quantum nuclei simulations NEEDS TO BE UPDATED

Simulations explicitly including nuclear quantum effects by path-integral molecular dynamics with i-PI are quite similar to classical nuclei simulations with LAMMPS. Although the input files are different (see `examples/i-PI_exploration/*.xml`), the preparation, launch and check phases (`exploration1_prep.py`, `exploration2_launch.py` and `exploration3_check.py`) can be done exactly as previously (see [LAMMPS classical nuclei simulations](#lammps-classical-nuclei-simulations) above). Then, before executing `exploration4_devi.py` and `exploration5_extract.py`, you must run `explorationX_selectbeads.py`, `explorationX_rerun.py` and `explorationX_recheck.py` in this order. These 3 scripts do not have options or special parameters that need to be tuned but require `VMD` and `Atomsk`. After that, you can run `exploration4_devi.py`, `exploration5_extract.py` and `exploration9_clean.py` as for LAMMPS MD simulations.

<div id="usage-labeling"></div>

## Labeling

In the labeling phase we will use the `CP2K` code to compute the electronic energies, atomic forces and (sometimes) the stress tensor of the candidate configurations obtained in the exploration phase. For this we need to go to the `XXX-labeling` folder and copy the corresponding input file (or generate it by running `prepare` with the default values). It is very important to have a look at the default input of the `prepare` phase to choose the computational resources to be used in the electronic structure calculations (number of nodes and MPI/OpenMP tasks). Note that defaults are insufficient for most condensed systems, you should have previously determined the resources required by your specific system(s). Once you have executed this phase, folders will have been created for each subsystem within which there will be as many folders as candidate configurations (maximum number of 99999 per iteration), containing all required files to run CP2K. Make sure that you have prepared (and correctly named!) template `Slurm` submission files for your machine in the `$WORK_DIR/user_files` folder. You can then submit the calculations by executing the `launch` script. Once these are finished you can check the results with `check`. Since candidate configurations are not always very stable (or even physically meaningful if you were too generous with deviation thresholds) some DFT calculations might not have converged, this will be indicated by the code. You can either perform manually the calculations with a different setup until the result is satisfactory or skip the problematic configurations by creating empty `skip` files in the folders that should be ignored. Keep running `check` until you get a "Success!" message. Use the `extract` phase to set up everything for the training phase and eventually run the `clean` script to clean up your folder. CP2K wavefunctions might be stored in an archive with a command given by the code that must be executed manually (if one wishes to keep these files as, for example, starting points for higher level calculations). You can also delete all files but the archives created by the code if you want. 

### EXAMPLE

After the first exploration phase we recovered 47, 50 and 92 candidates for our `ice`, `water` and `water-reactive` systems for which we must now compute the electronic structure at our chosen reference level of theory (for example revPBE0-D3). We will have prepared (during initialization) 2 `CP2K` scripts for each system, a first quick calculation at a lower level of theory (for example PBE) and then that at our reference level. We will first copy all this data to the machine were we will perform the labeling (here Irene, where we must have another copy of this repo as well, with a python environment in which the module was installed):
```
rsync -rvu $WORK_DIR USER@irene-amd-fr.ccc.cea.fr:PATH/TO/IRENE/WORK_DIR
```

 If we are using a larger number of atoms for the reactive system to ensure proper solvation and separation of the ion pair we might need to use more resources for those calculations. If we are using the 128 CPU nodes of the `rome` partition our `input.json` file might look something like this:
```json
{
    "user_machine_keyword_label": "cpu_gen7156",
    "walltime_first_job_h": [0.5, 0.5, 0.5],
    "walltime_second_job_h": [1.0, 1.0, 1.5],
    "nb_nodes": [1, 1, 1],
    "nb_mpi_per_node": [32, 32, 64],
    "nb_threads_per_mpi": [2, 2, 2],
    "job_email": ""
}
```

where the reactive water calculations use full nodes and have a higher wall time of 1h30min. The wall times should be set for the first iteration but can be guessed automatically later using the average time per CP2K calculation measured in the previous iteration. We can now run the first 2 phases and wait for the electronic structure calculations to finish. When running the check file the script might tell us that there are failed configurations in the `water-reactive` folder! we can see which calculations did not converge in the `water-reactive/water-reactive_step2_not_converged.txt` file. Suppose there were 2 failed jobs, the 13-th and the 54-th. We might just do `touch water-reactive/00013/skip` and `touch water-reactive/00054/skip` and run the `check` phase again. This time it will inform us that some configurations will be skipped, but the final message should be that check phase is a success. All that is left to do now is run the `extract` phase, clean up with the `clean` phase, store wavefunctions and remove all unwanted data and finally update our local folder. We have now augmented our total training set and might do a new training iteration and keep iterating until convergence is reached!

<div id="usage-test"></div>

## Test (optional)

<!-- TODO test-->

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
<div id="license"></div>

## License

Distributed under the GNU Affero General Public License v3.0. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
<div id="acknowledgments"></div>

## Acknowledgments & Sources

* [Stackoverflow](https://stackoverflow.com/)
* Hirel, P. Atomsk: A Tool for Manipulating and Converting Atomic Data Files. Comput. Phys. Commun. 2015, 197, 212–219. [https://doi.org/10.1016/j.cpc.2015.07.012](https://doi.org/10.1016/j.cpc.2015.07.012).
* Humphrey, W.; Dalke, A.; Schulten, K. VMD: Visual Molecular Dynamics. J. Mol. Graph. 1996, 14 (1), 33–38. [https://doi.org/10.1016/0263-7855(96)00018-5](https://doi.org/10.1016/0263-7855(96)00018-5).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/laagegroup/0_Template.svg?style=for-the-badge
[license-url]: https://github.com/laagegroup/0_Template/blob/main/LICENSE
