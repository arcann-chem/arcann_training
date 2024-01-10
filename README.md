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
  - an `inputs/` folder with 3 json files. These files contain all the keywords that can be given to each of the "steps" followed in an iteration (namely **exploration**, **labeling** and **training**), as well as their type and the default values taken by the code in case the keyword is not provided by the user. If the default is a list containing a single value it means that this value will be used for every **system** (see below). For the exploration step some keywords have 2 defau
lt values, the first one will be used if the exploration is conducted with LAMMPS and the second one will be used with i-PI.  
  - a `user_files/` folder with a `machine.json` file containing all the information about your cluster that the code will need (see [Cluster setup](#cluster-setup) below) and a `jobs/` folder with example `Slurm` submission files that will be used by the code to perform the different steps. You **must** to adapt these files so that they work in your machine, but careful not to modify the **replaceable** keywords (every word starting by `_R_`) that the different codes will replace by the user defined values (ex: wall time of labeling calculation, cluster partition to be used, etc.). This can be used as a starting point when using this semi-automatic procedure (see [Initialization](#initialization) below).
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

When training a neural network potential (NNP) for a chemical system (or several systems that you want to describe with the same NNP) you will often want to explore the chemical space as diversely as possible. In `deepmd_iterative_py` this is made possible by the use of **subsystems**. A subsystem will correspond to a given way of exploring the chemical space that interests you and will be ultimately associated with specific data sets within the total training set of the NNP. 

**Example**: when building an NNP to study liquid water and ice you will need to include both liquid and solid configurations into your training set, this can be done by defining two subsystems (`ice` and `liquid` for example), at every iteration you will perform explorations with each subsystem, thus getting (after candidate selection and labeling) corresponding `ice` and `liquid` data sets at every iteration (**Note** a "data set" corresponds to what DeePMD people call a "system"). If you also want to add configurations with self-dissociated molecules you might want to explore by adding biases with the PLUMED software, for which you need to have a corresponding subsystem. You get the idea, you need a subsystem for every kind of chemical composition, physical state (temperature, density, pressure, cell size, etc.), biased reactive pathway, etc. that you wish to include in your final training set. 

**Attention**, subsystems are defined once and for all in the [Initialization](#initialization) of the procedure. Therefore, before starting you will need to have prepared a representative **configuration** for each subsystem (which you will name `SYSNAME.lmp`, where `SYSNAME` refers to the subsystem name). A configuration of the subsystem contains a given atomic geometry of your subsystem (it will be used as starting point for the first exploration), the number of atoms, the simulation cell dimensions and the atomic masses, all in a LAMMPS compatible format (see, for example, `examples/shared/SYSTEM1.lmp`). If a subsystem will require the use of PLUMED for the explorations you will also need to have prepared all the PLUMED files, that you will name as `plumed_SYSNAME.dat` where `SYSNAME` refers to the subsystem name (additional PLUMED files can be used as `plumed_*_SYSNAME.dat` that will also be taken into account for explorations, see `examples/shared/plumed*SYSTEM2.dat`). The LAMMPS (or i-PI) and CP2K files used for carrying out the exploration and labeling phases of each subsystem should also be prepared before the initialization and follow the required naming scheme (`SYSNAME.in` for the LAMMPS input file, `SYSNAME.xml` for i-PI and `[1-2]_SYSNAME_labeling_XXXXX_[cluster].inp` for the 2 CP2K files required per subsystem, where `[cluster]` refers to the short string selected for the labeling cluster in the `machine_file.json`, see [Labeling](#labeling)). We **strongly** advise you to create this files starting from the ones given in the `examples/` folder, since they must contain replaceable strings for the key parameters that will be updated by the procedure. 
 A DeePMD-kit `.json` file for training needs also to be prepared and named as `VERSION_DESCRIPTOR.json` where `VERSION` is the DeePMD-kit version that you will use (ex: `2.1`, currently supported versions are `2.0` and `2.1` — see file `scripts/training1_prep.py`) and `DESCRIPTOR` is the [smooth-edition](https://papers.nips.cc/paper_files/paper/2018/hash/e2ad76f2326fbc6b56a45a56c59fafdb-Abstract.html) strategy used for creating the atomic configuration descriptor (ex: `se2_a` for two-body descriptors with radial and angular information, currently supported descriptors are `se_a`, `se_ar` and `se_e2_a` — see file `scripts/training1_prep.py`).

Finally, you also need to prepare at least one initial training dataset which will be used for your neural networks training. This follows DeePMD-kit standards and should contain a `type.raw` file and `set.000/` folder with `box.npy`, `coord.npy`, `energy.npy` and `force.npy` (see [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/master/)) You can prepare as many initial sets as you wish.

<div id="usage-initialization"></div>

## Initialization

Now that you have decided the subsystems that you want to train your NNP on and prepared all the files and DeePMD systems required you can initialize the `deepmd_iterative_py` procedure. For this go to `$WORK_DIR` and create to folders `inputs/` and `data/`. Copy your initial data sets (folders containing `type.raw` and `set.000/` folder) to the `data/` folder and add the prefix `init_` to their names. Copy all the input files and configurations that you prepared to the `inputs/` folder (respect the naming convention!). All you need to do now is copy the initialization script to your working directory (we assume that you installed or linked the github repo in your home directory):
```
cp ~/deepmd_iterative_py/scripts/initialization.py $WORK_DIR/.
```
Open the `initialization.py` script and define:
- `sys_name`, a string containing the name of the NNP that you want to create (it is a general purpose name describing the NNP, its choice is unimportant)
- `subsys_name`, a list of strings containing the names of all your subsystems. The number and name of the subsystems indicated here must match **exactly** those indicated in your input and configuration files! **Note**: try to avoid the use of underscores (`_`) since these are used by the code to search for patterns.
- `nb_nnp`, an integer (`>=2`) indicating the number of neural networks trained at each iteration. The deviation between these networks predictions are used to select candidates and each network will be used to propagate trajectories in the exploration phase. **DEFAULT** is 3.
- `exploration_type`, a string defining the software used to perform explorations. Only `"lammps"` and `"i-PI"` are allowed. **DEFAULT** is `"lammps"`.
- `temperature_K`, a list of floats containing the temperature at which the exploration of each subsystem will be performed. **DEFAULT** is 298.15 K for all subsystems (for i-PI explorations, this should be the same values as SUBSYS.xml input files).
- `timestep_ps`, a list of floats with the MD time-step used for the exploration in each subsystem. **DEFAULT** is 0.5 fs for all subsystems in LAMMPS and 0.25 fs in i-PI.
- `nb_candidates_max`, an integer defining the maximum number of candidate configurations that can be extracted per subsystem at a given exploration phase. **DEFAULT** is 500.
- `s_low`, `s_high` and `s_high_max` floats, defining the deviations threshold (lower and upper) for candidate selection. `s_high_max` defines an extra threshold, if deviations ever reach above it the rest of the exploration trajectory is discarded (this is used to prevent non-physical recombination). **DEFAULTS** are 0.1, 0.8 and 1.0 (in eV/Å)
- `ignore_first_x_ps` float, defines the length of trajectory that should be discarded as equilibration because it is too close to the initial configuration. **DEFAULT** is 0.5 ps.

All the keywords with default values can be changed at each iteration in the corresponding scripts (see below).

You can thus initialize the procedure by running (you need to have python3 and all the required packages loaded):
```
python initialization.py
```
which will create several folders. The most important one is the `control/` folder, in which essential data files will be stored throughout the iterative procedure. These files will be written in `.json` format and should NOT be modified (thus, to read them we recommend to download the [jq](https://stedolan.github.io/jq/) program and run `cat FILE.json | /PATH/TO/JQ/jq`). Right after initialization the files that are created are in `control/`:
- `control.json` which contains the essential information about your initialization choices (or defaults), such as your subsystem names and options.
- `path` a text file containing the path to the installation of `deepmd_iterative_py`. This is the file that you should modify if you installed the repo in a different folder than your home (and you should change it accordingly in every machine that you will use or you can define the `deepmd_iterative_apath` variable at the beginning of each script)

Finally `000-training` and `000-test` empty folders should also have been created by the execution of the python script, where you will perform the first iteration of [training](#training) (and eventually testing, although this is seldom useful at the beginning of the procedure)

### EXAMPLE

Let's use the above example of a NNP for water and ice that is able to describe water self-dissociation. We will use classical nuclei MD and perform the labeling on Irene and the training and explorations in Jean-Zay while keeping the original repo always updated in our local machine. Suppose that you want 3 subsystems (ice, un-dissociated liquid water, water with a dissociated pair), the header of your `initialization.py` file might look like this:

```python
## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Set your system name, subsystem ("easy" exploration, standard TEMP, presents from the START of the iterative training) and the number of NNP you want to use
sys_name: str = "water_phases_NNP"
subsys_name: list = ["ice", "water", "water-reactive"]
## These are the default
# nb_nnp: int = 3
# exploration_type: str = "lammps"
temperature_K: list = [250.,300.,300.]
timestep_ps: list = [0.00025, 0.00025, 0.00025] #float #LAMMPS
# timestep_ps: list = [0.00025, 0.00025] #float #i-PI
# nb_candidates_max = [500, 500]
# s_low: list = [0.1, 0.1]
# s_high: list = [0.8, 0.8]
# s_high_max: list = [1.0, 1.0]
# ignore_first_x_ps: list = [0.5, 0.5]
```
where only some of the default keywords have been modified. Before running the script you will have prepared a data set for each subsystem (not compulsory, but recommended), stored in the data directory: `data/init_ice`, `data/init_water` and `data/init_water-reactive`. In the `inputs/` folder you will have the following scripts:
- `2.1_se2_a.json` for the DeePMD-kit trainings (or any other version/descriptor with the corresponding name)
- `ice.in`, `water.in` and `water-reactive.in` LAMMPS inputs
- `ice.lmp`, `water.lmp` and `water-reactive.lmp` starting configurations
- `1_ice_labeling_XXXXX_ir.inp`, `2_ice_labeling_XXXXX_ir.inp`, `1_water_labeling_XXXXX_ir.inp`, `2_water_labeling_XXXXX_ir.inp`, `1_water-reactive_labeling_XXXXX_ir.inp` and `2_water-reactive_labeling_XXXXX_ir.inp` CP2K files (there are 2 input files per subsystem, see details in [labeling](#labeling))
- `plumed_water-reactive.dat` plumed file used for biasing only in the reactive system

The lists `temperature_K`, `timestep_ps`, `nb_candidates_max`, `s_low`, `s_high`, `s_high_max` and `ignore_first_x_ps` should have the same length as `subsys_name` (if commented, the default value will be used for each subsytem).

<div id="usage-training"></div>

## Training

During the training procedure you will use DeePMD-kit to train neural networks on the data sets that you have thus far generated (or on the initial ones only for the 000 iteration). In order to do this go to the current iteration training folder `XXX-training`. Copy all training scripts to this folder:
```
cp ~/deepmd_iterative_py/scripts/training*.py $WORK_DIR/XXX-training/.
```
There are 9 such scripts that you must now execute in order after having optionally modified their headers to define the relevant parameters (in case you want something different from the defaults defined during the initialization). The script that you should check the most carefully is the first one `training1_prep.py`, as this sets all the important parameters for the training. Some scripts will submit `Slurm` jobs (model training, freezing and compressing). You must wait for the jobs to finish before executing the next script (generally this will be a check script that will tell you that jobs have failed or are currently running). Once you have executed the first 8 scripts the training iteration is done! Using the 9-th script is optional, as this will only remove intermediary files.

### EXAMPLE

Suppose that you just ran the `initialization.py` file described in the previous example. You must now perform the first training phase in Jean-Zay. Update (or copy for the first time) the full `$WORK_DIR` from your local machine to Jean-Zay (where you must have also a copy of this repository, ideally in your home directory):
```
rsync -rvu $WORK_DIR USER@jean-zay.idris.fr:/PATH/TO/JZ/WORK_DIR
```
Now go to the empty `000-training` folder created by the script and copy the training scripts as described above. We might want to modify the `training1_prep.py` file to modify some parameters:
```python
## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Either shortcut (machine_file.json) or Project name / allocation / arch
user_spec = "a100"
# user_spec = ["nvs","v100","v100"]
# slurm_email: str = ""
## Training Parameters (Here are the default defaults)
# use_initial_datasets: bool = True
# use_extra_datasets: bool = False
# start_lr: float = 0.001
# stop_lr: float = 1e-06
# decay_rate: float = 0.90
# decay_steps: int = 5000
# numb_steps: int = 400000
# numb_test: int = 0
# deepmd_model_version: float = 2.1
# deepmd_model_type_descriptor: str = "se_e2_a"
## Guess for initial training walltime
initial_seconds_per_1000steps: float = 50
```
Here we changed the GPU partition to be used (default is `v100`, indicated in the `machine_file.json`) and slightly decreased the `initial_seconds_per_1000steps` variable, used to set the `Slurm` wall time. We can then execute all scripts in order (waiting for `Slurm` jobs to finish!). That's it! Now you just need to update the local folder:
```
rsync -rvu USER@jean-zay.idris.fr:/PATH/TO/JZ/WORK_DIR $WORK_DIR
```
and you are ready to move on to the exploration phase!

**Notes:**
- At some point during the iterative procedure we might want to get rid of our initial data sets, we would only need to set the `use_initial_datasets` variable to `False`
- We might also have generated some data independently from the iterative procedure that we might want to start using, this can be done by copying the corresponding DeePMD-kit systems to `data/`, prefixing their names by `extra_` and setting the `use_extra_datasets` variable to `True`
- At the end of the phase the last script `training8_update_iter.py` will create the folders needed for the next iteration, save the current NNPs (stored as graph files `graph_?_XXX[_compressed].pb`) into the `$WORK_DIR/NNP` folder and write a `control/training_XXX.json` file with all parameters used during training.

<div id="usage-exploration"></div>

## Exploration

In the exploration phase we will generate new configurations (referred to as **candidates**) to include in the training set. For this we will perform MD simulations with either the LAMMPS (classical nuclei) or i-PI (quantum nuclei) softwares. Go to the current iteration exploration folder `XXX-exploration` created at the end of the previous training phase. Copy all exploration scripts to this folder and have a look at the `exploration1_prep.py` to check the exploration parameters. The phase is slightly different if you use LAMMPS or i-PI, both phase workflows are detailed below.

### LAMMPS classical nuclei simulations

Once you are satisfied with your exploration parameters (see example below) you can execute the exploration files 1 to 3 (once the `Slurm` MD jobs are done!) to run MD trajectories with each subsystem. In `exploration4_devi.py` you can set important parameters for candidate selection, once you are satisfied you may execute it. In `exploration5_extract.py` an important choice can be made: whether to include "disturbed" candidates in the training set or not. This is done by changing the `disturbed_min_value` and `disturbed_candidates_value` variables from the defaults (0.0) and will include a set of candidates generated by applying a random perturbation to those obtained in the MD trajectories (this will multiply by 2 the number of selected candidates, make sure that the `disturbed_min_value` that you choose will still give physically meaningful configurations, otherwise you will deteriorate your NNP!). Once you execute this script a `candidates_SUBSYS_XXX.xyz` file will be created in each subsystem directory containing the candidate configurations that will be added to the training set (you might want to check that they make sense!). The `exploration9_clean.py` script can be executed to clean up all the temporary files. A `control/exploration_XXX.json` file will be written recording all the exploration parameters.

### EXAMPLE

After the first training phase of your ice&water NNP you now have starting models that can be used to propagate reactive MD. For this go to the `$WORK_DIR/001-exploration` folder (in Jean-Zay!) and copy the exploration scripts from the repository. For the first iteration we might be satisfied with the options selected during initialization and the defaults (2 simulations per NNP and per subsystem, 10 ps simulations with the time-step chosen during initialization, here 0.25 fs, etc.) so we might directly run exploration scripts 1 to 3 (waiting for the `Slurm` jobs to finish as always). These will have created 3 directories (one per subsystem) `ice/`, `water/` and `water-reactive/`, in which there will be 3 subdirectories (one per trained NNP) `1/`, `2/` and `3/`, in which again there will be 2 subdirectories (option chosen during initialization) `0001/` and `0002/`. This means that a total of 18 MD trajectories will be performed for this first iteration (180 ps total simulation time). Be careful, the total exploration time can quickly become huge, especially if you have many subsystems.

Since this is the first exploration phase we might want to generate only a few candidate configurations to check whether our initial NNP are stable enough to give physically meaningful configurations, we might as well want to use a relatively strict error criterion for candidate selection. All this can be done by modifying the header of the `exploration4_devi.py` script:
```python
## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## These are the default
nb_candidates_max: list= [50, 50, 100] #int
# s_low: list = [0.1, 0.1] #float
s_high: list = [0.5, 0.5, 0.8] #float
s_high_max: list = [1.0, 1.0, 1.5] #float
# ignore_first_x_ps: list = [0.5, 0.5] #float
```
Note that here we allow slightly larger deviations (0.8 eV/Ang) and collect a larger number of candidates (max 100) for the more complex third system (reactive water). Once we have executed this script all that is left is to decide wether we want to include disturbed candidates in the training set. Here we might want to do so only for the ice system, since explorations at lower temperature explore a more reduced zone of the phase space and it is easier to be trapped in meta-stable states. This can be done by modifying the `exploration5_extract.py` header:
```python
## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## These are the default
atomsk_fpath: str ="/gpfswork/rech/nvs/commun/programs/apps/atomsk/0.11.2/bin/atomsk"
# vmd_fpath: str=""
disturbed_min_value: list = [0.5, 0.0, 0.0] #float
disturbed_candidates_value: list = [0.5, 0.0, 0.0] #float
```
where we have indicated the path to the `Atomsk` code used for creating the disturbed geometries. `disturbed_min_value` and `disturbed_candidates_value` are `0.0` to avoid disturbance. A non-zero value sets the maximal amplitude of the random translation vector that will be applied to each atom (a different vector for each atom) in Å. The values in `disturbed_min_value` are used to disturb the starting structures for the next iteration. The values in `disturbed_candidates_value` are applied to each candidate.

**Note:** we have not indicated the path to a `VMD` executable, meaning that `vmd` should be in our path when executing the `exploration5_extract.py` script (loaded as a module for example). Similarly, we can comment `atomsk_fpath` if `atomsk` is already in the path.

We can finally clean up the working folder by running the `exploration9_clean.py` script and move on to the labeling phase! (Don't forget to keep your local folder updated so that you can analyze all these results)

### i-PI quantum nuclei simulations

Simulations explicitly including nuclear quantum effects by path-integral molecular dynamics with i-PI are quite similar to classical nuclei simulations with LAMMPS. Although the input files are different (see `examples/i-PI_exploration/*.xml`), the preparation, launch and check phases (`exploration1_prep.py`, `exploration2_launch.py` and `exploration3_check.py`) can be done exactly as previously (see [LAMMPS classical nuclei simulations](#lammps-classical-nuclei-simulations) above). Then, before executing `exploration4_devi.py` and `exploration5_extract.py`, you must run `explorationX_selectbeads.py`, `explorationX_rerun.py` and `explorationX_recheck.py` in this order. These 3 scripts do not have options or special parameters that need to be tuned but require `VMD` and `Atomsk`. After that, you can run `exploration4_devi.py`, `exploration5_extract.py` and `exploration9_clean.py` as for LAMMPS MD simulations.

<div id="usage-labeling"></div>

## Labeling

In the labeling phase we will use the `CP2K` code to compute the electronic energies, atomic forces and (sometimes) the stress tensor of the candidate configurations obtained in the exploration phase. For this we need to go to the `XXX-labeling` folder and copy all the labeling scripts. It is very important to have a look at the header of the `labeling1_prep.py` script to choose the computational resources to be used in the electronic structure calculations (number of nodes and MPI/OpenMP tasks). Note that defaults are insufficient for most condensed systems, you should have previously determined the resources required by your specific system(s). Once you have submitted this script, folders will have been created for each subsystem within which there will be as many folders as candidate configurations (maximum number of 99999 per iteration), containing all required files to run CP2K. Make sure that you have prepared (and correctly named!) template `Slurm` submission files for your machine in the `~/deepmd_iterative_py/jobs/` folder. You can then submit the calculations by executing the `labeling2_launch.py` script. Once these are finished you can check the results with `labeling3_check.py`. Since candidate configurations are not always very stable (or even physically meaningful if you were too generous with deviation thresholds) some DFT calculations might not have converged, this will be indicated by the code. You can either perform manually the calculations with a different setup until the result is satisfactory or skip the problematic configurations by creating empty `skip` files in the folders that should be ignored. Keep running `labeling3_check.py` until you get a "Success!" message. Use the `labeling4_extract.py` file to set up everything for the training phase and eventually run the `labeling9_clean.py` script to clean up your folder. CP2K wavefunctions might be stored in an archive with a command given by the code that must be executed manually (if one wishes to keep these files as, for example, starting points for higher level calculations). You can also delete all files but the archives created by the code if you want. 

### EXAMPLE

After the first exploration phase we recovered 47, 50 and 92 candidates for our `ice`, `water` and `water-reactive` systems for which we must now compute the electronic structure at our chosen reference level of theory (for example revPBE0-D3). We will have prepared (during initialization) 2 `CP2K` scripts for each system, a first quick calculation at a lower level of theory (for example PBE) and then that at our reference level. We will first copy all this data to the machine were we will perform the labeling (here Irene, where we must have another copy of this repo as well):
```
rsync -rvu $WORK_DIR USER@irene-amd-fr.ccc.cea.fr:PATH/TO/IRENE/WORK_DIR
```

 If we are using a larger number of atoms for the reactive system to ensure proper solvation and separation of the ion pair we might need to use more resources for those calculations. If we are using the 128 CPU nodes of the `rome` partition our `labeling1_prep.py` script might look something like this:
```python
## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Either shortcut (machine_file.json) or Project name / allocation / arch
# user_spec = "v100"
# user_spec = ["nvs","v100","v100"]
# slurm_email: str = ""
cp2k_1_walltime_h: list = [0.5, 0.5, 0.5] #float
cp2k_2_walltime_h: list = [1.0, 1.0, 1.5] #float
nb_NODES: list = [1, 1, 1] #int
nb_MPI_per_NODE: list = [32, 32, 64] #int
nb_OPENMP_per_MPI: list = [2, 2, 2] #int
```
where the reactive water calculations use full nodes and have a higher wall time of 1h30min. The wall times should be set for the first iteration but can be guessed automatically later using the average time per CP2K calculation measured in the previous iteration. Note that we did not change the `user_spec` variable even if the default is that of Jean-Zay because the cluster is automatically detected by the code and the corresponding default partition (indicated in `machine_file.json`) will be used. We can now run the first 2 scripts and wait for the electronic structure calculations to finish. When running the check file the script might tell us that there are failed configurations in the `water-reactive` folder! we can see which calculations did not converge in the `water-reactive/2_failed.txt` file. Suppose there were 2 failed jobs, the 13-th and the 54-th. We might just do `touch water-reactive/00013/skip` and `touch water-reactive/00054/skip` and run the `labeling3_check.py` script again. This time it will inform us that some configurations will be skipped, but the final message should be that check phase is a success. All that is left to do now is run the `labeling4_extract.py` script, clean up with the `labeling9_clean.py` script, store wavefunctions and remove all unwanted data and finally update our local folder. We have now augmented our total training set and might do a new training iteration and keep iterating until convergence is reached!

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
