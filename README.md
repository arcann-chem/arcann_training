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
        <li><a href="#machine">Cluster setup</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#usage-req">Iterative procedure prerequisites</a></li>
        <li><a href="#usage-initialization">Initialization</a></li>
        <li><a href="#usage-training">Training</a></li>
        <li><a href="#usage-exploration">Exploration</a></li>
        <li><a href="#usage-labeling">Labeling</a></li>
        <li><a href="#usage-test">Test (optional)</a></li>
      </ul>
    </li>
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
- `examples/` contains basis input files for the different codes used in the procedure (CP2K, LAMMPS, etc.). This can be used as a starting point when using this semi-automatic procedure (see [Initialization](#initialization) below).
- `jobs/` gathers generic Slurm job files that serve as templates to generate the submission files actually used during the procedure. They have been prepared for French national supercomputers Jean Zay (exploration on GPU, labeling on CPU, test and training on GPU) and Irene (labeling on CPU). The file naming scheme follows a strict convention: As it can be seen in the names of the job files, the type of computer resources (cpu or gpu) as well as a short name for the cluster (jz or ir in this example) should be present at a very specific place.
- `scripts/`: all the scripts you will call during the procedure to perform the different *active learning* steps.
- `tools/`: several codes and files required by the code to work

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
<div id="getting-started"></div>

# Getting Started

<div id="prerequisites"></div>

## Prerequisites

<!-- TODO: Prerequisites  -->

* python >= 3.6 (all steps)
* DeePMD-kit 2.0 <= 2.1 (exploration and training, includes LAMMPS and i-PI installation)
* CP2K 7.1 <= 8.2 (labeling)
* Slurm >= ? (cluster requirement)
* numpy >= 1.15 (exploration1_prep exploration4_devi exploration5_extract labeling4_extract training1_prep training3_check initialization)
* VMD >= 1.9.4 (exploration5_extract)
* Atomsk >= beta-0.11.2 (exploration5_extract)
* scipy >= ? (test5_plot)

<div id="installation"></div>

## Installation

To use `deepmd_iterative_py` you can clone or download this repository using the `Code` green button on top right corner of the main page of this repository. Keep a local copy of this repository on each computer that will be used to prepare, run or analyze any part of the iterative training process. This repository contains important files that are required at different stages of the process (for example in `tools/`) and should remain available at all time. We recommend to make it available (by either installing it or linking it) as `~/deepmd_iterative_py/` which is the default location used in the scripts but a different location is possible (see [Initialization](#initialization)).

<div id="machine"></div>

## Cluster setup

This repository is designed for use on Jean Zay (mainly) and Irene Rome. For a different computer/supercomputer, only some changes need to be made as long as it runs on `Slurm` (if it does not, good luck...):
- in `tools/common_functions.py`, the function `check_cluster()` (line 217) should be modified to include the identification of your machine and define a short string name for it. In order to do so, you must add an `elif` condition checking for a pattern included in the result that you get when you run `python -c "import socket ; print(socket.gethostbyaddr(socket.gethostname())[0])"` in your cluster and return the short string name that you wish to associate to your cluster.
- in `jobs/`, you will need to create job files following the same model as the existing files with the correct naming scheme (notably the short name of the cluster that you indicated in `check_cluster()` at the end of the file name, before `.extension`). This files should match your cluster requirements and keep the replace parameters with the exact same keys as in the existing files (replace parameters are all parameters named `_R_[PARAM_NAME]_`). You should also modify the paths/modules for DeePMD-kit and CP2K that correspond to you cluster.
- generate a `machine_file.json` for your cluster with the various parameters required to submit a Slurm job. This file will be placed in your iterative training working directory within the `inputs` folder (see [Initialization](#initialization)). To create this `machine_file.json` file you can copy the one installed with the github repo in your working directory (it is not advised to modify it directly in the repo directory).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE INSTRUCTIONS -->
<div id="usage"></div>

# Usage

At this stage, `deepmd_iterative_py` is available on your computer and you made the necessary changes for your computer (see [Cluster setup](#cluster-setup)). You can now start the procedure. Create an empty directory anywhere you like that will be your iterative training working directory. We will refer to this directory by the variable name `$WORK_DIR`. 

We will describe the prerequisites and then the intialization, training, exploration and labeling phases. At the end of each phase description we include an example.

<div id="usage-req"></div>

## Iterative procedure prerequisites

When training a neural network potential (NNP) for a chemical system (or several systems that you want to describe with the same NNP) you will often want to explore the chemical space as diversely as possible. In `deepmd_iterative_py` this is made possible by the use of **subsystems**. A subsystem will correspond to a given way of exploring the chemical space that interests you and will be ultimately associated with specific data sets within the total training set of the NNP. **Examples**: when building an NNP to study liquid water and ice you will need to include both liquid and solid configurations into your training set, this can be done by defining two subsystems (`ice` and `liquid` for example), at every iteration you will perform explorations with each subsystem, thus getting (after candidate selection and labelling) corresponding `ice` and `liquid` data sets at every iteration (**Note** a "data set" corresponds to what DeePMD people call a "system"). If you also want to add configurations with self-dissociated molecules you might want to explore by adding biases with the PLUMED software, for which you need to have a corresponding subsystem. You get the idea, you need a subsystem for every kind of chemical composition, physical state (temperature, density, pressure, cell size, etc.), biased reactive pathway, etc. that you wish to include in your final training set. **Attention**, subsystems are defined once and for all in the [Initialization](#initialization) of the procedure. Therefore, before starting you will need to have prepared a representative **configuration** for each subsystem (which you will name `SYSNAME.lmp`, where `SYSNAME` refers to the subsystem name). A configuration of the subsystem contains a given atomic geometry of your subsystem (it will be used as starting point for the first exploration), the number of atoms, the simulation cell dimensions and the atomic masses, all in a LAMMPS compatible format (see, for example, `examples/shared/SYSTEM1.lmp`). If a subsystem will require the use of PLUMED for the explorations you will also need to have prepared all the PLUMED files, that you will name as `plumed_SYSNAME.dat` where `SYSNAME` refers to the subsystem name. The LAMMPS (or i-PI) and CP2K files used for carrying out the exploration and labeling phases of each subsystem should also be prepared before the initialization and follow the required naming scheme (`SYSNAME.in` for the LAMMPS input file, `SYSNAME.xml` for i-PI and `[1-2]_SYSNAME_labeling_XXXXX_[cluster].inp` for the 2 CP2K files required per subsystem, where `[cluster]` refers to the short string selected for the labeling cluster in the `machine_file.json`, see [Labeling](#labeling)). We **strongly** advise you to create this files starting from the ones given in the `examples/` folder, since they must contain replaceable strings for the key parameters that will be updated by the procedure. 
 A DeePMD-kit `.json` file for training needs also to be prepared and named as `VERSION_DESCRIPTOR.json` where `VERSION` is the DeePMD-kit version that you will use (ex: `2.1`) and `DESCRIPTOR` is the [smooth-edition](https://papers.nips.cc/paper_files/paper/2018/hash/e2ad76f2326fbc6b56a45a56c59fafdb-Abstract.html) strategy used for creating the atomic configuration descriptor (ex: `se2_a` for two-body descriptors with radial and angular information).

Finally, you also need to prepare at least one initial training dataset which will be used for your neural networks training. This follows DeePMD-kit standards and should contain a `type.raw` file and `set.000/` folder with `box.npy`, `coord.npy`, `energy.npy` and `force.npy` (see [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/master/)) You can prepare as many initial sets as you wish.

<div id="usage-initialization"></div>

## Initialization

<!-- TODO initialization-->

Now that you have decided the subsystems that you want to train your NNP on and prepared all the files and DeePMD systems required you can initialize the `deepmd_iterative_py` procedure. For this go to `$WORK_DIR` and create to folders `inputs/` and `data/`. Copy your initial data sets (folders containing `type.raw` and `set.000/` folder) to the `data/` folder and add the prefix `init_` to their names. Copy all the input files and configurations that you prepared to the `inputs/` folder (respect the naming convention!). All you need to do now is copy the initialization script to your working directory (we assume that you installed or linked the github repo in your home directory):
```
cp ~/deepmd_iterative_py/scripts/initialization.py $WORK_DIR/.
```
Open the `initialization.py` script and define:
- `sys_name`, a string containing the name of the NNP that you want to create (it is a general purpose name describing the NNP, its choice is unimportant)
- `subsys_name`, a list of strings containing the names of all your subsystems. The number and name of the subsystems indicated here must match **exactly** those indicated in your input and configuration files! **Note**: try to avoid the use of underscores (`_`) since these are used by the code to search for patterns.
- `nb_nnp`, an integer (`>=2`) indicating the number of neural networks trained at each iteration. The deviation between these networks predictions are used to select candidates and each network will be used to propagate trajectories in the exploration phase. **DEFAULT** is 3.
- `exploration_type`, a string defining the software used to perform explorations. Only `"lammps"` and `"i-PI"` are allowed. **DEFAULT** is `"lammps"`.
- `temperature_K`, a list of floats containig the temperature at which the exploration of each subsystem will be performed. **DEFAULT** is 298.15 K for all subsystems.
- `timestep_ps`, a list of floats with the MD time-step used for the exploration in each subsystem. **DEFAULT** is 0.5 fs for all subsystems in LAMMPS and 0.25 fs in i-PI.
- `nb_candidates_max`, an integer defining the maximum number of candidate configurations that can be extracted per subsystem at a given exploration phase. **DEFAULT** is 500.
- `s_low`, `s_high` and `s_high_max` floats, defining the deviations threshold (lower and upper) for candidate selection. `s_high_max` defines an extra threshold, if deviations ever reach above it the rest of the exploration trajectory is discarded (this is used to prevent non-physical recombinations). **DEFAULTS** are 0.1, 0.8 and 1.0 (in eV per angstrom)
- `ignore_first_x_ps` float, defines the length of trajectory that should be discarded as equilibration because it is too close to the initial configuration. **DEFAULT** is 0.5 ps.

All the keywords with default values can be changed at each iteration in the corresponding scripts (see below).

You can thus initialize the procedure by running (you need to have python3 and all the required packages loaded):
```
python initialize.py
```
which will create several folders. The most important one is the `control/` folder, in which essential data files will be stored throughout the iterative procedure. These files will be written in `.json` format and should NOT be modified (thus, to read them we recommend to download the [jq](https://stedolan.github.io/jq/) program and run `cat FILE.json | /PATH/TO/JQ/jq`). Right after initialization the files that are created are in `control/`:
- `control.json` which contains the essential information about your initialization choices (or defaults), such as your subsystem names and options.
- `path` a text file containing the path to the installation of `deepmd_iterative_py`. This is the file that you should modify if you installed the repo in a different folder than your home (and you should change it accordingly in every machine that you will use!)

Finally `000-training` and `000-test` empty folders should also have been created, where you will perform the first iteration of [training](#training) (and eventually testing, although this is seldom useful at the beginning of the procedure)

### EXAMPLE

Let's use the above example of an NNP for water and ice that is able to describe water self-dissociation. We will use classical nuclei MD and perform the labeling in Irene and the training and explorations in Jean-Zay while keeping the original repo always updated in our local machine. Suppose that you want 3 subsystems (ice, undissociated liquid water, water with a dissociated pair), the header of your `initialization.py` file might look like this:

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

<div id="usage-training"></div>

## Training

<!-- TODO training-->

<div id="usage-exploration"></div>

During the training procedure you will use DeePMD-kit to train neural networks on the data sets that you have thus far generated (or on the initial ones only for the 000 iteration). In order to do this go to the current iteration training folder `XXX-training`. Copy all training scripts to this folder:
```
cp ~/deepmd_iterative_py/scripts/training*.py $WORK_DIR/XXX-training/.
```
There are 9 such scripts that you must now execute in order after having eventually modified their headers to define the pertinent parametes (in case you want something different from the defaults defined during the initialization). The script that you should check the most carefully is the first one `training1_prep.py`, as this sets all the important parameters for the training. Some scripts will submit `Slurm` jobs (model training, freezing and compressing). You must wait for the jobs to finish before executing the next script (generally this will be a check script that will tell you that jobs have failed or are currently running). Once you have executed the first 8 scripts the training iteration is done! Using the 9-th script is optional, as this will only remove intermediary files.

### Example

Suppose that you just ran the `initialization.py` file described in the previous example. You must now perform the first training phase in Jean-Zay. Update (or copy for the first time) the full `$WORK_DIR` from your local machine to Jean-Zay (where you must have also ):
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
here we changed the GPU partition to be used (default is `"v100`, indicated in the `machine_file.json`) and slightly decreased the `initial_seconds_per_1000steps` variable, used to set the `Slurm` wall time. We can then execute all scripts in order (waiting for slurm jobs to finish!). That's it! Now you just need to update the local folder:
```
rsync -rvu USER@jean-zay.idris.fr:/PATH/TO/JZ/WORK_DIR $WORK_DIR/.
```
and you are ready to move on to the exploration phase!

**Notes:**
- At some point during the iterative procedure we might want to get rid of our initial data sets, we would only need to set the `use_initial_datasets` variable to `False`
- We might also have generated some data independently from the iterative procedure that we might want to start using, this can be done by copying the corresponding DeePMD-kit systems to `data/`, prefixing their names by `extra_` and setting the `use_extra_datasets` variable to `True`
- At the end of the phase the last script `training8_update_iter.py` will create the folders needed for the next iteration, save the current NNPs (stored as graph files `graph_?_XXX[_compressed].pb`) into the `$WORK_DIR/NNP` folder and write a `control/training_XXX.json` file with all parameters used during training.

## Exploration

<!-- TODO exploration-->

In the exploration phase we will generate new configurations (referred to as **candidates**) to include in the training set. For this we will perform MD simulations with either the LAMMPS (classical nuclei) or i-PI (quantum nuclei) softwares. Go to the current iteration exploration folder `XXX-exploration` created at the end of the previous training phase. Copy all exploration scripts to this folder and have a look at the `exploration1_prep.py` to check the exploration parameters. The phase is slightly different if you use LAMMPS or i-PI, both phase workflows are detailed below.

### LAMMPS classical nuclei simulations

Once you are satisfied with your exploration parameters (see example below) you can execute the exploration files 1 to 3 (once the `Slurm` MD jobs are done!) to run MD trajectories with each subsystem. In `exploration4_devi.py` you can set important parameters for candidate selection, once you are satisfied you may execute it. In `exploration5_extract.py` an important choice can be made: whether to include "disturbed" candidates in the training set or not. This is done by changing the `disturbed_min_value` and `disturbed_candidates_value` variables from the defaults (0.0) and will include a set of candidates generated by applying a random perturbation to those obtained in the MD trajectories (this will multiply by 2 the number of selected candidates, make sure that the `disturbed_min_value` that you choose will still give physically meaniningful configurations, otherwise you will deteriorate your NNP!). Once you execute this script a `candidates_SUBSYS_XXX.xyz` file will be created in each subsystem directory containing the candidate configurations that will be added to the training set (you might want to check that they make sense!). The `exploration9_clean.py` script can be executed to clean up all the temporary files. A `control/exploration_XXX.json` file will be written recording all the exploration parameters.

### EXAMPLE

After the first training phase of your ice&water NNP you now have starting models that can be used to propagate reactive MD. For this go to the `$WORK_DIR/001-exploration` folder 


### i-PI quantum nuclei simulations

<!-- TODO i-PI -->

<div id="usage-labeling"></div>

## Labeling

<!-- TODO labeling-->

<div id="usage-test"></div>

## Test (optional)

<!-- TODO test-->

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
<div id="license"></div>

# License

Distributed under the GNU Affero General Public License v3.0. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
<div id="acknowledgments"></div>

# Acknowledgments & Sources

* [Stackoverflow](https://stackoverflow.com/)
* Hirel, P. Atomsk: A Tool for Manipulating and Converting Atomic Data Files. Comput. Phys. Commun. 2015, 197, 212–219. [https://doi.org/10.1016/j.cpc.2015.07.012](https://doi.org/10.1016/j.cpc.2015.07.012).
* Humphrey, W.; Dalke, A.; Schulten, K. VMD: Visual Molecular Dynamics. J. Mol. Graph. 1996, 14 (1), 33–38. [https://doi.org/10.1016/0263-7855(96)00018-5](https://doi.org/10.1016/0263-7855(96)00018-5).
* Wang, H.; Zhang, L.; Han, J.; E, W. DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics. Comput. Phys. Commun. 2018, 228, 178–184. [https://doi.org/10.1016/j.cpc.2018.03.016](https://doi.org/10.1016/j.cpc.2018.03.016)


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/laagegroup/0_Template.svg?style=for-the-badge
[license-url]: https://github.com/laagegroup/0_Template/blob/main/LICENSE
