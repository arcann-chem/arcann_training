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

## About The Project

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

## Getting Started

<div id="prerequisites"></div>

### Prerequisites

<!-- TODO: Prerequisites  -->

* python >= 3.8 (all steps)
* DeePMD-kit 2.0 <= 2.1 (exploration and training, includes LAMMPS installation)
* CP2K 7.1 <= 8.2 (labeling)
* Slurm >= ? (cluster requirement)
* numpy >= 1.15 (exploration1_prep exploration4_devi exploration5_extract labeling4_extract training1_prep training3_check initialization)
* VMD >= 1.9.4 (exploration5_extract)
* Atomsk >= beta-0.11.2 (exploration5_extract)
* scipy >= ? (test5_plot)

<div id="installation"></div>

### Installation

To use `deepmd_iterative_py` you can clone or download this repository using the `Code` green button on top right corner of the main page of this repository. Keep a local copy of this repository on each computer that will be used to prepare, run or analyze any part of the iterative training process. This repository contains important files that are required at different stages of the process (for example in `tools/`) and should remain available at all time. We recommend to make it available (by either installing it or linking it) as `~/deepmd_iterative_py/` which is the default location used in the scripts but a different location is possible (see [Initialization](#initialization)).

<div id="machine"></div>

### Cluster setup

This repository is designed for use on Jean Zay (mainly) and Irene Rome. For a different computer/supercomputer, only some changes need to be made as long as it runs on `Slurm` (if it does not, good luck...):
- in `tools/common_functions.py`, the function `check_cluster()` (line 217) should be modified to include the identification of your machine and define a short string name for it. In order to do so, you must add an `elif` condition checking for a pattern included in the result that you get when you run `python -c "import socket ; print(socket.gethostbyaddr(socket.gethostname())[0])"` in your cluster and return the short string name that you wish to associate to your cluster.
- in `jobs/`, you will need to create job files following the same model as the existing files with the correct naming scheme (notably the short name of the cluster that you indicated in `check_cluster()` at the end of the file name, before `.extension`). This files should match your cluster requirements and keep the replace parameters with the exact same keys as in the existing files (replace parameters are all parameters named `_R_[PARAM_NAME]_`). You should also modify the paths/modules for DeePMD-kit and CP2K that correspond to you cluster.
- generate a `machine_file.json` for your cluster with the various parameters required to submit a Slurm job. This file will be placed in your iterative training working directory within the `inputs` folder (see [Initialization](#initialization)). To create this `machine_file.json` file you can copy the one installed with the github repo in your working directory (it is not advised to modify it directly in the repo directory).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE INSTRUCTIONS -->
<div id="usage"></div>

## Usage

At this stage, `deepmd_iterative_py` is available on your computer and you made the necessary changes for your computer (see [Cluster setup](#cluster-setup)). You can now start the procedure. Create an empty directory anywhere you like that will be your iterative training working directory. We will refer to this directory by the variable name `$WORK_DIR`. 

<div id="usage-req"></div>

### Iterative procedure prerequisites

When training a neural network potential (NNP) for a chemical system (or several systems that you want to describe with the same NNP) you will often want to explore the chemical space as diversely as possible. In `deepmd_iterative_py` this is made possible by the use of **subsystems**. A subsystem will correspond to a given way of exploring the chemical space that interests you and will be ultimately associated with specific data sets within the total training set of the NNP. **Examples**: when building an NNP to study liquid water and ice you will need to include both liquid and solid configurations into your training set, this can be done by defining two subsystems (`ice` and `liquid` for example), at every iteration you will perform explorations with each subsystem, thus getting (after candidate selection and labelling) corresponding `ice` and `liquid` data sets at every iteration (**Note** a "data set" corresponds to what DeePMD people call a "system"). If you also want to add configurations with self-dissociated molecules you might want to explore by adding biases with the PLUMED software, for which you need to have a corresponding subsystem. You get the idea, you need a subsystem for every kind of chemical composition, physical state (temperature, density, pressure, cell size, etc.), biased reactive pathway, etc. that you wish to include in your final training set. **Attention**, subsystems are defined once and for all in the [Initialization](#initialization) of the procedure. Therefore, before starting you will need to have prepared a representative configuration for each subsystem that you

 prepare the different subsystems you are interested in and want to include in your training sets. They can be of different chemical composition, at different temperatures/box sizes, etc. But you need to prepare a representative configuration of each chosen subsystem. Subsytems can also differ based on MD restraints during exploration phase (*e.g.*, to explore the phase space in different directions).

You also need to prepare at least one initial training dataset which will be used for your neural networks training. This follows DeePMD-kit standards and should contain a `type.raw` file and `set.000/` folder with `box.npy`, `coord.npy`, `energy.npy` and `force.npy` (see [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/master/))

<div id="usage-initialization"></div>

### Initialization

<!-- TODO initialization-->

<div id="usage-training"></div>

### Training

<!-- TODO training-->

<div id="usage-exploration"></div>

### Exploration

<!-- TODO exploration-->

#### LAMMPS classical nuclei simulations

<!-- TODO LAMMPS -->

#### i-PI quantum nuclei simulations

<!-- TODO i-PI -->

<div id="usage-labeling"></div>

### Labeling

<!-- TODO labeling-->

<div id="usage-test"></div>

### Test (optional)

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
* Wang, H.; Zhang, L.; Han, J.; E, W. DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics. Comput. Phys. Commun. 2018, 228, 178–184. [https://doi.org/10.1016/j.cpc.2018.03.016](https://doi.org/10.1016/j.cpc.2018.03.016)


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/laagegroup/0_Template.svg?style=for-the-badge
[license-url]: https://github.com/laagegroup/0_Template/blob/main/LICENSE
