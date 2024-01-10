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

### Installation

<!-- TODO: Installation  -->
```bash
pip install -e .
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
<div id="usage"></div>

## Usage

<!-- TODO: Usage  -->

<p align="right">(<a href="#top">back to top</a>)</p>

### Inputs

Default value are located in __default_config.json__ located in __deepmd_iterative/assets/__

Initialization

```json
{
    "systems_auto": ["SYSTEM1", "SYSTEM2"],
    "nnp_count": 3,
}
```

* __systems_auto__: This key is associated with a list of system names provided by the user. The list contains two systems, "SYSTEM1" and "SYSTEM2" in this example. The value for this key is initially set to null, and the user must provide its value as a list of strings.
* __nnp_count__: This key represents the number of NNP trained, specifically for the Query by Committee. It is an integer value and the default value is __3__.

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
