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

## About The Project

Very fancy DeepMD-based semi-automatic highly-customizable iterative training procedure.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
<div id="getting-started"></div>

## Getting Started

<div id="prerequisites"></div>

### Prerequisites

<!-- TODO: Prerequisites  -->

* python >= 3.7.3
* numpy >= 1.17.3
* setuptools >= 40.8.0
* atomsk >= beta-0.12.1 <!-- TODO: List of steps  -->
* VMD >= 1.9.4 <!-- TODO: List of steps  -->

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

Default value are located in __default_config.json__ located in __src/arcann/assets/__

Initialization

```json
{
    "step_name": "initialization",
    "system": { "value": null },
    "subsys_nr": { "value": null },
    "nb_nnp": { "value": 3 },
    "exploration_type": { "value": "lammps" }
}
```

Fields:

* __step_name__: The name of the step, which is always "initialization".
* __system__: An object that contains the system information, represented as a string. The value is initially set to null and must be provided by the user.
* __subsys_nr__: An object that contains the subsystem name list, represented as a list of strings. The value is initially set to null and must be provided by the user.
* __nb_nnp__: An object that contains the number of NNP trained, represented as an integer. The default value is __3__ if not provided by the user.
* __exploration_type__: An object that contains the exploration type, represented as a string that can be either "lammps" or "i-PI". The default value is __"lammps"__ if not provided by the user.

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
