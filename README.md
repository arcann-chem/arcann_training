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

What is it ?

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
<div id="getting-started"></div>

## Getting Started

<div id="prerequisites"></div>

### Prerequisites

What is needed for your project.

ex: 

Compilation
* gfortran (version with Fortran 2008 support)
* make

Tests
* python3 & numpy

<div id="installation"></div>

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/laagegroup/PROJECT.git
   ```
2. Or get a release and untar the archive (source code)
   ```sh
   tar -xzvf program_name-x.x.tar.gz
   ```
3. Go into the folder and compile the program
   ```sh
   make program_name
   ```
4. Test the program (python3 with numpy)
   ```sh
   cd tests
   ../bin/program_name input_test output_test
   python test.py && echo "Success"
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
<div id="usage"></div>

## Usage

How to use the code ?

   ```sh
   program_name input output
   ```

The input_file a formatted file:

   ```sh
   coord test.dcd
   n_atoms 932
   n_frames 10
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
<div id="license"></div>

## License

Distributed under the GNU Affero General Public License v3.0. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
<div id="acknowledgments"></div>

## Acknowledgments & Sources

*
*
*

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/laagegroup/0_Template.svg?style=for-the-badge
[license-url]: https://github.com/laagegroup/0_Template/blob/main/LICENSE
