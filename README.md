[<picture><img alt="ArcaNN logo" src="./doc/_static/arcann_logo.svg"></picture>]

---

[![GNU AGPL v3.0 License](https://img.shields.io/github/license/arcann-chem/arcann_training.svg)](https://github.com/arcann-chem/arcann_training/blob/main/LICENSE)
[![Unit Tests Requirements](https://github.com/arcann-chem/arcann/actions/workflows/unittests_requirements.yml/badge.svg)](https://github.com/arcann-chem/arcann/actions/workflows/unittests_requirements.yml)
[![Unit Tests Matrix](https://github.com/arcann-chem/arcann/actions/workflows/unittests_matrix.yml/badge.svg?branch=main)](https://github.com/arcann-chem/arcann/actions/workflows/unittests_matrix.yml)
[![Docs](https://github.com/arcann-chem/arcann/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/arcann-chem/arcann/actions/workflows/docs.yml)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2407.07751-blue)](https://doi.org/10.48550/arXiv.2407.07751)

---

# ArcaNN #

ArcaNN proposes an automated enhanced sampling generation of training sets for chemically reactive machine learning interatomic potentials.
In its current version, it aims to simplify and to automate the iterative training process of a [DeePMD-kit](https://doi.org/10.1063/5.0155600) neural network potential for a user-chosen system, but the core concepts of the training procedure could be extended to other network architectures.
The main advantages of this code are its modularity, the ability to finely tune the training process to adapt to your system and workflow, and great traceability, as the code records every parameter set during the procedure.
During the iterative training process, you will iteratively train neural network potentials, use them as reactive force fields for molecular dynamics simulations (to explore the phase space), select and label some configurations based on a query by committee approach, and then train neural network potentials again with an improved training set, and so forth.
This workflow, sometimes referred to as active or concurrent learning, was heavily inspired by [DP-GEN](https://doi.org/10.1016/j.cpc.2020.107206), and we use their naming scheme for the steps of the iterative procedure.

We refer the reader to the [documentation](https://arcann-chem.github.io/arcann_training/) and the accompanying paper [ArcaNN: Automated Enhanced Sampling Generation of Training Sets for Chemically Reactive Machine Learning Interatomic Potentials](https://doi.org/10.48550/ARXIV.2407.07751).

## Installation ##

To install ArcaNN, please read the [documentation](https://arcann-chem.github.io/arcann_training/getting-started/requirements/).

## License ##

Distributed under the GNU Affero General Public License v3.0. See `LICENSE` for more information.

## How to cite ##

If you use this code, please cite the following publication:

David, R.; de la Puente, M.; Gomez, A.; Anton, O.; Stirnemann, G.; Laage, D. ArcaNN: Automated Enhanced Sampling Generation of Training Sets for Chemically Reactive Machine Learning Interatomic Potentials. arXiv 2024. [https://doi.org/10.48550/ARXIV.2407.07751](https://doi.org/10.48550/ARXIV.2407.07751).

## Fundings & HPC Allocations ##

- Idex ANR-10-IDEX-0001-02PSL
- ERC Grant Agreement No. 757111
- GENCI Grant 2023-A0130707156

## Acknowledgments & Sources ##

- [Stackoverflow](https://stackoverflow.com/)

### Beta-testers ###

- Olaia Anton, Zakarya Benayad, Miguel de la Puente, Axel Gomez
- Oscar Gayraud, Pierre Girard, Anne Milet
- Meritxell Malagarriga Perez, Adrián García
- Ashley Borkowski, Pauf Neupane, Ward Thompson
- Hadi Dinpajooh

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

