# Iterative procedure prerequisites 

When training a neural network potential (NNP) for a chemical system (or several systems that you want to describe with the same NNP), you will often want to explore the chemical space as diversely as possible.
In ArcaNN, this is made possible by the use of **systems**.
A **system** corresponds to a particular way of exploring the chemical space that interests you and will be represented by specific *datasets* within the total training set of the NNP.
A *dataset* corresponds to an ensemble of structures (*e.g.*, atomic positions, types of atoms, box size, etc.) and corresponding labels (*e.g.*, energy, forces, virial).

You get the idea: you need a subsystem for every kind of chemical composition, physical state (temperature, density, pressure, cell size, etc.), biased reactive pathway, and more that you wish to include in your final training *dataset*.

**Attention**, **systems** are defined once and for all in the [Initialization](../initialization) of the procedure.
Because of this, every time you want to include a new subsystem (such as transition state structures, see [SN2](../examples/sn2.md) example), you will need to initialize the procedure again.
This is very simple—you only need to create a new `$WORK_DIR` and include the necessary files in `user_files/` for each extra **system** you want to add.

To initiate the iterative training procedure, you should create in your `$WORK_DIR` two folders: `user_files/` and `data/`.

In `user_files/` you will store all the files needed for each step. You can find some templates to start with in the [GitHub Repository](https://github.com/arcann-chem/), now available in your machine at your ArcaNN installation location `arcann_training/examples/user_files/`.

- For the exploration step, you must adapt the template files found in `exploration_lammps/` or `exploration_sander_emle/` depending of your choice. You will need the following files:
    - The input files: `SYSTEM.in` for LAMMPS and `SYSTEM.xml` for i-PI.
    - The plumed files: `plumed_SYSTEM.dat` where `SYSNAME` refers to the **system** name (additional PLUMED files can be used as `plumed_*_SYSNAME.dat`, which will also be taken into account for explorations).


- For the labeling step, use the template files found in `labeling_cp2k/` or `labeling_orca/`. You will need the following files:
    - The input files : `[1-2]_SYSNAME_labeling_XXXXX_[cluster].inp`, where `[cluster]` refers to the short string selected for the labeling cluster in the `machine.json`; see [Labeling](../labeling).


- For the training step, from `training_deepmd` you will need: 
    - A DeePMD-kit JSON file named `dptrain_VERSION.json`, where `VERSION` is the DeePMD-kit version that you will use (*e.g.*, `2.1`; currently supported versions are `2.0`, `2.1`, `2.2`, and `3.0`).


- The SLURM scripts for individual jobs and for job arrays are organised by step in several fordels. Prepare your files according to your software choice for each step. You can fin them in: 
    - `job_exploration_lammps_slurm` and `job_exploration_sander_emle_slurm` for exploration.
    - `job_labeling_orca_slurm` and `job_labeling_cp2k_slurm` for the labeling.
    - `job_training_deepmd_slurm` for training. 
    - `job_test_deepmd_slurm` for testing. 


We **strongly** advise you to create the previous files starting from the templates, as they contain replaceable strings for the key parameters that will be updated by the procedure.

- A representative file in the [LAMMPS Data Format](https://docs.lammps.org/2001/data_format.html) format for each **system**, named `SYSTEM.lmp`, where `SYSTEM` refers to the **system** name  (we will refer to them as `LMP` files).
This file represent the configuration of the **system**, with the number of atoms and the number of types of atoms, the simulation cell dimension, the atomic masses for each type and the atomic geometry of your **system**. They will be used as starting point for the first exploration.


The order of the atoms in the `LMP` files **must** be identical for every **system** and **must** match the order indicated in the `"type_map"` keyword of the DeePMD-kit `dptrain_VERSION.json` training file. 

- A `properties` file must be provided and named `properties.txt`.
This is file will be used by [atomsk](https://atomsk.univ-lille.fr/tutorial_properties.php). In the case you have several **systems** with different chemical composition, the properties file must be the same for all systems to ensure consistant type mapping. 

Then, gather all these files and store them inside the `user_files/` directory. **Don't** create subdirectories inside `user_files`. 

Finally, you also need to prepare at least one initial *dataset* which will be used for your first neural networks training, that you will store in the `data/` directory. 
It must follow DeePMD-kit standards and should contain a `type.raw` file and `set.000/` folder with `box.npy`, `coord.npy`, `energy.npy` and `force.npy` (see [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/master/)).
You can prepare as many initial *datasets* as you wish and they should all be stored in the `$WORK_DIR/data/` folder with a folder name starting with `init_`.








