# Iterative procedure prerequisites 

When training a neural network potential (NNP) for a chemical system (or several systems that you want to describe with the same NNP), you will often want to explore the chemical space as diversely as possible.
In `ArcaNN`, this is made possible by the use of **systems**.
A **system** corresponds to a particular way of exploring the chemical space that interests you and will be represented by specific *datasets* within the total training set of the NNP.
A *dataset* corresponds to an ensemble of structures (*e.g.*, atomic positions, types of atoms, box size, etc.) and corresponding labels (*e.g.*, energy, forces, virial).

You get the idea: you need a subsystem for every kind of chemical composition, physical state (temperature, density, pressure, cell size, etc.), biased reactive pathway, and more that you wish to include in your final training *dataset*.

**Attention**, **systems** are defined once and for all in the [Initialization](../initialization) of the procedure.
Because of this, every time you want to include a new subsystem (such as transition state structures, see [SN2](../examples/sn2.md) example), you will need to initialize the procedure again.
This is very simple—you only need to create a new `$WORK_DIR` and include the necessary files in `user_files/` for each extra **system** you want to add.

To initiate the iterative training procedure, you should create in `$WORK_DIR` two folders: `user_files/` and `data/`.
In `user_files/` you will store all the files needed for each step (**not** in the `arcann_training/examples/user_files/` folder if you kept it!).
You can start with the examples given in the github `arcann_training/examples/user_files/`:

- The LAMMPS (or i-PI) and CP2K files used for carrying out the exploration and labeling phases of each **system** should also be prepared before initialization and follow the required naming scheme (`SYSNAME.in` for the LAMMPS input file, `SYSNAME.xml` for i-PI, and `[1-2]_SYSNAME_labeling_XXXXX_[cluster].inp` for the two CP2K files required per **system**, where `[cluster]` refers to the short string selected for the labeling cluster in the `machine.json`; see [Labeling](../labeling)).
We **strongly** advise you to create these files starting from those given in the github`arcann_training/examples/user_files/` folder, as they must contain replaceable strings for the key parameters that will be updated by the procedure.

- If a **system** requires the use of PLUMED for the explorations, you will need: a `plumed_SYSNAME.dat` where `SYSNAME` refers to the **system** name (additional PLUMED files can be used as `plumed_*_SYSNAME.dat`, which will also be taken into account for explorations).

- A DeePMD-kit JSON file for training also needs to be prepared and named `dptrain_VERSION.json`, where `VERSION` is the DeePMD-kit version that you will use (*e.g.*, `2.1`; currently supported versions are `2.0`, `2.1`, and `2.2`).

- A representative file in the [LAMMPS Data Format](https://docs.lammps.org/2001/data_format.html) format for each **system**, named `SYSNAME.lmp`, where `SYSNAME` refers to the **system** name, and we will refer them as `LMP` files.
This file represent the configuration of the **system**, with the number of atoms and the number of types of atoms, the simulation cell dimension, the atomic masses for each type and the atomic geometry of your **system**.
Theses will be used as starting point for the first exploration.


**Please note** that the order of the atoms in the `LMP` files **must** be identical for every **system** and **must** match the order indicated in the `"type_map"` keyword of the DeePMD-kit `dptrain_VERSION.json` training file. 

- A `properties` file must be provided and named `properties.txt`.
This is file will be used by `ArcaNN` with [atomsk](https://atomsk.univ-lille.fr/tutorial_properties.php)..
In the case you have several **systems** with different chemical composition (*e.g.*, NaCl in water and NaBr in water), the properties file must be the same for all systems to ensure consistant type mapping. 


- Finally, you also need to prepare at least one initial *dataset* which will be used for your first neural networks training.
This follows DeePMD-kit standards and should contain a `type.raw` file and `set.000/` folder with `box.npy`, `coord.npy`, `energy.npy` and `force.npy` (see [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/master/)).
You can prepare as many initial *datasets* as you wish and they should all be stored in the `$WORK_DIR/data/` folder with a folder name starting with `init_`.
