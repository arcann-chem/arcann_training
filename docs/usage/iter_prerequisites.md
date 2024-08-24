# Iterative procedure prerequisites #

In `$WORK_DIR`, you should create two folders: `user_files/` and `data/`.

When training a neural network potential (NNP) for a chemical system (or several systems that you want to describe with the same NNP), you will often want to explore the chemical space as diversely as possible.
In `ArcaNN`, this is made possible by the use of **systems**.
A **system** corresponds to a particular way of exploring the chemical space that interests you and will be represented by specific *datasets* within the total training set of the NNP.
A *dataset* corresponds to an ensemble of structures (*e.g.*, atomic positions, types of atoms, box size, etc.) and corresponding labels (*e.g.*, energy, forces, virial).

**Examples**:

- When building an NNP to study liquid water and ice, you will need to include both liquid and solid configurations in your training set.
This can be done by defining two **systems** (*e.g.*, `ice` and `liquid`).
At every iteration, you will perform explorations with each **system**, thus obtaining (after candidate selection and labeling) corresponding `ice` and `liquid` datasets at every iteration.
If you also want to include configurations with self-dissociated molecules, you might explore adding biases using the PLUMED software, for which you need to have a corresponding **system**.

- If you want train an NNP to study a reaction, such as an SN2 reaction, you would like to include in the training set configurations representing the reactant, product and transition states.
In this case, we would start by defining two **systems** (`reactant` and `product`) and generating structures in both bassins by performing several iterations (exploring the chemical space, labeling the generated structures and training the NNP on the extented *dataset*).
Next, you would also want to generate transition structures between the `reactant` and the `product`.
For that, you would need to performed biased explorations with the PLUMED software (see [Exploration](../exploration)) within different **systems**.

You get the idea: you need a subsystem for every kind of chemical composition, physical state (temperature, density, pressure, cell size, etc.), biased reactive pathway, and more that you wish to include in your final training *dataset*.

**Attention**, **systems** are defined once and for all in the [Initialization](../initialization) of the procedure.
Because of this, every time you want to include a new subsystem (such as self-dissociated structures in the first example or transition state structures in the SN2 example), you will need to initialize the procedure again.
This is very simple—you only need to create a new `$WORK_DIR` and include the necessary files in `user_files/` for each extra **system** you want to add.

To initiate the iterative training procedure, you will need to prepare several files.
You must create a `$WORK_DIR/user_files/` folder and store all the files there (**not** in the `arcann_training/examples/user_files/` folder if you kept it!).
You can start with the examples given in `arcann_training/examples/user_files/`:

- The LAMMPS (or i-PI) and CP2K files used for carrying out the exploration and labeling phases of each **system** should also be prepared before initialization and follow the required naming scheme (`SYSNAME.in` for the LAMMPS input file, `SYSNAME.xml` for i-PI, and `[1-2]_SYSNAME_labeling_XXXXX_[cluster].inp` for the two CP2K files required per **system**, where `[cluster]` refers to the short string selected for the labeling cluster in the `machine.json`; see [Labeling](../labeling)).
We **strongly** advise you to create these files starting from those given in the `arcann_training/examples/user_files/` folder, as they must contain replaceable strings for the key parameters that will be updated by the procedure.

- If a **system** requires the use of PLUMED for the explorations, you will need: a `plumed_SYSNAME.dat` where `SYSNAME` refers to the **system** name (additional PLUMED files can be used as `plumed_*_SYSNAME.dat`, which will also be taken into account for explorations).

- A DeePMD-kit JSON file for training also needs to be prepared and named `dptrain_VERSION.json`, where `VERSION` is the DeePMD-kit version that you will use (*e.g.*, `2.1`; currently supported versions are `2.0`, `2.1`, and `2.2`).

- A representative file in the [LAMMPS Data Format](https://docs.lammps.org/2001/data_format.html) format for each **system**, named `SYSNAME.lmp`, where `SYSNAME` refers to the **system** name, and we will refer them as `LMP` files.
This file represent the configuration of the **system**, with the number of atoms and the number of types of atoms, the simulation cell dimension, the atomic masses for each type and the atomic geometry of your **system**.
Theses will be used as starting point for the first exploration.
You can see the expected format for a water molecule below.

```Data
 # WATER

           3  atoms
           2  atom types

      0.000000000000       2.610000000000  xlo xhi
      0.000000000000       2.000000000000  ylo yhi
      0.000000000000       2.000000000000  zlo zhi

Masses

            1   15.99900000             # O
            2   1.00800000              # H

Atoms

         1    1          0.000000000000       0.000000000000       0.000000000000
         2    2          0.979000000000       0.000000000000       0.000000000000
         3    2         -0.326000000000       0.923000000000       0.000000000000
```

**Please note** that the order of the atoms in the `LMP` files **must** be identical for every **system** and **must** match the order indicated in the `"type_map"` keyword of the DeePMD-kit `dptrain_VERSION.json` training file.
Regarding the order of atoms, if we take the water `LMP` exemple: an oxygen atom will be type 1 (in the `LMP` file) and type 0 with the `"type_map": ["O", "H"]'`, in DeePMD-kit.
If we want to add NaCl in another **system**, the Na and Cl types can't be type 1 or 2.
If you are preparing these files with [atomsk](https://atomsk.univ-lille.fr/) from `.xyz` files, you can set the correct simulation cell and atom ordering by providing them in a [properties file](https://atomsk.univ-lille.fr/tutorial_properties.php).

- A `properties` file must be provided and named `properties.txt`.
This is file will be used by `ArcaNN` with `atomsk`.
In the case of your **system** not having the same chemical composition (*e.g.*, NaCl in water and NaBr in water), the propertie file must account for all types to ensure consistant type mapping.

Here an example:

```Data
type
O 1
H 2
Cl 3
Br 4
Na 5

masses
O 15.99900000
H 1.00800000
Cl 35.45000000
Br 79.90400000
Na 22.98900000
```

And your corresponding `LMP` files will have `type 1 2 3 5` for NaCl/H2O, `type 1 2 4 5` for NaBr/H2O.
In the dptrain JSON: `"type_map"` should be `["O", "H", "Cl", "Br", "Na"]`.

- Finally, you also need to prepare at least one initial *dataset* which will be used for your first neural networks training.
This follows DeePMD-kit standards and should contain a `type.raw` file and `set.000/` folder with `box.npy`, `coord.npy`, `energy.npy` and `force.npy` (see [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/master/)).
You can prepare as many initial *datasets* as you wish and they should all be stored in the `$WORK_DIR/data/` folder with a folder name starting with `init_`.
