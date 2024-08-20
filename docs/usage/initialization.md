# Initialization #

## Initialization ##

Now that you have decided the subsystems that you want to train your NNP on and prepared all the required files you can initialize the `ArcaNN` procedure by running (from the $WORK_DIR folder):

```bash
python -m arcann_training initialization start 
```

Now it should have generated your first `000-training` directory. In `$WORK_DIR` you will also find a `default_input.json` file that lools like this :

```JSON
{
    "step_name": "initialization",
    "systems_auto": ["SYSNAME1", "SYSNAME2", "SYSNAME3"],
    "nnp_count": 3
}
```

The `"systems_auto"` keyword contains the name of all the subsystems that were found in your `$WORK_DIR/user_files/` (i.e. all files lmp files) directory and `"nnp_count"` is the number of NNP that is used by default in the committee.

The initialization will create several folders. The most important one is the `control/` folder, in which essential data files will be stored throughout the iterative procedure. These files will be written in `.json` format and should NOT be modified. Right after initialization the only file in `control/` is `config.json`, which contains the essential information about your initialization choices (or defaults), such as your subsystem names and options. Finally the `000-training` empty folder should also have been created by the execution of the python script, where you will perform the first iteration of [training](../training).

If at this point you want to modify the datasets used for the first training you simply have to create an `input.json` from the `default_input.json` file and remove or add the system names to the list. You could also change the number of NNP if you wish. Then you only have have to execute the command of the initialization phase again and your `000-training` directory will be updated.

## EXAMPLE ##

Let's use the above example of a NNP for water and ice that is able to describe water self-dissociation. Suppose that you want 3 subsystems (ice, un-dissociated liquid water, water with a dissociated pair) your `defaut_input.json` file might look like this:

```JSON
{
    "step_name": "initialization",
    "systems_auto": ["ice", "water", "water-reactive"],
    "nnp_count": 3
}
```

Before executing this phase, you will have prepared a data set for each subsystem (not compulsory, but recommended), stored in the data directory: `data/init_ice`, `data/init_water` and `data/init_water-reactive`. In the `user_files/` folder you will have the following scripts:

- `dp_train_2.1.json` for the DeePMD-kit trainings (or any other version with the corresponding name)
- `machine.json` file containing the cluster parameters
- `ice.in`, `water.in` and `water-reactive.in` LAMMPS inputs
- `ice.lmp`, `water.lmp` and `water-reactive.lmp` starting configurations
- `plumed_water-reactive.dat` plumed file used for biasing only in the reactive system
- `1_ice_labeling_XXXXX_[cluster].inp`, `2_ice_labeling_XXXXX_[cluster].inp`, `1_water_labeling_XXXXX_[cluster].inp`, `2_water_labeling_XXXXX_[cluster].inp`, `1_water-reactive_labeling_XXXXX_[cluster].inp` and `2_water-reactive_labeling_XXXXX_[cluster].inp` CP2K files (there are 2 input files per subsystem, see details in [labeling](../labeling)), where "[cluster]" is the machine keyword indicated in the `machine.json` file.
- `job_lammps-deepmd_explore_gpu_myHPCkeyword1.sh` and `job-array_lammps-deepmd_explore_gpu_myHPCkeyword1.sh` job scripts for exploration, `job_CP2K_label_cpu_myHPCkeyword1.sh` and `job-array_CP2K_label_cpu_myHPCkeyword1.sh` job scripts for labeling, `job_deepmd_compress_gpu_myHPCkeyword1.sh`, `job_deepmd_freeze_gpu_myHPCkeyword1.sh` and `job_deepmd_train_gpu_myHPCkeyword1.sh` job scripts for training
- `dptrain_2.1.json` input for DeePMD /!\ il s'appelle training_2.1.json dans examples/user_files/training_deepmd
