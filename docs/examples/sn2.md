# SN2

Here we introduce the basic usage of the ArcaNN software, ilustrated by a SN2 reaction. All the files can be found in the [GitHub Repository](https://github.com/arcann-chem/arcann_training/); and after ArcaNN installation, you will find them at `examples/sn2_ch3cl_br/` inside your local `arcann_traininig` directory. 

For this reaction, two iterative trainings were performed : a first non-reactive training on reactant and products structures, followed by a reactive training where transition structures were generated. 

The non-reactive ArcaNN training in detailed in the following sections. 

## User files
We will start by creating a `user_files/` directory (See [Iterative procedure prerequisites](../usage/iter_prerequisites.md)) where we will include the necessary files for each step of the procedure. You also need to create a `data/` directory where the initial labeled datasets will be stored. For the reactive training, you will store the datasets of the non-reactive training in the corresponding `data/` directory, together with the initial datasets. 

For the non-reactive training, 6 systems were defined : 3 systems to explore the reactant basin (`ch3cl_br_close_300K`, `ch3cl_br_free_300K`, `ch3cl_br_smd_300K`) and 3 systems to explore the product basin (`ch3br_cl_close_300K`, `ch3br_cl_free_300K`, `ch3br_cl_smd_300K`). 

In the `user_files/` folder you will find the following files for each one of the systems (for clarity purposes, we only indicate the files of the `ch3cl_br_close_300K` system here). Note also that `hpc1`` and `hpc2` are the machine keywords indicated in the machine.json file, see [HPC Configuration](../getting-started/hpc_configuration.md). 

**JSON FILES**

- `machine.json` : file containing the cluster parameters.
- `dp_train_2.1.json` : input for DeePMD trainings. 


**JOB FILES**

- `job_lammps-deepmd_explore_gpu_hpc1.sh` and `job-array_lammps-deepmd_explore_gpu_hpc1.sh` : job scripts for exploration
- `job_CP2K_label_cpu_hpc1.sh` and `job-array_CP2K_label_hpc1.sh`: job scripts for labeling
- `job_deepmd_compress_gpu_hpc1.sh`, `job_deepmd_freeze_gpu_hpc1.sh` and `job_deepmd_train_gpu_hpc1.sh` job scripts for training


**CP2K FILES**

- `1_ch3cl_br_close_300K_labeling_XXXXX_hpc1.inp`,  `2_ch3cl_br_close_300K_labeling_XXXXX_hpc1.inp`, `1_ch3cl_br_close_300K_labeling_XXXXX_hpc1.inp`, `2_ch3cl_br_close_300K_labeling_XXXXX_hpc2.inp` : inputs for CP2K labeling. There are 2 input files per subsystem, see details in [labeling](../labeling). 


**LAMMPS FILES**

- `ch3cl_br_close_300K.lmp` : starting configurations for the first exploration in the LAMMPS format. 
- `ch3cl_br_close_300K.in` : inputs for LAMMPS exploration. 

- `plumed_SYSTEM_300K.dat` : plumed input files for the emplorations. 

Additional plumed files can be used, and must be named as `plumed_KEYWORD_SYSTEM.dat`. Here, we used an additional plumed file to store colvars and another to define the key atoms : `plumed_colvars_ch3cl_br_close_300K.dat` and `plumed_atomdef_ch3cl_br_close_300K.dat`. 


The atom order is defined in the `properties.txt` file. It makes sure that the order of the  atoms in the `SYSTEM.lmp` files match the order indicated in the `"type_map"` keyword of the DeePMD-kit `dptrain_2.1.json` training file. Also, it makes sure that the generated structures also presents the correct atom numbering to avoid conflicts. 




## Initialization

After the initialization step, a `default_input.json` file is generated, containing the name of the `LMP` systems found in the `user_files/`, and the default number of NNP for training defined in ArcaNN. 

```JSON
{
    "systems_auto": ["ch3br_cl_close_300K", "ch3br_cl_free_300K", "ch3br_cl_smd_300K", "ch3cl_br_close_300K", "ch3cl_br_free_300K", "ch3cl_br_smd_300K"],
    "nnp_count": 3
}
```

## Training

After running the `prepare` phase, a `default_input.json` file is created. In order to modify some of the default parameters, an `input.json` file must be created in the same directory, where only the parameters to be updated need to be indicated: 

```JSON
{
  "user_machine_keyword_train": "v100_myproject1",  
  "job_walltime_train_h": 12.0
}

```

Then, the input is updated and stored in the directory as `used_input.json`:


```JSON
{
    "user_machine_keyword_train": "v100_myproject1",
    "user_machine_keyword_freeze": "v100_myproject1",
    "user_machine_keyword_compress": "v100_myproject1",
    "job_email": "",
    "use_initial_datasets": true,
    "use_extra_datasets": false,
    "deepmd_model_version": 2.1,
    "job_walltime_train_h": 12.0,
    "mean_s_per_step": 0.108,
    "start_lr": 0.001,
    "stop_lr": 1e-06,
    "decay_rate": 0.9172759353897796,
    "decay_steps": 5000,
    "decay_steps_fixed": false,
    "numb_steps": 400000,
    "numb_test": 0
}
```

## Exploration

After the first training phase NNP you now have starting models that can be used to propagate reactive MD. For this go to the `$WORK_DIR/001-exploration` folder (in your HPC machine!) and execute the `prepare` phase to obtain an `default_input.json` file with default values. For the first iteration we might be satisfied with the defaults (2 simulations per NNP and per subsystem, 10 ps simulations with the LAMMPS time-step of 0.5 fs, etc.) so we might directly run exploration phases 2 and 3 right away (waiting for the `Slurm` jobs to finish as always). These will have created 6 directories (one per system), in which there will be 3 subdirectories (one per trained NNP) `1/`, `2/` and `3/`, in which again there will be 2 subdirectories (default) `0001/` and `0002/`. This means that a total of 32 MD trajectories will be performed for this first iteration (180 ps total simulation time). Be careful, the total exploration time can quickly become huge, especially if you have many systems.


We can finally clean up the working folder by running the `clean` phase and move on to the labeling phase! (Don't forget to keep your local folder updated so that you can analyze all these results)


We allow for slightly larger deviations (`"sigma_high"` keyword set to 0.8 eV/Ang) and collect a larger number of candidates (`"max_candidates"` set to 100) for the more complex third system (reactive water).

At this stage we should decide wether we want to include disturbed candidates in the training set. Here we might want to do so only for the ice system, since explorations at lower temperature explore a more reduced zone of the phase space and it is easier to be trapped in meta-stable states. This can be done by setting `disturbed_start_value` to `0.5`. The values in `disturbed_start_value` are used to disturb the starting structures for the next iteration. For the 2 other systems `disturbed_start_value` and `disturbed_candidate_value` are set to `0.0` in order to avoid disturbance. A non-zero value sets the maximal amplitude of the random translation vector that will be applied to each atom (a different vector for each atom) in Å.


## Labeling

## EXAMPLE ##

After the first exploration phase we recovered 47, 50 and 92 candidates for our `ice`, `water` and `water-reactive` systems for which we must now compute the electronic structure at our chosen reference level of theory (for example revPBE0-D3). We will have prepared (during initialization) 2 `CP2K` scripts for each system, a first quick calculation at a lower level of theory (for example PBE) and then that at our reference level. We will first copy all this data to the HPC machine were we will perform the labeling (where we must have another copy of this repo as well, with a python environment in which the module was installed):

```bash
rsync -rvu $WORK_DIR USER@OTHER_HPC_MACHINE:PATH_TO_WORKDIR
```

 If we are using a larger number of atoms for the reactive system to ensure proper solvation and separation of the ion pair we might need to use more resources for those calculations. In this example we are using 128 CPU nodes of a `"mykeyword1"` partition and the `input.json` file might look something like this:

```JSON
{
    "step_name": "labeling",
    "user_machine_keyword_label": "mykeyword1",
    "job_email": "",
    "walltime_first_job_h": [0.5, 0.5, 0.5],
    "walltime_second_job_h": [1.0, 1.0, 1.5],
    "nb_nodes": [1, 1, 1],
    "nb_mpi_per_node": [32, 32, 64],
    "nb_threads_per_mpi": [2, 2, 2],
}
```

Here the reactive water calculations use full nodes and have a higher wall time of 1h30min. The wall times should be set for the first iteration but can be guessed automatically later using the average time per CP2K calculation measured in the previous iteration. We can now run the first 2 phases and wait for the electronic structure calculations to finish. When running the check phase there could be a message telling us that there are failed configurations in the `water-reactive` folder! We can see which calculations did not converge in the `water-reactive/water-reactive_step2_not_converged.txt` file. Suppose there were 2 failed jobs, the 13-th and the 54-th. We might just do `touch water-reactive/00013/skip` and `touch water-reactive/00054/skip` and run the `check` phase again. This time it will inform us that some configurations will be skipped, but the final message should be that check phase is a success. All that is left to do now is run the `extract` phase, clean up with the `clean` phase, store wavefunctions and remove all unwanted data and finally update our local folder. We have now augmented our total training set and might do a new training iteration and keep iterating until convergence is reached!




## Test




### INPUTS
et quelle partie est utilisée dans quelle phase
### OUTPUTS
Decrire les control/*.json