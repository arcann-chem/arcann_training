# SN2

Here we introduce the basic usage of the ArcaNN software, ilustrated by a SN2 reaction. All the files are available in the [GitHub Repository](https://github.com/arcann-chem/arcann_training/); and after ArcaNN installation, you will find them at `examples/sn2_ch3cl_br/` inside your local `arcann_traininig` directory. 

The iterative training and dataset generation for the SN2 reaction, comprised two iterative trainings : a first non-reactive training was performed on reactant and products structures, followed by a reactive training where transition structures were generated. 

The files set up for the non-reactive SN2 ArcaNN training is illustrated bellow. Then, the ArcaNN inputs for each step of the first iteration and the corresponding control `json` files are detailed. 

## User files
We will start by creating a `user_files/` directory (See [Iterative procedure prerequisites](../usage/iter_prerequisites.md)) where we will include the necessary files for each step of the procedure. You also need to create a `data/` directory where the initial labeled datasets will be stored. For the reactive training, you will store the datasets of the non-reactive training in the corresponding `data/` directory, together with the initial datasets. 

For the non-reactive training, 6 systems were defined : 3 systems to explore the reactant basin (`ch3cl_br_close_300K`, `ch3cl_br_free_300K`, `ch3cl_br_smd_300K`) and 3 systems to explore the product basin (`ch3br_cl_close_300K`, `ch3br_cl_free_300K`, `ch3br_cl_smd_300K`). 

In the `user_files/` folder you will find the following files for each one of the systems (for clarity purposes, we only indicate the files of the `ch3cl_br_close_300K` system here). Note also that `hpc1` and `hpc2` are the machine keywords indicated in the machine.json file, see [HPC Configuration](../getting-started/hpc_configuration.md). 

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

You can now move to the `000-training` directory corresponding to the training of the first generation of NNP. After running the `prepare` phase, a `default_input.json` file is created. In order to modify some of the default parameters, an `input.json` file must be created in the same directory, where only the parameters to be updated need to be indicated as the following: 

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

The corresponding `control` file in your local `$WORKDIR/control/` is updated after the execution of each `phase`. Once the `000-training` step is finished, you will find the following `training_000.json` file: 

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
    "mean_s_per_step": 0.039030916666666665,
    "start_lr": 0.001,
    "stop_lr": 1e-06,
    "decay_rate": 0.9172759353897796,
    "decay_steps": 5000,
    "decay_steps_fixed": false,
    "numb_steps": 400000,
    "numb_test": 0,
    "training_datasets": ["init_ch3br_cl_xxxxx_1001_4001_60", "init_ch3cl_br_xxxxx_1001_4001_60"],
    "trained_count": 1000,
    "initial_count": 1000,
    "added_auto_count": 0,
    "added_adhoc_count": 0,
    "added_auto_iter_count": 0,
    "added_adhoc_iter_count": 0,
    "extra_count": 0,
    "is_prepared": true,
    "is_launched": true,
    "is_checked": true,
    "is_freeze_launched": true,
    "is_frozen": true,
    "is_compress_launched": true,
    "is_compressed": true,
    "is_incremented": true,
    "min_nbor_dist": 0.9898124626241066,
    "max_nbor_size": [30, 45, 1, 1, 17],
    "median_s_per_step": 0.038560000000000004,
    "stdeviation_s_per_step": 0.0011691332942493009
}
```
When a `phase` is executed succesfully, the corresponding `"is_prepared"`, `"is_launched"`, `"is_checked"`, etc. keywords are set to `true`
Additional performance data, such as the mean time (`"mean_s_per_step"`), median time (`"median_s_per_step"`) and standard deviation (`"stdeviation_s_per_step"`) per training step are reported in this file. 



## Exploration

After the first training phase you now have starting NNP that can be used to propagate reactive MD. After executing the `prepare` phase in the `0001-exploration/` folder, you will obtain an `default_input.json` file with default values. 

We allow for the first exploration for slightly larger deviations by setting `"sigma_low"` keyword set to 0.15 eV/Ang. This is done by modifying the `input.json` and running `prepare` again. 

```JSON
{
"sigma_low": 0.15
}
```

The `used_input.json` becomes then:
```JSON
{
    "user_machine_keyword_exp": "v100_myproject1",
    "job_email": "",
    "atomsk_path": "/programs/apps/atomsk/0.13.1/atomsk",
    "vmd_path": "/prod/vmd/1.9.4a43/bin/vmd_LINUXAMD64",
    "exploration_type": ["lammps", "lammps", "lammps", "lammps", "lammps", "lammps"],
    "traj_count": [2, 2, 2, 2, 2, 2],
    "temperature_K": [300.0, 300.0, 300.0, 300.0, 300.0, 300.0],
    "timestep_ps": [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005],
    "previous_start": [true, true, true, true, true, true],
    "disturbed_start": [false, false, false, false, false, false],
    "print_interval_mult": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "job_walltime_h": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "exp_time_ps": [10.0, 10.0, 41.0, 10.0, 10.0, 41.0],
    "max_exp_time_ps": [400, 400, 400, 400, 400, 400],
    "max_candidates": [50, 50, 50, 50, 50, 50],
    "sigma_low": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    "sigma_high": [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    "sigma_high_limit": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "ignore_first_x_ps": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "disturbed_start_value": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "disturbed_start_indexes": [[], [], [], [], [], []],
    "disturbed_candidate_value": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "disturbed_candidate_indexes": [[], [], [], [], [], []]
}
```

For the first iteration the default parameters are a good starting point. The `"traj_count"` keyword sets to 2  the number of simulations per NNP. and per system and `"timestep_ps"` sets to 0.0005 ps the timestep of the simulations. The `"disturbed_candidate_value"` keywords are all set to 0, so no disturbance is applied to the candidate structures that will be added to the training set. 

To perform the explorations, one directory per system is created, in which there will be 3 subdirectories (one per trained NNP) `1/`, `2/` and `3/`, in which again there will be 2 subdirectories (by default) `0001/` and `0002/`. This means that a total of 36 MD trajectories will be performed for this first iteration. Be careful, the total exploration time can quickly become huge, especially if you have many systems.

If we have a look at the `exploration_001.json` file inside the `$WORKDIR/control/` folder:

```JSON
{
    "atomsk_path": "/programs/apps/atomsk/0.13.1/atomsk",
    "user_machine_keyword_exp": "v100_myproject1",
    "deepmd_model_version": 2.1,
    "nnp_count": 3,
    "systems_auto": {
        "ch3br_cl_close_300K": {
        // exploration parameters from used_input.json 
        },
        "ch3br_cl_free_300K": {
        //
        },
        "ch3br_cl_smd_300K": {
        //
        },
        "ch3cl_br_close_300K": {
        //
        },
        "ch3cl_br_free_300K": {
        //
        },
        "ch3cl_br_smd_300K": {
        //
        }
    },
    "is_locked": true,
    "is_launched": true,
    "is_checked": true,
    "is_deviated": true,
    "is_extracted": true,
    "nb_sim": 36,
    "vmd_path": "/prod/vmd/1.9.4a43/bin/vmd_LINUXAMD64"
}
```

The total number of MD simulations is indicated by the `"nb_sim"` keyword. The `"vmd_path"` and the `"atomsk_path"` correspond to the ones indicated in the `used_input.json`, but are not necessary if the code is already available in the ArcaNN path. When the `exploration` step is succesfully finished, all the `phase` keywords are set to `"true"`.




## Labeling

For the last `step` of the first iteration, we move to the `$WORKDIR/001-labeling/` folder to run the different `phases`. You should adapt the Slurm parameters for the electronic structure calculation to match the architecture of your system. In this case, the number of MPI processes per node is set to 16 with the `"nb_mpi_per_node"` keyword in the `input.json`:

```JSON
{
    "user_machine_keyword_label": "mykeyword1",
    "nb_mpi_per_node": 16
}
```

As usual, the `used_input.json` file will be updated consequently when re running the `prepare` phase: 

```JSON
{
    "user_machine_keyword_label": "mykeyword1",
    "job_email": "",
    "labeling_program": "cp2k",
    "walltime_first_job_h": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "walltime_second_job_h": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "nb_nodes": [1, 1, 1, 1, 1, 1],
    "nb_mpi_per_node": [16, 16, 16, 16, 16, 16],
    "nb_threads_per_mpi": [1, 1, 1, 1, 1, 1]
}
```
The number of MPI processes hs been set to 16 for the 6 systems. The walltimes of both calculations (2 calculation are performed when using CP2, a first quick calculation at a lower level of theory and then the reference level) are kept at the default values. 

Here the reactive water calculations use full nodes and have a higher wall time of 1h30min. The wall times should be set for the first iteration but can be guessed automatically later using the average time per CP2K calculation measured in the previous iteration. We can now run the first 2 phases and wait for the electronic structure calculations to finish. When running the check phase there could be a message telling us that there are failed configurations in the `water-reactive` folder! We can see which calculations did not converge in the `water-reactive/water-reactive_step2_not_converged.txt` file. Suppose there were 2 failed jobs, the 13-th and the 54-th. We might just do `touch water-reactive/00013/skip` and `touch water-reactive/00054/skip` and run the `check` phase again. This time it will inform us that some configurations will be skipped, but the final message should be that check phase is a success. All that is left to do now is run the `extract` phase, clean up with the `clean` phase, store wavefunctions and remove all unwanted data and finally update our local folder. We have now augmented our total training set and might do a new training iteration and keep iterating until convergence is reached!


Finally, we can check the `labeling_001.json` file in `$WORKDIR/control/`: 

```JSON
{
    "labeling_program": "cp2k",
    "user_machine_keyword_label": "mykeyword1",
    "systems_auto": {
        "ch3br_cl_close_300K": {
        // labeling parameters from used_input.json
        },
        "ch3br_cl_free_300K": {
        //
        },
        "ch3br_cl_smd_300K": {
        //
         },
        "ch3cl_br_close_300K": {
        //
        },
        "ch3cl_br_free_300K": {
        //
         },
        "ch3cl_br_smd_300K": {
        }
    },
    "total_to_label": 50,
    "launch_all_jobs": true,
    "is_locked": true,
    "is_launched": true,
    "is_checked": true,
    "is_extracted": true
}
```

The total number of structures that have been selected labeled from the selected candidates in the previous exploration step is indicated with the `"total_to_label"` keyword. 

The first iteration is done. After executing the `extract` phase, the directories for the next iteration will be created. 
