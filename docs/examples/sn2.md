# SN2

Here we introduce the basic usage of the ArcaNN software, ilustrated by a SN2 reaction. All the files can be found in the github page. 

Two iterative trainings were done : a first non-reactive training on reactant and products structures, followed by a reactive training where transition structures were generated. 

The non-reactive ArcaNN training in detailed in the following sections. 

## User files
We will start by creating a `user_files/` directory (See [Iterative procedure prerequisites](../usage/iter_prerequisites.md)) where we will include the necessary files for each step of the procedure. You also need to create a `data/` directory where the initial labeled datasets will be stored. For the reactive training, you will store the datasets of the non-reactive training in the corresponding `data/` directory, together with the initial datasets. 

For the non-reactive training, 6 systems were defined : 3 systems to explore the reactant basin (`ch3cl_br_close_300K`, `ch3cl_br_free_300K`, `ch3cl_br_smd_300K`) and 3 systems to explore the product basin (`ch3br_cl_close_300K`, `ch3br_cl_free_300K`, `ch3br_cl_smd_300K`). 

In the `user_files/` folder you will find the following files for each one of the systems (for clarity purposes, we replaced the system's name by `SYSTEM`)

### JSON FILES

- `machine.json` : file containing the cluster parameters.
- `dp_train_2.1.json` : input for DeePMD trainings. 


### JOB FILES 

- `job_lammps-deepmd_explore_gpu_hpc1.sh` and `job-array_lammps-deepmd_explore_gpu_hpc1.sh` : job scripts for exploration
- `job_CP2K_label_cpu_hpc1.sh` and `job-array_CP2K_label_hpc1.sh`: job scripts for labeling
- `job_deepmd_compress_gpu_hpc1.sh`, `job_deepmd_freeze_gpu_hpc1.sh` and `job_deepmd_train_gpu_hpc1.sh` job scripts for training


### CP2K FILES 

- `1_SYSTEM_labeling_XXXXX_hpc1.inp`, `2_SYSTEM_labeling_XXXXX_hpc1.inp`, `1_SYSTEM_labeling_XXXXX_hpc1.inp`, `2_SYSTEM_labeling_XXXXX_hpc2.inp` : inputs for CP2K labeling. There are 2 input files per subsystem, see details in [labeling](../labeling). `hpc1` and `hpc2` are the machine keywords indicated in the `machine.json` file.


### LAMMPS FILES

- `SYSTEM.lmp` : starting configurations for the first exploration in the LAMMPS format. 
- `SYSTEM.in` : inputs for LAMMPS exploration. 


#### plumed files 

- `plumed_SYSTEM_300K.dat` : plumed input files for the emplorations. 

Additional plumed files can be used, and must be named as `plumed_KEYWORD_SYSTEM.dat`. Here, we used an additional plumed file to store colvars and another to define the key atoms : `plumed_colvars_SYSTEM.dat` and `plumed_atomdef_SYSTEM.dat`. 


#### properties file

The atom order is defined in the `properties.txt` file. It makes sure that the order of the  atoms in the `SYSTEM.lmp` files match the order indicated in the `"type_map"` keyword of the DeePMD-kit `dptrain_2.1.json` training file. Also, it makes sure that the generated structures also presents the correct atom numbering to avoid conflicts. 




## Initialization

### INPUTS

Let's use the above example of a NNP for water and ice that is able to describe water self-dissociation. Suppose that you want 3 subsystems (ice, un-dissociated liquid water, water with a dissociated pair) your `defaut_input.json` file might look like this:

```JSON
{
    "step_name": "initialization",
    "systems_auto": ["ice", "water", "water-reactive"],
    "nnp_count": 3
}
```

Decrire le used_inputs
et quelle partie est utilisée dans quelle phase
### OUTPUTS
Decrire les control/*.json

## Training

### INPUTS
### OUTPUTS

## Exploration

After the first training phase NNP you now have starting models that can be used to propagate reactive MD. For this go to the `$WORK_DIR/001-exploration` folder (in your HPC machine!) and execute the `prepare` phase to obtain an `default_input.json` file with default values. For the first iteration we might be satisfied with the defaults (2 simulations per NNP and per subsystem, 10 ps simulations with the LAMMPS time-step of 0.5 fs, etc.) so we might directly run exploration phases 2 and 3 right away (waiting for the `Slurm` jobs to finish as always). These will have created 6 directories (one per system), in which there will be 3 subdirectories (one per trained NNP) `1/`, `2/` and `3/`, in which again there will be 2 subdirectories (default) `0001/` and `0002/`. This means that a total of 32 MD trajectories will be performed for this first iteration (180 ps total simulation time). Be careful, the total exploration time can quickly become huge, especially if you have many systems.

### INPUTS

For the first exploration phase we might want to generate only a few candidate configurations to check whether our initial NNP are stable enough to give physically meaningful configurations. We might as well want to use a relatively strict error criterion for candidate selection. All this can be done by modifying the default values written to `input.json` at the `deviate` phase and re-running this phase. In the end, your input file might look like this:

```JSON
{
    "step_name": "exploration",
    "user_machine_keyword_exp": "mykeyword1",
    "slurm_email": "",
    "atomsk_path": "PATH_TO_THE_ATOMSK_BINARY",
    "vmd_path": "PATH_TO_THE_VMD_BINARY",
    "exploration_type": ["lammps", "lammps", "lammps"],
    "traj_count": [2, 2, 2],
    "temperature_K": [273.0, 300.0, 300.0],
    "timestep_ps": [0.0005, 0.0005, 0.0005],
    "previous_start": [true, true, true],
    "disturbed_start": [false, false, false],
    "print_interval_mult": [0.01, 0.01, 0.01],
    "job_walltime_h": [-1, -1, -1],
    "exp_time_ps": [10, 10, 10],
    "max_exp_time_ps": [400, 400, 400],
    "max_candidates": [50, 50, 100],
    "sigma_low": [0.1, 0.1, 0.1],
    "sigma_high": [0.8, 0.8, 0.8],
    "sigma_high_limit": [1.5, 1.5, 1.5],
    "ignore_first_x_ps": [0.5, 0.5, 0.5],
    "init_exp_time_ps": [-1, -1, -1],
    "init_job_walltime_h": [-1, -1, -1],
    "disturbed_candidate_value": [0.5, 0, 0],
    "disturbed_start_value": [0.0, 0.0, 0.0],
    "disturbed_start_indexes": [[], [], []],
    "disturbed_candidate_indexes": [[], [], []]
}
```

We have indicated the path to the `Atomsk` code used for creating the disturbed geometries at the beginning of the input file. We allow for slightly larger deviations (`"sigma_high"` keyword set to 0.8 eV/Ang) and collect a larger number of candidates (`"max_candidates"` set to 100) for the more complex third system (reactive water).
At this stage we should decide wether we want to include disturbed candidates in the training set. Here we might want to do so only for the ice system, since explorations at lower temperature explore a more reduced zone of the phase space and it is easier to be trapped in meta-stable states. This can be done by setting `disturbed_start_value` to `0.5`. The values in `disturbed_start_value` are used to disturb the starting structures for the next iteration. For the 2 other systems `disturbed_start_value` and `disturbed_candidate_value` are set to `0.0` in order to avoid disturbance. A non-zero value sets the maximal amplitude of the random translation vector that will be applied to each atom (a different vector for each atom) in Å.

**Note:** we have indicated the path to a `VMD` executable, this is not needed if `vmd` is inmediately available in our path when executing the `extract` phase (loaded as a module for example). Similarly, we can remove `atomsk_path` if `atomsk` is already in the path.

We can finally clean up the working folder by running the `clean` phase and move on to the labeling phase! (Don't forget to keep your local folder updated so that you can analyze all these results)



### OUTPUTS

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


### INPUTS
### OUTPUTS

## Test

### INPUTS
### OUTPUTS


- If you want train an NNP to study a reaction, such as an SN2 reaction, you would like to include in the training set configurations representing the reactant, product and transition states.
In this case, we would start by defining two **systems** (`reactant` and `product`) and generating structures in both bassins by performing several iterations (exploring the chemical space, labeling the generated structures and training the NNP on the extented *dataset*).
Next, you would also want to generate transition structures between the `reactant` and the `product`.
For that, you would need to performed biased explorations with the PLUMED software (see [Exploration](../exploration)) within different **systems**.

