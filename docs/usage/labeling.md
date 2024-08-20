# Labeling #

## Labeling ##

In the labeling phase we will use the `CP2K` code to compute the electronic energies, atomic forces and (sometimes) the stress tensor of the candidate configurations obtained in the exploration phase. For this we need to go to the `XXX-labeling` folder and as usual run the `prepare` phase. It is very important to have a look at the `default_input.json` of the `prepare` phase to choose the computational resources to be used in the electronic structure calculations (number of nodes and MPI/OpenMP tasks). Note that the default values are insufficient for most condensed systems, so you should have previously determined the resources required by your specific system(s). Once you have executed this phase, folders will have been created for each subsystem within which there will be as many folders as candidate configurations (maximum number of 99999 per iteration), containing all required files to run CP2K. Make sure that you have prepared (and correctly named!) template `Slurm` submission files for your machine in the `$WORK_DIR/user_files` folder ([Initialization](../initialization)). You can then submit the calculations by executing the `launch` phase. Once these are finished you can check the results with  the `check` phase. Since candidate configurations are not always very stable (or even physically meaningful if you were too generous with deviation thresholds) some DFT calculations might not have converged, this will be indicated by the code. You can either perform manually the calculations with a different setup until the result is satisfactory or skip the problematic configurations by creating empty `skip` files in the folders that should be ignored. Keep running `check` until you get a "Success!" message. Use the `extract` phase to set up everything for the training phase and eventually run the `clean` phase to clean up your folder. CP2K wavefunctions might be stored in an archive with a command given by the code that must be executed manually (if one wishes to keep these files as, for example, starting points for higher level calculations). You can also delete all files but the archives created by the code if you want.

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
