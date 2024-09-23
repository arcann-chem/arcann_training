# Labeling 

In the labeling phase we will use the CP2K code to compute the electronic energies, atomic forces and (sometimes) the stress tensor of the candidate configurations obtained in the exploration phase.

In the case you are performing the labeling step in a different HPC machine, don't forget to copy the data (you must also install the ArcaNN software and create a python environment!) beforehand:

```bash
rsync -rvu $WORK_DIR USER@OTHER_HPC_MACHINE:PATH_TO_WORKDIR
```

For this we need to go to the `XXX-labeling` folder and as usual run the `prepare` phase. It is very important to have a look at the `default_input.json` of the `prepare` phase to choose the computational resources to be used in the electronic structure calculations (number of nodes and MPI/OpenMP tasks). Note that the default values are insufficient for most condensed systems (due to the large number of atoms), so you should have previously determined the resources required by your specific system(s). 

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

The `"use_machine_keyword_label"` keyword corresponds to the partition in the HPC machine, The `"nb_mpi_per_node"` and `"nb_nodes"` keywords set the number of CPU nodes used for the labeling. The wall times should be set for the first iteration but can be guessed automatically later using the average time per CP2K calculation measured in the previous iteration. 

Once you have executed this phase, folders will have been created for each subsystem within which there will be as many folders as candidate configurations (maximum number of 99999 per iteration), containing all required files to run CP2K. Make sure that you have prepared (and correctly named!) Slurm submission files for your machine in the `$WORK_DIR/user_files/` folder (see [Initialization](../initialization)), from the template files. 

For CP2K calculations, 2 scripts must be prepared : a first quick calculation at a lower level of theory and then a second one at our reference level.

You can then submit the calculations by executing the `launch` phase. Once these are finished you can check the results with  the `check` phase. Since candidate configurations are not always very stable (or even physically meaningful if you were too generous with deviation thresholds) some DFT calculations might not have converged. This will be indicated in the output of the `check` phase.  You can either perform manually the calculations with a different setup until the result is satisfactory or skip the problematic configurations by creating empty `skip` files in the folders that should be ignored. Keep running `check` until you get a "Success!" message. Use the `extract` phase to set up everything for the training phase and eventually run the `clean` phase to clean up your folder. CP2K wavefunctions might be stored in an archive with a command given by the code that must be executed manually (if one wishes to keep these files as, for example, starting points for higher level calculations). You can also delete all files but the archives created by the code if you want. We have now augmented our total training set and might do a new training iteration and keep iterating until convergence is reached!