# Training 

During the training procedure you will use DeePMD-kit to train neural networks on the data sets that you have thus far generated (or on the initial ones only for the `000-training`). In order to do this go to the current iteration training folder `XXX-training`.
There are 9 phases (see [Iterations, Steps and Phases of the Iterative Procedure](../start)) that you must now execute in order after having optionally modified the `input.json` file to define the relevant parameters (in case you want something different from the defaults, which are written to `default_input.json` in the `prepare` phase). The input keywords that you should check the most carefully are those related to the first phase `prepare`, as this sets all the important parameters for the training. Some phases will simply submit `Slurm` jobs (model training, freezing and compressing). You must wait for the jobs to finish before executing the next phase (generally this will be a check phase that will tell you that jobs have failed or are currently running). Once you have executed the first 8 phases, the training iteration is done! Executing the 9-th phases is optional, as this will only remove intermediary files.

After running the `initialization` step described in the previous example, you must now perform the first training phase. Update (or copy for the first time) the full `$WORK_DIR` from your local machine to your HPC machine (where you must have also a copy of this repository and an environment in which it is installed):

```bash
rsync -rvu $WORK_DIR USER@HPC-MACHINE:/PATH/TO/WORK_DIR
```

Now go to the empty `000-training` folder created by the script execute the `prepare` phase:

```bash
python -m arcann_training training prepare
```

This will create three folders `1/`, `2/` and `3/` and a copy of your `data/` folder, as well as a `default_input.json` file containing the default training parameters. If you want to modify some of the default values you can create a `input.json` file from the `default_input.json` file that looks like this:

```JSON
{
    "step_name": "training",
    "user_machine_keyword_train": "mykeyword1",
    "user_machine_keyword_freeze": "mykeyword2",
    "user_machine_keyword_compress": "mykeyword2",
    "slurm_email": "",
    "use_initial_datasets": true,
    "use_extra_datasets": false,
    "deepmd_model_version": 2.2,
    "job_walltime_train_h": 4,
    "mean_s_per_step": -1,
    "start_lr": 0.001,
    "stop_lr": 1e-06,
    "decay_rate": 0.9172759353897796,
    "decay_steps": 5000,
    "decay_steps_fixed": false,
    "numb_steps": 400000,
    "numb_test": 0,
}
```

Here the `"user_machine_keyword"` should match the `"myHPCkeyword1"` keyword in the `machine.json` (see [HPC Configuration](../../getting-started/hpc_conf)). Note that the more performant GPUs should ideally be used for training, while the other steps could be alllocated to less performant GPUs or even to CPUs. Here we used a user chosen walltime of 4 h (instead of the default indicated by `-1`, which will calculate the job walltime automatically based on your previous trainings).
The followiing keywords are the DeePMD training parameters, that you can eventually modify or keep the default values.
 We can then execute all the other phases in order (waiting for `Slurm` jobs to finish!). That's it! Now you just need to update the local folder:

```bash
rsync -rvu USER@HPC-MACHINE.fr:/PATH/TO/WORK_DIR $WORK_DIR
```

and you are ready to move on to the exploration phase!

**Notes:**

- At some point during the iterative procedure we might want to get rid of our initial data sets, we would only need to set the `use_initial_datasets` variable to `False`.
- We might also have generated some data independently from the iterative procedure that we might want to start using, this can be done by copying the corresponding DeePMD-kit systems to `data/`, prefixing their names by `extra_` and setting the `use_extra_datasets` variable to `True`.
- At the end of the step the last phase `increment` will create the folders needed for the next iteration, save the current NNPs (stored as graph files `graph_[nnp_count]_XXX[_compressed].pb`) into the `$WORK_DIR/NNP` folder and write a `control/training_XXX.json` file with all parameters used during training.
