# Initialization 


Now that you have decided the subsystems that you want to train your NNP on and prepared all the required files you can initialize the ArcaNN procedure by running (from the $WORK_DIR folder):

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

