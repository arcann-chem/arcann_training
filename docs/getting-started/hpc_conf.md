# HPC Configuration #

ArcaNN was designed for use on one or several HPC machines whose specificities must be indicated by the user through a `machine.json` file.
You can find a general example file in `arcann_training/examples/user_files/machine.json` that you should modify to adapt it to your setup and that you will need to copy to the `user_files/` folder of your working directory (see [Usage](../usage/iter_prerequisites)).
This file is organized as a `JSON` dictionary with one or several keys that designate the different HPC machines, the typical structure looks like this:

```json
{
    "myHPCkeyword1":
        {ENTRIES THAT DESCRIBE HPC 1},
    "myHPCkeyword2":
        {ENTRIES THAT DESCRIBE HPC 2},
    etc.
}
```

Each key of the JSON file is a short string designating the name of the machine (here `"myHPCkeyword1"`, `"myHPCkeyword2"`, etc.).
The associated entry is also a dictionary whose keys are keywords (or further dictionaries of keywords) associated with information needed to run jobs in the corresponding HPC machine.
Let's have a look at the first few entries of the `"myHPCkeyword1"` machine for a SLURM job scheduler:

```json
{
    "myHPCkeyword1":
    {
        "hostname": "myHPC1",
        "walltime_format": "hours",
        "job_scheduler": "slurm",
        "launch_command": "sbatch",
        "max_jobs" : 200,
        "max_array_size" : 500,
        "mykeyword1": {
            "project_name": "myproject",
            "allocation_name": "myallocationgpu1",
            "arch_name": "a100",
            "arch_type": "gpu",
            "partition": "mypartitiongpu1",
            "subpartition": "mysubpartitiongpu1",
            "qos": {"myqosgpu1": 72000, "myqosgpu2": 360000},
            "valid_for":  ["training"],
            "default": ["training"]
        },
        "mykeyword2": { etc. },
        etc.
    }
    etc.
}
```

- `"hostname"` is a substring contained in the output of the following command: `python -c "import socket ; print(socket.gethostname())"`, which should indicate your machine's name.
- `"walltime_format"` is the time unit used to specify the wall time to the cluster.
- `"job_scheduler"` is the name of the job scheduler used in your HPC machine. The code has been extensively used with `Slurm` and has been tested with other schedulers.
- `"launch_command"` is the bash command used for submitting jobs to your cluster (typically `sbatch` in normal `Slurm` setups, but you can adapt it to match your cluster requirements, such as `qsub` for machines running `PBS/Torque`).
- `"max_jobs"` is the maximum number of jobs per user allowed by the job scheduler of your HPC machine.
You can also use this to set a safety limit if your scheduler does not impose one by default.
- `"max_array_size"` is the maximum number of jobs that can be submitted in a single job array (in `Slurm`, the preferred usage of the `ArcaNN` suite relies heavily on arrays to submit jobs).
- The next keyword is the key name of a partition.
It should contain all the information needed to run a job in that partition of your cluster.
The keyword names are self-explanatory.
The keyword `"valid_for"` indicates the steps that can be performed in this partition (possible options include `["training", "freezing", "compressing", "exploration", "test", "labeling"]`).
The `"default"` keyword indicates that this partition of the machine is the default one used (if not explicitly indicated by the user) for the specified steps.

You can add as many partition keywords as you need.
In the above example, `"mykeyword1"` is a GPU partition that uses `A100` `GPU` nodes.
We will use this for every iteration, unless the user explicitly specifies a different partition for the **training** step.
Note that this example assumes that the HPC machine is divided into projects with allocated time (indicated in `"project_name"` and `"allocation_name"`), as is typical in large HPC facilities used by different groups.
If this does not apply to your HPC machine, you don't need to provide these keywords.
Likewise, if there are no partitions or subpartitions, the corresponding keywords need not be provided.
Finally, to use your HPC machine, you will need to provide example submission files tailored to your machine.
These should follow the style of the `examples/user_files/job*/*.sh` files, **keeping the replaceable strings** indicated by a `_R_` prefix and a `_` suffix.
Place these files in the `$WORK_DIR/user_files/` folder, which you must create to use `ArcaNN` for a particular system (see [Usage](../usage/iter_prerequisites)).
