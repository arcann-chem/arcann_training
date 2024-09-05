# HPC Configuration 

ArcaNN is designed for use on one or several HPC machines, whose specific configurations must be specified by the user through a `machine.json` file.
A general example file can be found in the [GitHub Repository](https://github.com/arcann-chem/arcann_training/blob/main/examples/user_files/machine.json).
You should modify this file to suit your setup and then copy it to the `user_files/` folder in your working directory (see later in [Usage](../usage/iter_prerequisites.md)).

## Structure of the `machine.json` File ##

The `machine.json` file is organized as a JSON dictionary with one or more keys that designate different HPC machines. The typical structure looks like this:

```json
{
    "myHPCkeyword1": {ENTRIES THAT DESCRIBE HPC 1},
    "myHPCkeyword2": {ENTRIES THAT DESCRIBE HPC 2},
    // Additional machines can be added here
}
```


Each key in the JSON file is a short string designating the name of the machine (e.g., `"myHPCkeyword1"`, `"myHPCkeyword2"` are the names of 2 different HPC machines).
The value associated with each key is a dictionary indicating the configuration entries for running jobs on the corresponding HPC machine.

Below is an example of the initial entries for an HPC machine using a SLURM job scheduler:

```json
{
    "myHPCkeyword1": {
        "hostname": "myHPC1",
        "walltime_format": "hours",
        "job_scheduler": "slurm",
        "launch_command": "sbatch",
        "max_jobs": 200,
        "max_array_size": 500,
        "mykeyword1": {
            "project_name": "myproject",
            "allocation_name": "myallocationgpu1",
            "arch_name": "a100",
            "arch_type": "gpu",
            "partition": "mypartitiongpu1",
            "subpartition": "mysubpartitiongpu1",
            "qos": {
                "myqosgpu1": 72000,
                "myqosgpu2": 360000
            },
            "valid_for": ["training"],
            "default": ["training"]
        },
        "mykeyword2": { /* Additional partition configurations */ }
    },
    /* Additional HPC machines can be added here */
}
```

## HPC Entry ##

Each HPC machine entry contains a JSON directonary where each key corresponds to a configuration entry. 

- **hostname**: A substring contained in the output of `python -c "import socket ; print(socket.gethostname())"`. This should match your machine's name.
- **walltime_format**: The unit of time (e.g., hours) used to specify wall time on the cluster.
- **job_scheduler**: The job scheduler used by your HPC machine (e.g., `slurm`, `PBS/Torque`). ArcaNN is extensively tested with `Slurm`.
- **launch_command**: The command for submitting jobs (e.g., `sbatch` for `Slurm`, `qsub` for `PBS/Torque`).
- **max_jobs**: Maximum number of jobs per user allowed by the scheduler. Can also be a user-defined safety limit.
- **max_array_size**: Maximum number of jobs in a single job array. This is important for `Slurm` as ArcaNN relies heavily on job arrays.

## Resource Configuration ##

Several resources can be available for calculation within the same HPC machine. 
Each available resource in the HPC machine is represented by a key (e.g., `"mykeyword1"`) and includes:

- **project_name**: Name of the project using the HPC resources. 
It will correspond to the `_R_PROJECT_` keyword in the `#SBATCH --account=_R_PROJECT_` line of the slurm job. 
- **allocation_name**: Allocation or account name, typically used in large HPC facilities.
It will correspond to the `_R_ALLOC_` keyword in the `#SBATCH --account=_R_PROJECT_@_R_ALLOC_` line of the slurm job. 
- **arch_name**: Architecture name (e.g., `a100` for GPU nodes).
- **arch_type**: Architecture type (e.g., `gpu` or `cpu`).
- **partition**: The partition on the HPC machine.
- **subpartition**: (Optional) Subpartition within the main partition.
- **qos**: Quality of Service settings, with corresponding time limits in seconds.
- **valid_for**: Specifies the steps this partition is valid for (e.g., `["training", "freezing", "compressing", "exploration", "test", "labeling"]`).
- **default**: Indicates the default partition for specific steps.

## Customization and Submission Files ##

You can add multiple partition configurations as needed. For example, `"mykeyword1"` could represent a GPU partition using A100 GPU nodes, which is used for training unless a different partition is specified.

If your HPC setup does not include projects, allocations, partitions, or subpartitions, you can omit the corresponding keywords.

To run ArcaNN on your HPC machine, you must provide example submission files tailored to your system.
These files should be modeled after the `examples/user_files/job*/*.sh` files and **must include the replaceable strings** indicated by a `_R_` prefix and suffix.
Place these files in the `$WORK_DIR/user_files/` folder, which you must create to use ArcaNN for a specific system (see [Usage](../usage/iter_prerequisites)).

