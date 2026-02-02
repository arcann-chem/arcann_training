# Getting Started with ArcaNN

Welcome! This section will guide you through everything you need to set up ArcaNN on your system and configure it for your HPC cluster.

## Quick Start Path

Follow these steps in order to get ArcaNN up and running:

1. Check Requirements → 2. Install ArcaNN → 3. Configure HPC → 4. Start Using!

---

## Step-by-Step Setup Guide

### Step 1 : Check Your System Requirements

**Status**: Start here first

[**Requirements**](./requirements.md)

Before installing ArcaNN, ensure your system has all necessary dependencies:

- **Python** version >= 3.10
- **External tools**: VMD, Atomsk
- **Scientific packages**: NumPy, pip, setuptools
- **Domain-specific software**: DeePMD-kit, LAMMPS (or i-PI), CP2K

---

### Step 2 : Install ArcaNN

**Status**: After requirements are satisfied

[**Installation Guide**](./installation.md)

Two installation methods available:

- **With internet access**: Direct conda environment creation + pip install
- **Without internet access**: Download packages offline, transfer to target machine

**Choose your installation method**:

- Have internet? [Install with internet](./installation.md#installation-on-machines-with-internet-access)
- No internet? [Install offline](./installation.md#installation-on-machines-without-internet-access)

---

### Step 3 : Configure Your HPC Cluster

**Status**: After ArcaNN installation

[**HPC Configuration**](./hpc_configuration.md)

Set up your `machine.json` file to connect ArcaNN to your HPC resources:

- Define cluster connection details (hostname, scheduler)
- Configure compute partitions (CPU/GPU allocations)
- Set up job submission parameters (walltime, nodes, etc.)
- Support for: **SLURM**, **PBS/Torque**, and other schedulers

**Key file**: `machine.json` in your `$WORK_DIR/user_files/` folder

---

### Step 4 : Ready to Use ArcaNN

**Status**: After HPC configuration

Once you complete the three steps above, you're ready to start the iterative training procedure!

**Next Steps**:

1. Go to [Using ArcaNN](../usage/start.md) section
2. Set up your [Prerequisites](../usage/iter_prerequisites.md) (create user files)
3. Run [Initialization](../usage/initialization.md)
4. Start your first training iteration!

---

## Navigation

← [Home](../index.md) | [Next: Check Requirements →](./requirements.md)

**Full Workflow**: [Home](../index.md) > **Getting Started** > [Requirements](./requirements.md) > [Installation](./installation.md) > [HPC Configuration](./hpc_configuration.md) > [Using ArcaNN](../usage/start.md)
