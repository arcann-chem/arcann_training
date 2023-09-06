#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2023/09/06
# Project/Account
#SBATCH --account=_R_PROJECT_@_R_ALLOC_
# QoS/Partition/SubPartition
#SBATCH --qos=_R_QOS_
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o DeepMD_Compress.%j
#SBATCH -e DeepMD_Compress.%j
# Name of job
#SBATCH -J DeepMD_Compress
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

# Input files
DeepMD_MODEL_VERSION="_R_DEEPMD_VERSION_"
DeepMD_MODEL="_R_DEEPMD_MODEL_"

#----------------------------------------------
# Nothing needed to be changed past this point

# Project Switch and update SCRATCH
PROJECT_NAME=${SLURM_JOB_ACCOUNT:0:3}
eval "$(idrenv -d "${PROJECT_NAME}")"
# Compare PROJECT_NAME and IDRPROJ for inequality
if [[ "${PROJECT_NAME}" != "${IDRPROJ}" ]]; then
    SCRATCH=${SCRATCH/${IDRPROJ}/${PROJECT_NAME}}
fi

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || exit 1

# Load the environment depending on the version
if [ "${SLURM_JOB_QOS:4:3}" == "gpu" ]; then
    if [ "${DeepMD_MODEL_VERSION}" == "2.1" ]; then
        module purge
        . /gpfsssd/worksf/projects/rech/kro/commun/programs/apps/deepmd-kit/2.1.5-cuda11.6_plumed-2.8.1/etc/profile.d/conda.sh
        conda activate /gpfsssd/worksf/projects/rech/kro/commun/programs/apps/deepmd-kit/2.1.5-cuda11.6_plumed-2.8.1
        log="--log-path ${DeepMD_MODEL}_compress.log"
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not installed on ${SLURM_JOB_QOS}. Aborting..."; exit 1
    fi
elif [ "${SLURM_JOB_QOS:3:4}" == "cpu" ]; then
    echo "GPU on a CPU partition?? Aborting..."; exit 1
else
    echo "There is no ${SLURM_JOB_QOS}. Aborting..."; exit 1
fi
DeepMD_EXE=$(command -v dp) ||  ( echo "Executable (dp) not found. Aborting..."; exit 1 )

# Test if input file is present
if [ ! -f ${DeepMD_MODEL}.pb ]; then echo "No pb file found. Aborting..."; exit 1; fi

# MPI/OpenMP setup
echo "# [$(date)] Started"
export EXIT_CODE="0"
echo "Running on node(s): ${SLURM_NODELIST}"
echo "Running on ${SLURM_NNODES} node(s)."
# Calculate missing values
if [ -z "${SLURM_NTASKS}" ]; then
    export SLURM_NTASKS=$(( SLURM_NTASKS_PER_NODE * SLURM_NNODES))
fi
if [ -z "${SLURM_NTASKS_PER_NODE}" ]; then
    export SLURM_NTASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES ))
fi
echo "Running ${SLURM_NTASKS} task(s), with ${SLURM_NTASKS_PER_NODE} task(s) per node."
echo "Running with ${SLURM_CPUS_PER_TASK} thread(s) per task."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Launch command
SRUN_DeepMD_EXE="srun --export=ALL --mpi=pmix --ntasks=${SLURM_NTASKS} --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} --cpus-per-task=${SLURM_CPUS_PER_TASK} ${DeepMD_EXE}"
LAUNCH_CMD="${SRUN_DeepMD_EXE} compress -i ${DeepMD_MODEL}.pb -o ${DeepMD_MODEL}_compressed.pb ${log}"

${LAUNCH_CMD} >"${DeepMD_MODEL}_compress".out 2>&1 || export EXIT_CODE="1"
echo "# [$(date)] Ended"

if [ -f compress.json ]; then rm compress.json; fi

# Done
echo "Have a nice day !"

# A small pause before SLURM savage clean-up
sleep 5
exit ${EXIT_CODE}
