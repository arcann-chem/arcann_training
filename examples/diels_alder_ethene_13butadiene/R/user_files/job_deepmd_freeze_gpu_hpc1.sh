#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2024/06/26
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
# You must keep the name file as job_deepmd_freeze_ARCHTYPE_myHPCkeyword.sh.
#----------------------------------------------
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
#SBATCH -o DeepMD_Freeze.%j
#SBATCH -e DeepMD_Freeze.%j
# Name of job
#SBATCH -J DeepMD_Freeze
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

DeepMD_MODEL_VERSION="_R_DEEPMD_VERSION_"
DeepMD_MODEL_FILE="_R_DEEPMD_MODEL_FILE_"
DeepMD_CKPT_FILE="_R_DEEPMD_CKPT_FILE_"
DeepMD_LOG_FILE="_R_DEEPMD_LOG_FILE_"
DeepMD_OUT_FILE="_R_DEEPMD_OUTPUT_FILE_"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------


# Project switch
PROJECT_NAME=${SLURM_JOB_ACCOUNT:0:3}
# Compare PROJECT_NAME and IDRPROJ for inequality
if [[ "${PROJECT_NAME}" != "${IDRPROJ}" ]]; then
    SCRATCH=${SCRATCH/${IDRPROJ}/${PROJECT_NAME}}
fi

# Get the available projects
readarray -t AVAILABLE_PROJECTS < <(idr_compuse | awk '/@/{print $1}' | cut -d '@' -f 1 | sort -u)

IS_PRJ2_AVAILABLE=0
IS_PRJ1_AVAILABLE=0
for PROJECT in "${AVAILABLE_PROJECTS[@]}"; do
    if [[ "$PROJECT" == "myproject2" ]]; then
        IS_PRJ2_AVAILABLE=1
    elif [[ "$PROJECT" == "myproject1" ]]; then
        IS_PRJ1_AVAILABLE=1
    fi
done

# Get the version and the environment path
if [[ "${DeepMD_MODEL_VERSION}" == "2.2" ]]; then
    if [[ "${IS_PRJ2_AVAILABLE}" == 1 || "${PROJECT_NAME}" == "myproject2" ]]; then
        DEEPMD_CONDA_ENV_PATH=/programs/apps/deepmd-kit/2.2.7-cuda11.6_plumed-2.8.3
    elif [[ "${IS_PRJ1_AVAILABLE}" == 1 || "${PROJECT_NAME}" == "myproject1" ]]; then
        DEEPMD_CONDA_ENV_PATH=/apps/deepmd-kit/2.2.7-cuda11.6_plumed-2.8.0
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not accessible for ${USER}. Aborting..."; exit 1
    fi
elif [[ "${DeepMD_MODEL_VERSION}" == "2.1" ]]; then
    if [[ "${IS_PRJ2_AVAILABLE}" == 1 || "${PROJECT_NAME}" == "myproject2" ]]; then
        DEEPMD_CONDA_ENV_PATH=/programs/apps/deepmd-kit/2.1.5-cuda11.6_plumed-2.8.1
    elif [[ "${IS_PRJ1_AVAILABLE}" == 1 || "${PROJECT_NAME}" == "myproject1" ]]; then
        DEEPMD_CONDA_ENV_PATH=/apps/deepmd-kit/2.1.4-cuda11.6_plumed-2.8.0
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not accessible for ${USER}. Aborting..."; exit 1
    fi
elif [[ "${DeepMD_MODEL_VERSION}" == "2.0" ]]; then
    if [[ "${IS_PRJ1_AVAILABLE}" == 1 || "${PROJECT_NAME}" == "myproject1" ]]; then
        DEEPMD_CONDA_ENV_PATH=/apps/deepmd-kit/2.0.3-cuda10.1_plumed-2.7.4
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not accessible for ${USER}. Aborting..."; exit 1
    fi
else
    echo "DeePMD ${DeepMD_MODEL_VERSION} is not installed on ${SLURM_JOB_QOS}. Aborting..."; exit 1
fi

# Load the environment
. "${DEEPMD_CONDA_ENV_PATH}/etc/profile.d/conda.sh"
conda activate ${DEEPMD_CONDA_ENV_PATH}
DeepMD_EXE=$(command -v dp) || { echo "Executable (dp) not found. Aborting..."; exit 1 ; }

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f ${DeepMD_CKPT_FILE} ] || { echo "${DeepMD_CKPT_FILE} does not exist. Aborting..."; exit 1; }

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

SRUN_DeepMD_EXE="srun --export=ALL --mpi=pmix --ntasks=${SLURM_NTASKS} --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} --cpus-per-task=${SLURM_CPUS_PER_TASK} ${DeepMD_EXE}"

# Run the DeepMD freeze
echo "# [$(date)] Running DeepMD freeze..."
${SRUN_DeepMD_EXE} freeze -o ${DeepMD_MODEL_FILE} --log-path ${DeepMD_LOG_FILE} > ${DeepMD_OUT_FILE} 2>&1
echo "# [$(date)] DeepMD freeze finished."

# Done
echo "Have a nice day !"

sleep 2
exit