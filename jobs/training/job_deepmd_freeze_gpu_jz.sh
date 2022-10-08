#!/bin/bash
# Author: Rolf DAVID
# Date: 2021/03/16
# Modified: 2022/10/08
# Account
#SBATCH --account=_PROJECT_@_ALLOC_
# Queue
#SBATCH --qos=_QOS_
#SBATCH --partition=_PARTITION_
#SBATCH -C _SUBPARTITION_
# Number of nodes/processes/tasksperprocess
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
# Wall-time
#SBATCH -t _WALLTIME_
# Merge Output/Error
#SBATCH -o DeepMD_Freeze.%j
#SBATCH -e DeepMD_Freeze.%j
# Name of job
#SBATCH -J DeepMD_Freeze
# Email (Remove the space between # and SBATCH on the next two lines)
##SBATCH --mail-type FAIL,BEGIN,END,ALL
##SBATCH --mail-user _EMAIL_
#

# Input files
DeepMD_MODEL_VERSION="SET_DEEPMD_MODEL_VERSION"
DeepMD_CHKPT_F="checkpoint"
DeepMD_PB="DeepMD_PB_F"

#----------------------------------------------
## Nothing needed to be changed past this point

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || exit 1

# Load the environment depending on the version
if [ "${SLURM_JOB_QOS:4:3}" == "gpu" ]; then
    if [ "${DeepMD_MODEL_VERSION}" == "2.1" ]; then
        module purge
        . /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.1.4-cuda11.6_plumed-2.8.0/etc/profile.d/conda.sh
        conda activate /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.1.4-cuda11.6_plumed-2.8.0
        log="--log-path ${DeepMD_PB}_freeze.log"
    elif [ "${DeepMD_MODEL_VERSION}" = "2.0" ]; then
        module purge
        . /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.0.3-cuda10.1_plumed-2.7.4/etc/profile.d/conda.sh
        conda activate /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.0.3-cuda10.1_plumed-2.7.4
        log="--log-path ${DeepMD_PB}_freeze.log"
     elif [ "${DeepMD_MODEL_VERSION}" = "1.3" ]; then
        module purge
        . /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/1.3.3-cuda10.1_plumed-2.6.2/etc/profile.d/conda.sh
        conda activate /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/1.3.3-cuda10.1_plumed-2.6.2
        log=""
# This one is IDRIS installed, so it is here forever
    elif [ "${DeepMD_MODEL_VERSION}" = "1.1" ]; then
        module purge
        module load tensorflow-gpu/py3/1.14-deepmd
        log=""
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not installed on ${SLURM_JOB_QOS}. Aborting..."; exit 1
    fi
elif [ "${SLURM_JOB_QOS:3:4}" == "cpu" ]; then
    echo "GPU on a CPU partition?? Aborting..."; exit 1
else
    echo "There is no ${SLURM_JOB_QOS}. Aborting..."; exit 1
fi
DeepMD_EXE=$(which dp) || ( echo "Executable not found. Aborting..."; exit 1 )

# Test if input file is present
if [ ! -f ${DeepMD_CHKPT_F} ]; then echo "No checkpoint file found. Aborting..."; exit 1; fi

# MPI/OpenMP setup
echo "# [$(date)] Started"
export EXIT_CODE="0"
export TASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES ))
echo "Running on node(s): ${SLURM_NODELIST}"
echo "Running on ${SLURM_NNODES} node(s)."
echo "Running ${SLURM_NTASKS} task(s), with ${TASKS_PER_NODE} task(s) per node."
echo "Running with ${SLURM_CPUS_PER_TASK} thread(s) per task."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
SRUN_DeepMD_EXE="srun --export=ALL --mpi=pmix --ntasks=${SLURM_NTASKS} --nodes=${SLURM_NNODES} --ntasks-per-node=${TASKS_PER_NODE} --cpus-per-task=${SLURM_CPUS_PER_TASK} ${DeepMD_EXE}"
LAUNCH_CMD="${SRUN_DeepMD_EXE} freeze -o ${DeepMD_PB}.pb ${log}"

# Launch command
${LAUNCH_CMD} >"${DeepMD_PB}_freeze.out" 2>&1 || export EXIT_CODE="1"
echo "# [$(date)] Ended"

# Done
echo "Have a nice day !"

# A small pause before SLURM savage clean-up
sleep 5
exit ${EXIT_CODE}