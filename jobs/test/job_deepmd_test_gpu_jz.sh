#!/bin/bash
# Author: Rolf DAVID
# Date: 2021/03/16
# Modified: 2022/10/12
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
#SBATCH -o DeepMD_Test.%j
#SBATCH -e DeepMD_Test.%j
# Name of job
#SBATCH -J DeepMD_Test_XXX
# Email (Remove the space between # and SBATCH on the next two lines)
##SBATCH --mail-type FAIL,BEGIN,END,ALL
##SBATCH --mail-user _EMAIL_
#

# Input files
DeepMD_MODEL_VERSION="_DEEPMD_MODEL_VERSION_"
DeepMD_PB="_DEEPMD_PB_"
DeepMD_DATA_DIR="data"

#!!Nothing needed to be changed past this point

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || exit 1

# Load the environment depending on the version
if [ "${SLURM_JOB_QOS:4:3}" == "gpu" ]; then
    if [ "${DeepMD_MODEL_VERSION}" == "2.1" ]; then
        module purge
        . /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.1.4-cuda11.6_plumed-2.8.0/etc/profile.d/conda.sh
        conda activate /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.1.4-cuda11.6_plumed-2.8.0
    elif [ "${DeepMD_MODEL_VERSION}" = "2.0" ]; then
        module purge
        . /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.0.3-cuda10.1_plumed-2.7.4/etc/profile.d/conda.sh
        conda activate /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.0.3-cuda10.1_plumed-2.7.4
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
if [ ! -f ${DeepMD_PB}.pb ]; then echo "No input file found. Aborting..."; exit 1; fi

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}"/JOB-"${SLURM_JOBID}"

# Copy files to the temporary work directory
cp ${DeepMD_PB}.pb "${TEMPWORKDIR}" && echo "${DeepMD_PB}.pb copied successfully"
[ -d ${DeepMD_DATA_DIR} ] && mkdir -p "${TEMPWORKDIR}"/data && cp -r ${DeepMD_DATA_DIR}/* "${TEMPWORKDIR}"/data && echo "${DeepMD_DATA_DIR} copied successfully"
cd "${TEMPWORKDIR}" || exit 1

# MPI/OpenMP setup
echo "# [$(date)] Started"
export TASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES ))
echo "Running on node(s): ${SLURM_NODELIST}"
echo "Running on ${SLURM_NNODES} node(s)."
echo "Running ${SLURM_NTASKS} task(s), with ${TASKS_PER_NODE} task(s) per node."
echo "Running with ${SLURM_CPUS_PER_TASK} thread(s) per task."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
SRUN_DeepMD_EXE="srun --export=ALL --mpi=pmix --ntasks=${SLURM_NTASKS} --nodes=${SLURM_NNODES} --ntasks-per-node=${TASKS_PER_NODE} --cpus-per-task=${SLURM_CPUS_PER_TASK} ${DeepMD_EXE}"

# Launch command
for f in data/* ; do
    LAUNCH_CMD="${SRUN_DeepMD_EXE} test -m ${DeepMD_PB}.pb -s ${f} -d out -n 1000000"
    echo "${LAUNCH_CMD}"
    export EXIT_CODE="0"
    ${LAUNCH_CMD} > "${DeepMD_PB}_${f/data\//}.log" 2> "${DeepMD_PB}_${f/data\//}.err" || export EXIT_CODE="1"
    mv out.e.out "${DeepMD_PB}_${f/data\//}.e.out"
    mv out.f.out "${DeepMD_PB}_${f/data\//}.f.out"
    mv out.v.out "${DeepMD_PB}_${f/data\//}.v.out"
done

echo "# [$(date)] Ended"

# Move back data from the temporary work directory and scratch, and clean-up
rm -rf "${TEMPWORKDIR}"/data
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || exit 1
if [ ! -d logs_XXX ]; then
    mkdir logs_XXX || exit 1
fi
if [ ! -d out_XXX ]; then
    mkdir out_XXX || exit 1
fi
if [ "${DeepMD_MODEL_VERSION}" = "2.0" ] || [ "${DeepMD_MODEL_VERSION}" = "2.1" ] ;then
    for f in "${DeepMD_PB}"*.err ; do grep 'DEEPMD INFO' "${f}" > "${f/.err/.log}" ; done
fi
mv ./*.log ./*.err logs_XXX/
mv ./*.out out_XXX/
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

# Done
echo "Have a nice day !"

# A small pause before SLURM savage clean-up
sleep 5
exit ${EXIT_CODE}
