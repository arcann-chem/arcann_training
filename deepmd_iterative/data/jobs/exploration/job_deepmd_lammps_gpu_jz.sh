#!/bin/bash
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
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o LAMMPS_DeepMD.%j
#SBATCH -e LAMMPS_DeepMD.%j
# Name of job
#SBATCH -J LAMMPS_DeepMD
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

# Input file (extension is automatically added as .in for INPUT)
# Support a list of files as a bash array
DeepMD_MODEL_VERSION="_R_DEEPMD_VERSION_"
DeepMD_MODEL=("_R_MODEL_FILES_LIST_")
LAMMPS_INPUT="_R_INPUT_FILE_"
EXTRA_FILES=("_R_DATA_FILE_" "_R_PLUMED_FILES_LIST_" "_R_RERUN_FILE_")

#----------------------------------------------
## Nothing needed to be changed past this point

### Project Switch
eval "$(idrenv -d _R_PROJECT_)"

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
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not installed on ${SLURM_JOB_QOS}. Aborting..."; exit 1
    fi
elif [ "${SLURM_JOB_QOS:3:4}" == "cpu" ]; then
    echo "GPU on a CPU partition?? Aborting..."; exit 1
else
    echo "There is no ${SLURM_JOB_QOS}. Aborting..."; exit 1
fi
LAMMPS_EXE=$(command -v lmp) || ( echo "Executable (lmp) not found. Aborting..."; exit 1 )

# Test if input file is present
if [ ! -f "${LAMMPS_INPUT}".in ]; then echo "No input file found. Aborting..."; exit 1; fi

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}"/JOB-"${SLURM_JOBID}"

# Copy files to the temporary work directory
cp "${LAMMPS_INPUT}".in "${TEMPWORKDIR}" && echo "${LAMMPS_INPUT}.in copied successfully"
cp "${LAMMPS_INPUT}".in "${LAMMPS_INPUT}".in."${SLURM_JOBID}"
for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done
for f in "${DeepMD_MODEL[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
cd "${TEMPWORKDIR}" || exit 1

# Run LAMMPS
echo "# [$(date)] Started"
export EXIT_CODE="0"
export TASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES ))
echo "Running on node(s): ${SLURM_NODELIST}"
echo "Running on ${SLURM_NNODES} node(s)."
echo "Running ${SLURM_NTASKS} task(s), with ${TASKS_PER_NODE} task(s) per node."
echo "Running with ${SLURM_CPUS_PER_TASK} thread(s) per task."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Launch command
SRUN_LAMMPS_EXE="srun --ntasks=${SLURM_NTASKS} --nodes=${SLURM_NNODES} --ntasks-per-node=${TASKS_PER_NODE} --cpus-per-task=${SLURM_CPUS_PER_TASK} ${LAMMPS_EXE}"
LAUNCH_CMD="${SRUN_LAMMPS_EXE} -in ${LAMMPS_INPUT}.in -log ${LAMMPS_INPUT}.log -screen none"

echo "${LAUNCH_CMD}"
${LAUNCH_CMD} || export EXIT_CODE="1"
echo "# [$(date)] Ended"

# Move back data from the temporary work directory and scratch, and clean-up
if [ -f log.cite ]; then rm log.cite ; fi
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || exit 1
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }
rm "${LAMMPS_INPUT}".in."${SLURM_JOBID}"

# Done
echo "Have a nice day !"

# A small pause before SLURM savage clean-up
sleep 5
exit ${EXIT_CODE}
