#!/bin/bash
# Author: Rolf DAVID
# Date: 2021/03/18
# Modified: 2022/10/08
# Account
#SBATCH --account=_PROJECT_@_ALLOC_
# Queue
#SBATCH --qos=qos_cpu-t3
# Number of nodes/processes/tasksperprocess
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 10
#SBATCH --cpus-per-task 1
#SBATCH --hint=nomultithread
# Wall-time
#SBATCH -t _WALLTIME_
# Merge Output/Error
#SBATCH -o CP2K.%A_%a
#SBATCH -e CP2K.%A_%a
# Name of job
#SBATCH -J _CP2K_JOBNAME_
# Email (Remove the space between # and SBATCH on the next two lines)
##SBATCH --mail-type FAIL,BEGIN,END,ALL
##SBATCH --mail-user _EMAIL_
#
#SBATCH --array=1-_ARRAYCOUNT_%300
#

echo "${SLURM_ARRAY_TASK_ID}"
SLURM_ARRAY_TASK_ID_PADDED=$(printf "%05d\n" "${SLURM_ARRAY_TASK_ID}")
echo "$SLURM_ARRAY_TASK_ID_PADDED"

# Input file (extension is automatically added as .inp for INPUT, wfn for WFRST, restart for MDRST)
CP2K_INPUT_F1="1_labeling_${SLURM_ARRAY_TASK_ID_PADDED}"
CP2K_INPUT_F2="2_labeling_${SLURM_ARRAY_TASK_ID_PADDED}"
CP2K_WFRST_F="labeling_${SLURM_ARRAY_TASK_ID_PADDED}-SCF"
CP2K_XYZ_F="labeling_${SLURM_ARRAY_TASK_ID_PADDED}.xyz"

#!!Nothing needed to be changed past this point

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}" || exit 1

# Load the environment depending on the version
module purge
module load cp2k/6.1-mpi-popt
CP2K_EXE=$(which cp2k.popt) || ( echo "Executable not found. Aborting..."; exit 1 )

# Test if input file is present
if [ ! -f "${CP2K_INPUT_F1}".inp ]; then echo "No input file found. Aborting..."; exit 1; fi
if [ ! -f "${CP2K_INPUT_F2}".inp ]; then echo "No input file found. Aborting..."; exit 1; fi

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}"/JOB-"${SLURM_JOBID}"

# Copy files to the temporary work directory
cp "${CP2K_INPUT_F1}".inp "${TEMPWORKDIR}" && echo "${CP2K_INPUT_F1}.inp copied successfully"
cp "${CP2K_INPUT_F2}".inp "${TEMPWORKDIR}" && echo "${CP2K_INPUT_F2}.inp copied successfully"
[ -f "${CP2K_XYZ_F}" ] && cp "${CP2K_XYZ_F}" "${TEMPWORKDIR}" && echo "${CP2K_XYZ_F} copied successfully"
[ -f "${CP2K_WFRST_F}".wfn ] && cp "${CP2K_WFRST_F}".wfn "${TEMPWORKDIR}" && echo "${CP2K_WFRST_F}.wfn copied successfully"
cd "${TEMPWORKDIR}" || exit 1

# Run CP2K
echo "# [$(date)] Started"
echo "Running on node(s): ${SLURM_NODELIST}"
echo "Running on ${SLURM_NNODES} node(s)."
export TASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES ))
echo "Running ${SLURM_NTASKS} task(s), with ${TASKS_PER_NODE} task(s) per node."
echo "Running with ${SLURM_CPUS_PER_TASK} thread(s) per task."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
SRUN_CP2K_EXE="srun --ntasks=${SLURM_NTASKS} --nodes=${SLURM_NNODES} --ntasks-per-node=${TASKS_PER_NODE} --cpus-per-task=${SLURM_CPUS_PER_TASK} ${CP2K_EXE}"

# Run CP2K
echo "# [$(date)] Started"
echo "Running on node(s): ${SLURM_NODELIST}"
echo "Running on ${SLURM_NNODES} node(s)."
export TASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES ))
echo "Running ${SLURM_NTASKS} task(s), with ${TASKS_PER_NODE} task(s) per node."
echo "Running with ${SLURM_CPUS_PER_TASK} thread(s) per task."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
SRUN_CP2K_EXE="srun --ntasks=${SLURM_NTASKS} --nodes=${SLURM_NNODES} --ntasks-per-node=${TASKS_PER_NODE} --cpus-per-task=${SLURM_CPUS_PER_TASK} ${CP2K_EXE}"

# Launch command
echo "${SRUN_CP2K_EXE} -i ${CP2K_INPUT_F1}.inp > ${CP2K_INPUT_F1}.out"
export EXIT_CODE="0"
${SRUN_CP2K_EXE} -i "${CP2K_INPUT_F1}".inp > "${CP2K_INPUT_F1}".out || export EXIT_CODE="1"
cp ${CP2K_WFRST_F}.wfn 1_${CP2K_WFRST_F}.wfn
${SRUN_CP2K_EXE} -i "${CP2K_INPUT_F2}".inp > "${CP2K_INPUT_F2}".out || export EXIT_CODE="1"
mv ${CP2K_WFRST_F}.wfn 2_${CP2K_WFRST_F}.wfn

# Move back data from the temporary work directory and scratch, and clean-up
mv ./* "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}"
cd "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}" || exit 1
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

# Done
echo "Have a nice day !"

# A small pause before SLURM savage clean-up
sleep 5
exit ${EXIT_CODE}
