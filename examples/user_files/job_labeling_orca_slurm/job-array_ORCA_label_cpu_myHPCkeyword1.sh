#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Very fancy DeepMD-based semi-automatic highly-customizable iterative training procedure. #
#   Copyright 2022-2023 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2024/04/09
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
# You must keep the name file as job-array_ORCA_label_ARCHTYPE_myHPCkeyword1.sh.
#----------------------------------------------
# Project/Account
#SBATCH --account=_R_PROJECT_@_R_ALLOC_
# QoS/Partition/SubPartition
#SBATCH --qos=_R_QOS_
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes _R_nb_NODES_
#SBATCH --ntasks-per-node _R_nb_MPIPERNODE_
#SBATCH --cpus-per-task _R_nb_THREADSPERMPI_
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o ORCA.%A_%a
#SBATCH -e ORCA.%A_%a
# Name of job
#SBATCH -J _R_ORCA_JOBNAME_
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
# Array
#SBATCH --array=_R_ARRAY_START_-_R_ARRAY_END_%300
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------
SLURM_ARRAY_TASK_ID_LARGE=$((SLURM_ARRAY_TASK_ID + _R_NEW_START_))
SLURM_ARRAY_TASK_ID_PADDED=$(printf "%05d\n" "${SLURM_ARRAY_TASK_ID_LARGE}")

ORCA_IN_FILE1="1_labeling__R_PADDEDSTEP_.inp"
ORCA_OUT_FILE1="1_labeling__R_PADDEDSTEP_.out"
ORCA_XYZ_FILE="labeling__R_PADDEDSTEP_.xyz"

#----------------------------------------------
# Adapt the following lines to your HPC system
# It should be the close to the job_ORCA_label_ARCHTYPE_myHPCkeyword1.sh
# Don't forget to replace the job_labeling_array_ARCHTYPE_myHPCkeyword1.sh at the end of the file (replacling ARCHTYPE and myHPCkeyword1)
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}. Aborting..."; exit 1; }

# Check
[ -f "${ORCA_IN_FILE1}" ] || { echo "${ORCA_IN_FILE1} does not exist. Aborting..."; exit 1; }
[ -f "${ORCA_XYZ_FILE}" ] || { echo "${ORCA_XYZ_FILE} does not exist. Aborting..."; exit 1; }

# Example if your run in a scratch folder
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}/JOB-${SLURM_JOBID}"

cp "${ORCA_IN_FILE1}" "${TEMPWORKDIR}" && echo "${ORCA_IN_FILE1} copied successfully"
cp "${ORCA_XYZ_FILE}" "${TEMPWORKDIR}" && echo "${ORCA_XYZ_FILE} copied successfully"

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

echo "# [$(date)] Running ORCA..."
orca "${ORCA_IN_FILE1}" > "${ORCA_OUT_FILE1}" 2>&1
echo "# [$(date)] ORCA job finished."

# Move back data from the temporary work directory and scratch, and clean-up
mv ./* "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}"
cd "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

# Logic to launch the next job
if [ "${SLURM_ARRAY_TASK_ID}" == "_R_ARRAY_END_" ]; then
    if [ "_R_LAUNCHNEXT_" == "1" ]; then
        cd "_R_CD_WHERE_" || { echo "Could not go to _R_CD_WHERE_. Aborting..."; exit 1;}
        sbatch job_labeling_array_ARCHTYPE_myHPCkeyword1__R_NEXT_JOB_FILE_.sh
    fi
fi
sleep 2
exit












# Array specifics
SLURM_ARRAY_TASK_ID_LARGE=$((SLURM_ARRAY_TASK_ID + _R_NEW_START_))
SLURM_ARRAY_TASK_ID_PADDED=$(printf "%05d\n" "${SLURM_ARRAY_TASK_ID_LARGE}")

# Input file (extension is automatically added as .inp for INPUT, wfn for WFRST, restart for MDRST)
ORCA_INPUT_F="1_labeling_${SLURM_ARRAY_TASK_ID_PADDED}"
ORCA_XYZ_F="labeling_${SLURM_ARRAY_TASK_ID_PADDED}.xyz"

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
cd "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}" || exit 1

# Load the environment depending on the version
module purge
module load orca/5.0.3-mpi

if [ "$(command -v orca)" ]; then
    ORCA_EXE=$(command -v orca)
else
    echo "Executable orca not found. Aborting..."
fi

# Test if input file is present
if [ ! -f "${ORCA_INPUT_F}".inp ]; then echo "No input file found. Aborting..."; exit 1; fi

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}"/JOB-"${SLURM_JOBID}"

# Copy files to the temporary work directory
cp "${ORCA_INPUT_F}".inp "${TEMPWORKDIR}" && echo "${ORCA_INPUT_F}.inp copied successfully"
[ -f "${ORCA_XYZ_F}" ] && cp "${ORCA_XYZ_F}" "${TEMPWORKDIR}" && echo "${ORCA_XYZ_F} copied successfully"
cd "${TEMPWORKDIR}" || exit 1

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
export EXIT_CODE="0"
${ORCA_EXE} "${ORCA_INPUT_F}".inp > "${ORCA_INPUT_F}".out || export EXIT_CODE="1"
echo "# [$(date)] Ended"

# Move back data from the temporary work directory and scratch, and clean-up
mv ./* "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}"
cd "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}" || exit 1
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

# Done
echo "Have a nice day !"

# Logic to launch the next job
if [ "${SLURM_ARRAY_TASK_ID}" == "_R_ARRAY_END_" ]; then
    if [ "_R_LAUNCHNEXT_" == "1" ]; then
        cd "_R_CD_WHERE_" || exit 1
        sbatch job_labeling_array_cpu_jz__R_NEXT_JOB_FILE_.sh
    fi
fi

# A small pause before SLURM savage clean-up
sleep 5
exit ${EXIT_CODE}
