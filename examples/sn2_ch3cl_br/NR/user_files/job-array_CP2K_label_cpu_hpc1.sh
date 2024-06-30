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
# You must keep the name file as job-array_CP2K_label_ARCHTYPE_myHPCkeyword1.sh.
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
#SBATCH -o CP2K.%A_%a
#SBATCH -e CP2K.%A_%a
# Name of job
#SBATCH -J _R_CP2K_JOBNAME_
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
# Array
#SBATCH --array=_R_ARRAY_START_-_R_ARRAY_END_%250
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------
SLURM_ARRAY_TASK_ID_LARGE=$((SLURM_ARRAY_TASK_ID + _R_NEW_START_))
SLURM_ARRAY_TASK_ID_PADDED=$(printf "%05d\n" "${SLURM_ARRAY_TASK_ID_LARGE}")

CP2K_IN_FILE1="1_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.inp"
CP2K_OUT_FILE1="1_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.out"
CP2K_IN_FILE2="2_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.inp"
CP2K_OUT_FILE2="2_labeling_${SLURM_ARRAY_TASK_ID_PADDED}.out"
CP2K_XYZ_FILE="labeling_${SLURM_ARRAY_TASK_ID_PADDED}.xyz"
CP2K_WFRST_FILE="labeling_${SLURM_ARRAY_TASK_ID_PADDED}-SCF.wfn"

#----------------------------------------------
# Adapt the following lines to your HPC system
# It should be the close to the job_CP2K_label_ARCHTYPE_myHPCkeyword1.sh
# Don't forget to replace the job_labeling_array_ARCHTYPE_myHPCkeyword1.sh at the end of the file (replacling ARCHTYPE and myHPCkeyword1)
#----------------------------------------------


# Project switch
PROJECT_NAME=${SLURM_JOB_ACCOUNT:0:3}
# Compare PROJECT_NAME and IDRPROJ for inequality
if [[ "${PROJECT_NAME}" != "${IDRPROJ}" ]]; then
    SCRATCH=${SCRATCH/${IDRPROJ}/${PROJECT_NAME}}
fi

# Load the environment depending on the version
module purge
module load cp2k/6.1-mpi-popt

if [ "$(command -v cp2k.psmp)" ]; then
    CP2K_EXE=$(command -v cp2k.psmp)
elif [ "$(command -v cp2k.popt)" ]; then
    if [ "${SLURM_CPUS_PER_TASK}" -lt 2 ]; then
        CP2K_EXE=$(command -v cp2k.popt)
    else
        echo "Only executable (cp2k.popt) was found and OpenMP was requested. Aborting..."; exit 1
    fi
else
    echo "Executable (cp2k.popt/cp2k.psmp) not found. Aborting..."; exit 1
fi

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}. Aborting..."; exit 1; }

# Check
[ -f "${CP2K_IN_FILE1}" ] || { echo "${CP2K_IN_FILE1} does not exist. Aborting..."; exit 1; }
[ -f "${CP2K_IN_FILE2}" ] || { echo "${CP2K_IN_FILE2} does not exist. Aborting..."; exit 1; }
[ -f "${CP2K_XYZ_FILE}" ] || { echo "${CP2K_XYZ_FILE} does not exist. Aborting..."; exit 1; }

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}/JOB-${SLURM_JOBID}"

cp "${CP2K_IN_FILE1}" "${TEMPWORKDIR}" && echo "${CP2K_IN_FILE1} copied successfully"
cp "${CP2K_IN_FILE2}" "${TEMPWORKDIR}" && echo "${CP2K_IN_FILE2} copied successfully"
cp "${CP2K_XYZ_FILE}" "${TEMPWORKDIR}" && echo "${CP2K_XYZ_FILE} copied successfully"
[ -f "${CP2K_WFRST_FILE}" ] && cp "${CP2K_WFRST_FILE}" "${TEMPWORKDIR}" && echo "${CP2K_WFRST_FILE} copied successfully"

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "Running ${SLURM_NTASKS} task(s), with ${SLURM_NTASKS_PER_NODE} task(s) per node."
echo "Running with ${SLURM_CPUS_PER_TASK} thread(s) per task."

SRUN_CP2K_EXE="srun --ntasks=${SLURM_NTASKS} --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} --cpus-per-task=${SLURM_CPUS_PER_TASK} ${CP2K_EXE}"

echo "# [$(date)] Running CP2K first job..."
${SRUN_CP2K_EXE} -i "${CP2K_IN_FILE1}" > "${CP2K_OUT_FILE1}"
cp "${CP2K_WFRST_FILE}" "1_${CP2K_WFRST_FILE}"
echo "# [$(date)] CP2K first job finished."
echo "# [$(date)] Running CP2K second job..."
${SRUN_CP2K_EXE} -i "${CP2K_IN_FILE2}" > "${CP2K_OUT_FILE2}"
cp "${CP2K_WFRST_FILE}" "2_${CP2K_WFRST_FILE}"
echo "# [$(date)] CP2K second job finished."

# Move back data from the temporary work directory and scratch, and clean-up
mv ./* "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}"
cd "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

# Logic to launch the next job
if [ "${SLURM_ARRAY_TASK_ID}" == "_R_ARRAY_END_" ]; then
    if [ "_R_LAUNCHNEXT_" == "1" ]; then
        cd "_R_CD_WHERE_" || { echo "Could not go to _R_CD_WHERE_. Aborting..."; exit 1;}
        sbatch job-array_CP2K_label_cpu_jz__R_NEXT_JOB_FILE_.sh
    fi
fi

# Done
echo "Have a nice day !"

sleep 2
exit
