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
#SBATCH --array=_R_ARRAY_START_-_R_ARRAY_END_%300
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

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}. Aborting..."; exit 1; }

# Check
[ -f "${CP2K_IN_FILE1}" ] || { echo "${CP2K_IN_FILE1} does not exist. Aborting..."; exit 1; }
[ -f "${CP2K_IN_FILE2}" ] || { echo "${CP2K_IN_FILE2} does not exist. Aborting..."; exit 1; }
[ -f "${CP2K_XYZ_FILE}" ] || { echo "${CP2K_XYZ_FILE} does not exist. Aborting..."; exit 1; }

# Example if your run in a scratch folder
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID_PADDED}/JOB-${SLURM_JOBID}"

cp "${CP2K_IN_FILE1}" "${TEMPWORKDIR}" && echo "${CP2K_IN_FILE1} copied successfully"
cp "${CP2K_IN_FILE2}" "${TEMPWORKDIR}" && echo "${CP2K_IN_FILE2} copied successfully"
cp "${CP2K_XYZ_FILE}" "${TEMPWORKDIR}" && echo "${CP2K_XYZ_FILE} copied successfully"
[ -f "${CP2K_WFRST_FILE}" ] && cp "${CP2K_WFRST_FILE}" "${TEMPWORKDIR}" && echo "${CP2K_WFRST_FILE} copied successfully"

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

echo "# [$(date)] Running CP2K first job..."
cp2k.popt -i "${CP2K_IN_FILE1}" > "${CP2K_OUT_FILE1}" 2>&1
cp "${CP2K_WFRST_FILE}" "1_${CP2K_WFRST_FILE}"
echo "# [$(date)] CP2K first job finished."
echo "# [$(date)] Running CP2K second job..."
cp2k.popt -i "${CP2K_IN_FILE2}" > "${CP2K_OUT_FILE2}" 2>&1
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
        sbatch job_labeling_array_ARCHTYPE_myHPCkeyword1__R_NEXT_JOB_FILE_.sh
    fi
fi
exit