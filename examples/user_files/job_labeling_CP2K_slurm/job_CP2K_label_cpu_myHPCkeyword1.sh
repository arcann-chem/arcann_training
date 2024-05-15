#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2024/05/15
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
# You must keep the name file as job_CP2K_label_ARCHTYPE_myHPCkeyword1.sh.
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
#SBATCH -o CP2K.%j
#SBATCH -e CP2K.%j
# Name of job
#SBATCH -J _R_CP2K_JOBNAME_
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------

CP2K_IN_FILE1="1_labeling__R_PADDEDSTEP_.inp"
CP2K_OUT_FILE1="1_labeling__R_PADDEDSTEP_.out"
CP2K_IN_FILE2="2_labeling__R_PADDEDSTEP_.inp"
CP2K_OUT_FILE2="2_labeling__R_PADDEDSTEP_.out"
CP2K_XYZ_FILE="labeling__R_PADDEDSTEP_.xyz"
CP2K_WFRST_FILE="labeling__R_PADDEDSTEP_-SCF.wfn"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f "${CP2K_IN_FILE1}" ] || { echo "${CP2K_IN_FILE1} does not exist. Aborting..."; exit 1; }
[ -f "${CP2K_IN_FILE2}" ] || { echo "${CP2K_IN_FILE2} does not exist. Aborting..."; exit 1; }
[ -f "${CP2K_XYZ_FILE}" ] || { echo "${CP2K_XYZ_FILE} does not exist. Aborting..."; exit 1; }

# Example if your run in a scratch folder
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/JOB-${SLURM_JOBID}"

cp "${CP2K_IN_FILE1}" "${TEMPWORKDIR}" && echo "${CP2K_IN_FILE1} copied successfully"
cp "${CP2K_IN_FILE2}" "${TEMPWORKDIR}" && echo "${CP2K_IN_FILE2} copied successfully"
cp "${CP2K_XYZ_FILE}" "${TEMPWORKDIR}" && echo "${CP2K_XYZ_FILE} copied successfully"
[ -f "${CP2K_WFRST_FILE}" ] && cp "${CP2K_WFRST_FILE}" "${TEMPWORKDIR}" && echo "${CP2K_WFRST_FILE} copied successfully"

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

echo "# [$(date)] Running CP2K first job..."
cp2k.popt -i "${CP2K_IN_FILE1}" > "${CP2K_OUT_FILE1}"
cp "${CP2K_WFRST_FILE}" "1_${CP2K_WFRST_FILE}"
echo "# [$(date)] CP2K first job finished."
echo "# [$(date)] Running CP2K second job..."
cp2k.popt -i "${CP2K_IN_FILE2}" > "${CP2K_OUT_FILE2}"
cp "${CP2K_WFRST_FILE}" "2_${CP2K_WFRST_FILE}"
echo "# [$(date)] CP2K second job finished."

# Move back data from the temporary work directory and scratch, and clean-up
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

sleep 2
exit