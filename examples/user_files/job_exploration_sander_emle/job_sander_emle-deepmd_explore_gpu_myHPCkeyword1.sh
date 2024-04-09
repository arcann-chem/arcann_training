#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2024/04/09
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
# You must keep the name file as job_sander_emle-deepmd_explore_ARCHTYPE_myHPCkeyword.sh.
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
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o SANDER-EMLE_DeepMD.%j
#SBATCH -e SANDER-EMLE_DeepMD.%j
# Name of job
#SBATCH -J SANDER-EMLE_DeepMD
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
# Input files (variables) - They should not be changed
#----------------------------------------------
DeepMD_MODEL_VERSION="_R_DEEPMD_VERSION_"
DeepMD_MODEL_FILES=("_R_MODEL_FILES_")
SANDER_IN_FILE="_R_SANDER_IN_FILE_"
EMLE_IN_FILE="_R_EMLE_IN_FILE_"
SANDER_LOG_FILE="_R_SANDER_LOG_FILE_"
SANDER_OUT_FILE="_R_SANDER_OUT_FILE_"
SANDER_RESTART_FILE="_R_SANDER_RESTART_FILE_"
EMLE_OUT_FILE="_R_EMLE_OUT_FILE_"
SANDER_TRAJOUT_FILE="_R_SANDER_TRAJOUT_FILE_"
EXTRA_FILES=("_R_TOP_FILE_" "_R_SANDER_COORD_FILE_" "_R_EMLE_MODEL_FILE_" "_R_PLUMED_FILES_")

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f "${SANDER_IN_FILE}" ] || { echo "${SANDER_IN_FILE} does not exist. Aborting..."; exit 1; }
[ -f "${EMLE_IN_FILE}" ] || { echo "${EMLE_IN_FILE} does not exist. Aborting..."; exit 1; }

# Example to use the DeepMD_MODEL_VERSION variable
if [ "${DeepMD_MODEL_VERSION}" == "2.2" ]; then
    # Load the DeepMD module
    module load DeepMD-kit
elif [ "${DeepMD_MODEL_VERSION}" == "2.1" ]; then
    # Load the DeepMD module
    module "load DeepMD-kit/${DeepMD_MODEL_VERSION}"
else
    echo "DeepMD version ${DeepMD_MODEL_VERSION} is not available. Aborting..."
    exit 1
fi

# Example if your run in a scratch folder
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}"/JOB-"${SLURM_JOBID}"

cp "${SANDER_IN_FILE}" "${TEMPWORKDIR}" && echo "${SANDER_IN_FILE} copied successfully"
cp "${EMLE_IN_FILE}" "${TEMPWORKDIR}" && echo "${EMLE_IN_FILE} copied successfully"
for f in "${DeepMD_MODEL_FILES[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

echo "# [$(date)] Running SANDER-EMLE..."
emle-server --config "${EMLE_IN_FILE}" > "${EMLE_OUT_FILE}" 2>&1 &
wait 5
sander -O -i "${SANDER_IN_FILE}" -o "${SANDER_LOG_FILE}" -p _R_TOP_FILE_ -c _R_SANDER_COORD_FILE_ -r "${SANDER_RESTART_FILE}" -x "${SANDER_TRAJOUT_FILE}" > "${SANDER_OUT_FILE}" 2>&1
echo "# [$(date)] SANDER-EMLE finished."

# Move back data from the temporary work directory and scratch, and clean-up
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

exit