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
#SBATCH -o LAMMPS_DeepMD.%A_%a
#SBATCH -e LAMMPS_DeepMD.%A_%a
# Name of job
#SBATCH -J LAMMPS_DeepMD
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#
#SBATCH --array=_R_ARRAY_START_-_R_ARRAY_END_%300
#

#----------------------------------------------
# This part use the job-array-params_sander_emle-deepmd_explore_ARCHTYPE_myHPCkeyword.lst created
# Don't forget to replace the array_line with the correct name of the file (namely ARCHTYPE and myHPCkeyword)
# The rest should not be changed
#----------------------------------------------
SLURM_ARRAY_TASK_ID_LINE=$((SLURM_ARRAY_TASK_ID + 2))

array_line=$(sed -n "${SLURM_ARRAY_TASK_ID_LINE}p" "job-array-params_sander_emle-deepmd_explore_ARCHTYPE_myHPCkeyword.lst")
IFS='/' read -ra array_param <<< "${array_line}"

JOB_PATH=${array_param[0]}
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"

DeepMD_MODEL_VERSION=${array_param[1]}
IFS='" "' read -r -a DeepMD_MODEL_FILES <<< "${array_param[2]}"
SANDER_IN_FILE=${array_param[3]}
EMLE_IN_FILE=${array_param[6]}
SANDER_LOG_FILE=${SANDER_IN_FILE/.in/.log}
SANDER_OUT_FILE=${SANDER_IN_FILE/.in/.out}
SANDER_RESTART_FILE=${SANDER_IN_FILE/.in/.ncrst}
EMLE_OUT_FILE=${EMLE_IN_FILE/.yaml/_emle.out}
SANDER_TRAJOUT_FILE=${SANDER_IN_FILE/.in/.nc}
SANDER_PRMTOP_FILE=${array_param[4]}
SANDER_COORD_FILE=${array_param[5]}
EXTRA_FILES=()
EXTRA_FILES+=("${array_param[4]}")
EXTRA_FILES+=("${array_param[5]}")
EXTRA_FILES+=("${array_param[7]}")
if [ -n "${array_param[8]}" ]; then
    IFS='" "' read -r -a PLUMED_FILES <<< "${array_param[6]}"

    EXTRA_FILES+=("${PLUMED_FILES[@]}")
fi

#----------------------------------------------
# Adapt the following lines to your HPC system
# It should be the close to the job_sander_emle-deepmd_explore_ARCHTYPE_myHPCkeyword.sh
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f "${SANDER_IN_FILE}" ] || { echo "${SANDER_IN_FILE} does not exist. Aborting..."; exit 1; }
[ -f "${EMLE_IN_FILE}" ] || { echo "${EMLE_IN_FILE} does not exist. Aborting..."; exit 1; }

# Example to use the DeepMD_MODEL_VERSION variable
if [ "${DeepMD_MODEL_VERSION}" == "2.2" ]; then
    # Load the DeepMD module
    module load DeepMD-kit
elif [ "${DeepMD_MODEL_VERSION}" == "2.1" ]; then
    # Load the DeepMD module
    module load "DeepMD-kit/${DeepMD_MODEL_VERSION}"
else
    echo "DeepMD version ${DeepMD_MODEL_VERSION} is not available. Aborting..."
    exit 1
fi

# Example if your run in a scratch folder
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/${JOB_PATH}/JOB-${SLURM_JOBID}"

cp "${SANDER_IN_FILE}" "${TEMPWORKDIR}" && echo "${SANDER_IN_FILE} copied successfully"
cp "${EMLE_IN_FILE}" "${TEMPWORKDIR}" && echo "${EMLE_IN_FILE} copied successfully"
for f in "${DeepMD_MODEL_FILES[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

echo "# [$(date)] Running SANDER-EMLE..."
emle-server --config "${EMLE_IN_FILE}" > "${EMLE_OUT_FILE}" 2>&1 &
wait 5
sander -O -i "${SANDER_IN_FILE}" -o "${SANDER_LOG_FILE}" -p "${SANDER_PRMTOP_FILE}" -c "${SANDER_COORD_FILE}" -r "${SANDER_RESTART_FILE}" -x "${SANDER_TRAJOUT_FILE}" > "${SANDER_OUT_FILE}" 2>&1
echo "# [$(date)] SANDER-EMLE finished."

# Move back data from the temporary work directory and scratch, and clean-up
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}/${JOB_PATH}"
cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${JOB_PATH}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

sleep 2
exit