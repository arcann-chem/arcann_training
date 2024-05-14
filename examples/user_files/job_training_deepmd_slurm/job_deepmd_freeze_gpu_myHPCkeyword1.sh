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
# You must keep the name file as job_deepmd_freeze_ARCHTYPE_myHPCkeyword.sh.
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
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o DeepMD_Freeze.%j
#SBATCH -e DeepMD_Freeze.%j
# Name of job
#SBATCH -J DeepMD_Freeze
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

DeepMD_MODEL_VERSION="_R_DEEPMD_VERSION_"
DeepMD_MODEL_FILE="_R_DEEPMD_MODEL_FILE_"
DeepMD_CKPT_FILE="_R_DEEPMD_CKPT_FILE_"
DeepMD_LOG_FILE="_R_DEEPMD_LOG_FILE_"
DeepMD_OUT_FILE="_R_DEEPMD_OUTPUT_FILE_"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f ${DeepMD_CKPT_FILE} ] || { echo "${DeepMD_CKPT_FILE} does not exist. Aborting..."; exit 1; }

# Example to use the DeepMD_MODEL_VERSION variable
if [ ${DeepMD_MODEL_VERSION} == "2.2" ]; then
    # Load the DeepMD module
    module load DeepMD-kit
elif [ ${DeepMD_MODEL_VERSION} == "2.1" ]; then
    # Load the DeepMD module
    module load DeepMD-kit/${DeepMD_MODEL_VERSION}
else
    echo "DeepMD version ${DeepMD_MODEL_VERSION} is not available. Aborting..."
    exit 1
fi

# Run the DeepMD freeze
echo "# [$(date)] Running DeepMD freeze..."
dp freeze -o ${DeepMD_MODEL_FILE} --log-path ${DeepMD_LOG_FILE} > ${DeepMD_OUT_FILE} 2>&1
echo "# [$(date)] DeepMD freeze finished."

sleep 2
exit