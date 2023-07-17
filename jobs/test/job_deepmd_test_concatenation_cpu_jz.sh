#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Very fancy DeepMD-based semi-automatic highly-customizable iterative training procedure. #
#   Copyright 2022-2023 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
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
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o DeepMD_Test_Concatenation.%j
#SBATCH -e DeepMD_Test_Concatenation.%j
# Name of job
#SBATCH -J DeepMD_Test_Concatenation
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
## Nothing needed to be changed past this point

### Project Switch
eval "$(idrenv -d _R_PROJECT_)"

module purge
module load anaconda-py3/2022.05

cd "${SLURM_SUBMIT_DIR}" || exit 1

echo "# [$(date)] Started"
python _deepmd_test_concatenation.py
echo "# [$(date)] Ended"

# Done
echo "Have a nice day !"

# A small pause before SLURM savage clean-up
sleep 5
