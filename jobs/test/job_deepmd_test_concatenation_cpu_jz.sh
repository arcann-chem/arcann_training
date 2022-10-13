#!/bin/bash
# Author: Rolf DAVID
# Date: 2021/05/20
# Modified: 2022/10/12
# Account
#SBATCH --account=_PROJECT_@_ALLOC_
# Queue
#SBATCH --qos=_QOS_
#SBATCH --partition=_PARTITION_
#SBATCH -C _SUBPARTITION_
# Number of nodes/processes/tasksperprocess
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 10
#SBATCH --cpus-per-task 1
#SBATCH --hint=nomultithread
# Wall-time
#SBATCH -t _WALLTIME_
# Merge Output/Error
#SBATCH -o DeepMD_Test_Concatenation.%j
#SBATCH -e DeepMD_Test_Concatenation.%j
# Name of job
#SBATCH -J DeepMD_Test_Concatenation
# Email (Remove the space between # and SBATCH on the next two lines)
##SBATCH --mail-type FAIL,BEGIN,END,ALL
##SBATCH --mail-user _EMAIL_
#

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