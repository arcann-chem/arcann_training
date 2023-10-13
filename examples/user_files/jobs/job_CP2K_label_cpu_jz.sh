#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Very fancy DeepMD-based semi-automatic highly-customizable iterative training procedure. #
#   Copyright 2022-2023 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2023/09/06
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

# Input file (extension is automatically added as .inp for INPUT, wfn for WFRST, restart for MDRST)
CP2K_INPUT_F1="1_labeling_XXXXX"
CP2K_INPUT_F2="2_labeling_XXXXX"
CP2K_WFRST_F="labeling_XXXXX-SCF"
CP2K_XYZ_F="labeling_XXXXX.xyz"

#----------------------------------------------
## Nothing needed to be changed past this point

# Project Switch and update SCRATCH
PROJECT_NAME=${SLURM_JOB_ACCOUNT:0:3}
eval "$(idrenv -d "${PROJECT_NAME}")"
# Compare PROJECT_NAME and IDRPROJ for inequality
if [[ "${PROJECT_NAME}" != "${IDRPROJ}" ]]; then
    SCRATCH=${SCRATCH/${IDRPROJ}/${PROJECT_NAME}}
fi

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || exit 1

# Load the environment depending on the version
module purge
module load cp2k/6.1-mpi-popt

if [ "$(command -v cp2k.psmp)" ]; then
    CP2K_EXE=$(command -v cp2k.psmp)
elif [ "$(command -v cp2k.popt)" ]; then
    if [ "${SLURM_CPUS_PER_TASK}" -lt 2 ]; then
        CP2K_EXE=$(command -v cp2k.popt)
    else
        echo "Only executable (cp2k.popt) was found and OpenMP was requested. Aborting..."
    fi
else
    echo "Executable (cp2k.popt/cp2k.psmp) not found. Aborting..."
fi

# Test if input file is present
if [ ! -f "${CP2K_INPUT_F1}".inp ]; then echo "No input file found. Aborting..."; exit 1; fi
if [ ! -f "${CP2K_INPUT_F2}".inp ]; then echo "No input file found. Aborting..."; exit 1; fi

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}"/JOB-"${SLURM_JOBID}"

# Copy files to the temporary work directory
cp "${CP2K_INPUT_F1}".inp "${TEMPWORKDIR}" && echo "${CP2K_INPUT_F1}.inp copied successfully"
cp "${CP2K_INPUT_F2}".inp "${TEMPWORKDIR}" && echo "${CP2K_INPUT_F2}.inp copied successfully"
[ -f "${CP2K_XYZ_F}" ] && cp "${CP2K_XYZ_F}" "${TEMPWORKDIR}" && echo "${CP2K_XYZ_F} copied successfully"
[ -f "${CP2K_WFRST_F}".wfn ] && cp "${CP2K_WFRST_F}".wfn "${TEMPWORKDIR}" && echo "${CP2K_WFRST_F}.wfn copied successfully"
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

SRUN_CP2K_EXE="srun --ntasks=${SLURM_NTASKS} --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} --cpus-per-task=${SLURM_CPUS_PER_TASK} ${CP2K_EXE}"

# Launch command
export EXIT_CODE="0"
${SRUN_CP2K_EXE} -i "${CP2K_INPUT_F1}".inp > "${CP2K_INPUT_F1}".out || export EXIT_CODE="1"
cp "${CP2K_WFRST_F}.wfn" "1_${CP2K_WFRST_F}.wfn"
${SRUN_CP2K_EXE} -i "${CP2K_INPUT_F2}".inp > "${CP2K_INPUT_F2}".out || export EXIT_CODE="1"
cp "${CP2K_WFRST_F}.wfn" "2_${CP2K_WFRST_F}.wfn"
echo "# [$(date)] Ended"

# Move back data from the temporary work directory and scratch, and clean-up
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || exit 1
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

# Done
echo "Have a nice day !"

# A small pause before SLURM savage clean-up
sleep 5
exit ${EXIT_CODE}
