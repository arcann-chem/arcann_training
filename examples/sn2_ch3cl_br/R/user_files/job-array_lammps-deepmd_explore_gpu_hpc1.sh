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
# You must keep the name file as job-array_lammps-deepmd_explore_ARCHTYPE_myHPCkeyword.sh.
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
# This part use the job-array-params_lammps-deepmd_explore_ARCHTYPE_myHPCkeyword.lst created
# Don't forget to replace the array_line with the correct name of the file (namely ARCHTYPE and myHPCkeyword)
# The rest should not be changed
#----------------------------------------------


SLURM_ARRAY_TASK_ID_LINE=$((SLURM_ARRAY_TASK_ID + 2))
array_line=$(sed -n "${SLURM_ARRAY_TASK_ID_LINE}p" "job-array-params_lammps-deepmd_explore_gpu_jz.lst")
IFS='/' read -ra array_param <<< "${array_line}"

JOB_PATH=${array_param[0]}
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"

DeepMD_MODEL_VERSION=${array_param[1]}
IFS='" "' read -r -a DeepMD_MODEL_FILES <<< "${array_param[2]}"
LAMMPS_IN_FILE=${array_param[3]}
LAMMPS_LOG_FILE=${LAMMPS_IN_FILE/.in/.log}
LAMMPS_OUT_FILE=${LAMMPS_IN_FILE/.in/.out}
EXTRA_FILES=()
EXTRA_FILES+=("${array_param[4]}")
if [ -n "${array_param[5]}" ]; then
    EXTRA_FILES+=("${array_param[5]}")
fi
if [ -n "${array_param[6]}" ]; then
    IFS='" "' read -r -a PLUMED_FILES <<< "${array_param[6]}"
    EXTRA_FILES+=("${PLUMED_FILES[@]}")
fi

#----------------------------------------------
# Adapt the following lines to your HPC system
# It should be the close to the job_lammps-deepmd_explore_ARCHTYPE_myHPCkeyword.sh
#----------------------------------------------


# Project switch
PROJECT_NAME=${SLURM_JOB_ACCOUNT:0:3}
# Compare PROJECT_NAME and IDRPROJ for inequality
if [[ "${PROJECT_NAME}" != "${IDRPROJ}" ]]; then
    SCRATCH=${SCRATCH/${IDRPROJ}/${PROJECT_NAME}}
fi

# Get the available projects
readarray -t AVAILABLE_PROJECTS < <(idr_compuse | awk '/@/{print $1}' | cut -d '@' -f 1 | sort -u)

IS_PRJ2_AVAILABLE=0
IS_PRJ1_AVAILABLE=0
for PROJECT in "${AVAILABLE_PROJECTS[@]}"; do
    if [[ "$PROJECT" == "myproject2" ]]; then
        IS_PRJ2_AVAILABLE=1
    elif [[ "$PROJECT" == "myproject1" ]]; then
        IS_PRJ1_AVAILABLE=1
    fi
done

# Get the version and the environment path
if [[ "${DeepMD_MODEL_VERSION}" == "2.2" ]]; then
    if [[ "${IS_PRJ2_AVAILABLE}" == 1 || "${PROJECT_NAME}" == "myproject2" ]]; then
        DEEPMD_CONDA_ENV_PATH=/programs/apps/deepmd-kit/2.2.7-cuda11.6_plumed-2.8.3
    elif [[ "${IS_PRJ1_AVAILABLE}" == 1 || "${PROJECT_NAME}" == "myproject1" ]]; then
        DEEPMD_CONDA_ENV_PATH=/apps/deepmd-kit/2.2.7-cuda11.6_plumed-2.8.0
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not accessible for ${USER}. Aborting..."; exit 1
    fi
elif [[ "${DeepMD_MODEL_VERSION}" == "2.1" ]]; then
    if [[ "${IS_PRJ2_AVAILABLE}" == 1 || "${PROJECT_NAME}" == "myproject2" ]]; then
        DEEPMD_CONDA_ENV_PATH=/programs/apps/deepmd-kit/2.1.5-cuda11.6_plumed-2.8.1
    elif [[ "${IS_PRJ1_AVAILABLE}" == 1 || "${PROJECT_NAME}" == "myproject1" ]]; then
        DEEPMD_CONDA_ENV_PATH=/apps/deepmd-kit/2.1.4-cuda11.6_plumed-2.8.0
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not accessible for ${USER}. Aborting..."; exit 1
    fi
elif [[ "${DeepMD_MODEL_VERSION}" == "2.0" ]]; then
    if [[ "${IS_PRJ1_AVAILABLE}" == 1 || "${PROJECT_NAME}" == "myproject1" ]]; then
        DEEPMD_CONDA_ENV_PATH=/apps/deepmd-kit/2.0.3-cuda10.1_plumed-2.7.4
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not accessible for ${USER}. Aborting..."; exit 1
    fi
else
    echo "DeePMD ${DeepMD_MODEL_VERSION} is not installed on ${SLURM_JOB_QOS}. Aborting..."; exit 1
fi

# Load the environment
module purge
. "${DEEPMD_CONDA_ENV_PATH}/etc/profile.d/conda.sh"
conda activate ${DEEPMD_CONDA_ENV_PATH}
LAMMPS_EXE=$(command -v lmp) || { echo "Executable (lmp) not found. Aborting..."; exit 1 ; }

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${JOB_PATH}. Aborting..."; exit 1; }

# Check
[ -f "${LAMMPS_IN_FILE}" ] || { echo "${LAMMPS_IN_FILE} does not exist. Aborting..."; exit 1; }

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/${JOB_PATH}/JOB-${SLURM_JOBID}"

cp "${LAMMPS_IN_FILE}" "${TEMPWORKDIR}" && echo "${LAMMPS_IN_FILE} copied successfully"
for f in "${DeepMD_MODEL_FILES[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done

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

SRUN_LAMMPS_EXE="srun --ntasks=${SLURM_NTASKS} --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} --cpus-per-task=${SLURM_CPUS_PER_TASK} ${LAMMPS_EXE}"

# Run the DeepMD train
echo "# [$(date)] Running LAMMPS..."
${SRUN_LAMMPS_EXE} -in "${LAMMPS_IN_FILE}" -log "${LAMMPS_LOG_FILE}" -screen none > "${LAMMPS_OUT_FILE}" 2>&1
echo "# [$(date)] LAMMPS finished."

# Move back data from the temporary work directory and scratch, and clean-up
if [ -f log.cite ]; then rm log.cite ; fi
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}/${JOB_PATH}"
cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${JOB_PATH}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

# Done
echo "Have a nice day !"

sleep 2
exit