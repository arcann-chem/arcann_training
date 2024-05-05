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
# You must keep the name file as job_i-PI-deepmd_explore_ARCHTYPE_myHPCkeyword.sh.
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
#SBATCH -o i-PI_DeepMD.%j
#SBATCH -e i-PI_DeepMD.%j
# Name of job
#SBATCH -J i-PI_DeepMD
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
# Input files (variables) - They should not be changed
# Except the number of i-PI client per GPU
#----------------------------------------------

DeepMD_MODEL_VERSION="_R_DEEPMD_VERSION_"
DeepMD_MODEL_FILES=("_R_MODEL_FILES_")
IPI_IN_FILE="_R_IPI_IN_FILE_"
DPIPI_IN_FILE="_R_DPIPI_IN_FILE_"
IPI_OUT_FILE="_R_IPI_OUT_FILE_"
EXTRA_FILES=("_R_DATA_FILE_" "_R_PLUMED_FILES_")
NB_CLIENT_PER_GPU=8

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f "${IPI_IN_FILE}" ] || { echo "${IPI_IN_FILE} does not exist. Aborting..."; exit 1; }
[ -f "${DPIPI_IN_FILE}" ] || { echo "${DPIPI_IN_FILE} does not exist. Aborting..."; exit 1; }

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

cp "${IPI_IN_FILE}" "${TEMPWORKDIR}" && echo "${IPI_IN_FILE} copied successfully"
cp "${DPIPI_IN_FILE}" "${TEMPWORKDIR}" && echo "${DPIPI_IN_FILE} copied successfully"
for f in "${DeepMD_MODEL_FILES[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

# This is very i-PI specific
CURRENT_HOST=$(hostname)

PORT_OK=0
SEED=$((RANDOM + SLURM_JOBID))
PORT=$((SEED % 12001 + 30000)) # Random port between 30000 and 42000
while [ ${PORT_OK} -eq 0 ]; do
    echo "Testing availability of port ${PORT}"
    if netstat -tuln | grep -q ":${PORT} " ; then
        wait $((RANDOM % 5 + 1)) # Wait a random time between 1 and 5 seconds
        SEED=$((RANDOM + SLURM_JOBID)) # Generate a new random seed
        PORT=$((RANDOM % 12001 + 30000))  # Generate a new random port
    else
        PORT_OK=1
    fi
done
echo "${PORT} will be used as port."

echo "# [$(date)] Running i-PI..."
# This launch the i-PI server
if  [ -f RESTART ]; then
    sed -i "s/address>[^<]*</address>${CURRENT_HOST}</" RESTART
    sed -i "s/port>[^<]*</port>${PORT}</" RESTART
    i-pi RESTART &>> "${IPI_OUT_FILE}" &
else
    sed -i "s/address>[^<]*</address>${CURRENT_HOST}</" "${IPI_IN_FILE}"
    sed -i "s/port>[^<]*</port>${PORT}</" "${IPI_OUT_FILE}"
    i-pi "${IPI_IN_FILE}" &>> "${IPI_OUT_FILE}" &
fi

# Wait for the server to be ready (40 seconds)
sleep 40

# The clients
# The CUDA_VISIBLE_DEVICES variable is a comma-separated list of GPU IDs
IFS=',' read -r -a CUDA_ARR <<< "${CUDA_VISIBLE_DEVICES}"
echo "GPU visible: ${CUDA_VISIBLE_DEVICES}"

# We need to change the address and port in the JSON file
sed -i "s/_R_ADDRESS_/${CURRENT_HOST}/" "${DPIPI_IN_FILE}"
sed -i "s/\"_R_NB_PORT_\"/${PORT}/" "${DPIPI_IN_FILE}"

# Launch dp_ipi
for j in "${CUDA_ARR[@]}"; do
    export CUDA_VISIBLE_DEVICES=${j}
    for ((i=0; i<NB_CLIENT_PER_GPU; i++)); do
        dp-ipi "${DPIPI_IN_FILE}" > /dev/null 2> /dev/null &
        echo "GPU ${CUDA_VISIBLE_DEVICES}, client ${i} launched."
        sleep 2
    done
done

wait
echo "# [$(date)] i-PI finished."

# Move back data from the temporary work directory and scratch, and clean-up
if [ -f log.cite ]; then rm log.cite ; fi
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

sleep 2
exit