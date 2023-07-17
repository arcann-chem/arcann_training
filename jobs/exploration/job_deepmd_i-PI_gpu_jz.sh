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

# Input file (extension is automatically added as .in for INPUT)
# Support a list of files as a bash array
DeepMD_MODEL_VERSION="_R_DEEPMD_VERSION_"
DeepMD_MODEL=("_R_MODELS_LIST_")
IPI_INPUT="_R_IPI_INPUT_"
EXTRA_FILES=("_R_XYZ_FILE_" "_R_PLUMED_FILES_LIST_")
NB_CLIENT_PER_GPU=8

#----------------------------------------------
## Nothing needed to be changed past this point

### Project Switch
eval "$(idrenv -d _R_PROJECT_)"

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || exit 1

# Load the environment depending on the version
if [ "${SLURM_JOB_QOS:4:3}" == "gpu" ]; then
    if [ "${DeepMD_MODEL_VERSION}" == "2.1" ]; then
        module purge
        . /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.1.4-cuda11.6_plumed-2.8.0/etc/profile.d/conda.sh
        conda activate /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.1.4-cuda11.6_plumed-2.8.0
        if [[ " ${EXTRA_FILES[*]} " == *"plumed"* ]]; then
            source /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.1.4-cuda11.6_plumed-2.8.0/etc/profile.d/plumed.sh
            export PLUMED_TYPESAFE_IGNORE=yes
        fi
    elif [ "${DeepMD_MODEL_VERSION}" = "2.0" ]; then
        module purge
        . /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.0.3-cuda10.1_plumed-2.7.4/etc/profile.d/conda.sh
        conda activate /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.0.3-cuda10.1_plumed-2.7.4
        if [[ " ${EXTRA_FILES[*]} " == *"plumed"* ]]; then
            source /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.0.3-cuda10.1_plumed-2.7.4/etc/profile.d//plumed.sh
        fi
    else
        echo "DeePMD ${DeepMD_MODEL_VERSION} is not installed on ${SLURM_JOB_QOS}. Aborting..."; exit 1
    fi
elif [ "${SLURM_JOB_QOS:3:4}" == "cpu" ]; then
    echo "GPU on a CPU partition?? Aborting..."; exit 1
else
    echo "There is no ${SLURM_JOB_QOS}. Aborting..."; exit 1
fi
IPI_EXE=$(command -v i-pi) ||  ( echo "Executable (i-pi) not found. Aborting..."; exit 1 )
DP_IPI_EXE=$(command -v dp_ipi) || ( echo "Executable (dp_ipi) not found. Aborting..."; exit 1 )

# Test if input file is present
if [ ! -f "${IPI_INPUT}".xml ]; then echo "No input file found. Aborting..."; exit 1; fi
if [ ! -f "${IPI_INPUT}".json ]; then echo "No input file found. Aborting..."; exit 1; fi

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}"/JOB-"${SLURM_JOBID}"

# Copy files to the temporary work directory
cp "${IPI_INPUT}".xml "${TEMPWORKDIR}" && echo "${IPI_INPUT}.xml copied successfully"
cp "${IPI_INPUT}".xml "${IPI_INPUT}".xml."${SLURM_JOBID}"
cp "${IPI_INPUT}".json "${TEMPWORKDIR}" && echo "${IPI_INPUT}.json copied successfully"

for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done
for f in "${DeepMD_MODEL[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
cd "${TEMPWORKDIR}" || exit 1


# Run i-PI
echo "# [$(date)] Started"
export EXIT_CODE="0"
export TASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES ))
echo "Running on node(s): ${SLURM_NODELIST}"
echo "Running on ${SLURM_NNODES} node(s)."
echo "Running ${SLURM_NTASKS} task(s), with ${TASKS_PER_NODE} task(s) per node."
echo "Running with ${SLURM_CPUS_PER_TASK} thread(s) per task."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CURRENT_HOST=$(hostname)

PORT_OK=0
TEST_COUNT=0
PORT=$(python -c "import random
random.seed(a=${SLURM_JOBID})
for n in range(0,1+${TEST_COUNT}):
    new_port = random.randint(30000,42000)
print(new_port)")
while [ ${PORT_OK} -eq 0 ]; do
    echo "Testing availability of port ${PORT}"
    if netstat -tuln | grep :"${PORT}" ; then
        PORT=$(python -c "import random
random.seed(a=${SLURM_JOBID})
for n in range(0,1+${TEST_COUNT}):
    new_port = random.randint(30000,42000)
print(new_port)")
    TEST_COUNT=$((TEST_COUNT + 1))
    else
        PORT_OK=1
    fi
done
echo "${PORT} will be used as port."

if  [ -f RESTART ]; then
    sed -i "s/address>[^<]*</address>${CURRENT_HOST}</" RESTART
    sed -i "s/port>[^<]*</port>${PORT}</" RESTART
    LAUNCH_CMD="${IPI_EXE} RESTART"
    ${LAUNCH_CMD} &>> ${IPI_INPUT}.i-PI.server.log &
else
    sed -i "s/address>[^<]*</address>${CURRENT_HOST}</" ${IPI_INPUT}.xml
    sed -i "s/port>[^<]*</port>${PORT}</" ${IPI_INPUT}.xml
    LAUNCH_CMD="${IPI_EXE} ${IPI_INPUT}.xml"
    ${LAUNCH_CMD} &> ${IPI_INPUT}.i-PI.server.log &
fi

# Sleep
sleep 40

IFS=',' read -r -a CUDA_ARR <<< "${CUDA_VISIBLE_DEVICES}"
echo "GPU visible: ${CUDA_VISIBLE_DEVICES}"


sed -i "s/_R_ADDRESS_/${CURRENT_HOST}/" ${IPI_INPUT}.json
sed -i "s/\"_R_NB_PORT_\"/${PORT}/" ${IPI_INPUT}.json

# Launch dp_ipi
for j in "${CUDA_ARR[@]}"; do
export CUDA_VISIBLE_DEVICES=${j}
    for ((i=0; i<NB_CLIENT_PER_GPU; i++)); do
        LAUNCH_CMD="${DP_IPI_EXE} ${IPI_INPUT}.json"
        ${LAUNCH_CMD} > /dev/null 2> /dev/null &
        echo "GPU ${CUDA_VISIBLE_DEVICES}, client ${i} launched."
        sleep 2
    done
done
wait
echo "# [$(date)] Ended"

# Move back data from the temporary work directory and scratch, and clean-up
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || exit 1
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }
rm "${IPI_INPUT}".xml."${SLURM_JOBID}"

# Done
echo "Have a nice day !"

# A small pause before SLURM savage clean-up
sleep 5
exit ${EXIT_CODE}
