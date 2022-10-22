#!/bin/bash
# Author: Rolf DAVID
# Date: 2021/12/08
# Modified: 2022/04/27
# Account
#MSUB -A _R_PROJECT_
# Queue
#MSUB -q _R_ALLOC_
#MSUB -m scratch,work,store
#MSUB -Q normal
# Number of nodes/processes/tasksperprocess
#MSUB -N _R_nb_NODES_
#MSUB -n _R_nb_MPI_
#MSUB -c _R_nb_OPENMP_per_MPI_
# Wall-time
#MSUB -T _R_WALLTIME_
# Merge Output/Error
#MSUB -o CP2K.%A_%a
#MSUB -e CP2K.%A_%a
# Name of job
#MSUB -r _R_CP2K_JOBNAME_
# Email (Remove the space between # and SBATCH on the next two lines)
##MSUB -@ _EMAIL_:begin,end
#
#MSUB -E "--array=_R_ARRAY_START_-_R_ARRAY_END_%250"
#

echo "${SLURM_ARRAY_TASK_ID}"
SLURM_ARRAY_TASK_ID_LARGE=$((SLURM_ARRAY_TASK_ID + _R_NEW_START_))
SLURM_ARRAY_TASK_ID_PADDED=$(printf "%05d\n" "${SLURM_ARRAY_TASK_ID_LARGE}")
echo "$SLURM_ARRAY_TASK_ID_PADDED"

# Input file (extension is automatically added as .inp for INPUT, wfn for WFRST, restart for MDRST)
CP2K_INPUT_F1="1_labeling_${SLURM_ARRAY_TASK_ID_PADDED}"
CP2K_INPUT_F2="2_labeling_${SLURM_ARRAY_TASK_ID_PADDED}"
CP2K_WFRST_F="labeling_${SLURM_ARRAY_TASK_ID_PADDED}-SCF"
CP2K_XYZ_F="labeling_${SLURM_ARRAY_TASK_ID_PADDED}.xyz"

#!!Nothing needed to be changed past this point

# Load the environment depending on the version
module purge
module switch dfldatadir/_R_PROJECT_
module load flavor/buildcompiler/intel/20 flavor/buildmpi/openmpi/4.0 flavor/cp2k/xc
module load cp2k/7.1
CP2K_EXE=$(which cp2k.popt) || ( echo "Executable not found. Aborting..."; exit 1 )

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}" || exit 1

# Test if input file is present
if [ ! -f "${CP2K_INPUT_F1}".inp ]; then echo "No input file found. Aborting..."; exit 1; fi
if [ ! -f "${CP2K_INPUT_F2}".inp ]; then echo "No input file found. Aborting..."; exit 1; fi

# Set the temporary work directory
export TEMPWORKDIR=${CCCSCRATCHDIR}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}"/JOB-"${SLURM_JOBID}"

# Copy files to the temporary work directory
cp "${CP2K_INPUT_F1}".inp "${TEMPWORKDIR}" && echo "${CP2K_INPUT_F1}.inp copied successfully"
cp "${CP2K_INPUT_F2}".inp "${TEMPWORKDIR}" && echo "${CP2K_INPUT_F2}.inp copied successfully"
[ -f "${CP2K_XYZ_F}" ] && cp "${CP2K_XYZ_F}" "${TEMPWORKDIR}" && echo "${CP2K_XYZ_F} copied successfully"
[ -f "${CP2K_WFRST_F}".wfn ] && cp "${CP2K_WFRST_F}".wfn "${TEMPWORKDIR}" && echo "${CP2K_WFRST_F}.wfn copied successfully"
cd "${TEMPWORKDIR}" || exit 1

# Run CP2K
echo "# [$(date)] Started"
echo "Running on node(s): ${SLURM_NODELIST}"
echo "Running on ${SLURM_NNODES} node(s)."
export TASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES ))
echo "Running ${SLURM_NTASKS} task(s), with ${TASKS_PER_NODE} task(s) per node."
echo "Running with ${SLURM_CPUS_PER_TASK} thread(s) per task."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
CCC_MPRUN_CP2K_EXE="ccc_mprun ${CP2K_EXE}"

# Launch command
export EXIT_CODE="0"
${CCC_MPRUN_CP2K_EXE} -i "${CP2K_INPUT_F1}".inp > "${CP2K_INPUT_F1}".out || export EXIT_CODE="1"
cp ${CP2K_WFRST_F}.wfn 1_${CP2K_WFRST_F}.wfn
${CCC_MPRUN_CP2K_EXE} -i "${CP2K_INPUT_F2}".inp > "${CP2K_INPUT_F2}".out || export EXIT_CODE="1"
cp ${CP2K_WFRST_F}.wfn 2_${CP2K_WFRST_F}.wfn
echo "# [$(date)] Ended"

# Move back data from the temporary work directory and scratch, and clean-up
mv ./* "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}"
cd "${SLURM_SUBMIT_DIR}"/"${SLURM_ARRAY_TASK_ID_PADDED}" || exit 1
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

# Done
echo "Have a nice day !"

if [ "${SLURM_ARRAY_TASK_ID}" == "_R_ARRAY_END_" ]; then
    if [ "_R_LAUNCHNEXT_" == "1" ]; then
        cd "_R_CD_WHERE_" || exit 1
        ccc_msub job_labeling_array_cpu_ir__R_NEXT_JOB_FILE_.sh
    fi
fi

# A small pause before SLURM savage clean-up
sleep 5
exit ${EXIT_CODE}
