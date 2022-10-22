## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name: str = "nvs"
# allocation_name: str = "v100"
# arch_name: str = "v100"
# slurm_email: str = ""
## These are the default
# temperature_K: list = [298.15, 298.15]
# timestep_ps: list = [0.0005, 0.0005]
## print_freq is every 1% / nb_steps_exploration is initial/auto-calculated (local subsys)
## These are the default
# nb_steps_exploration: list = [20000, 20000]
# print_freq: list = [200, 200]
# nb_steps_initial: list = [20000, 20000]
# nb_traj: int = 2
# disturbed_start: bool = [False, False]

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import subprocess
import numpy as np
import random

training_iterative_apath = Path("..").resolve()
### Check if the deepmd_iterative_apath is defined
deepmd_iterative_apath_error = 1
if "deepmd_iterative_apath" in globals():
    if (Path(deepmd_iterative_apath)/"scripts"/"common_functions.py").is_file():
        deepmd_iterative_apath_error = 0
elif (Path().home()/"deepmd_iterative_py"/"scripts"/"common_functions.py").is_file():
    deepmd_iterative_apath = Path().home()/"deepmd_iterative_py"
    deepmd_iterative_apath_error = 0
elif (training_iterative_apath/"control"/"path").is_file():
    deepmd_iterative_apath = Path((training_iterative_apath/"control"/"path").read_text())
    if (deepmd_iterative_apath/"scripts"/"common_functions.py").is_file():
        deepmd_iterative_apath_error = 0
if deepmd_iterative_apath_error == 1:
    logging.critical("Can\"t find common_functions.py in usual places:")
    logging.critical("deepmd_iterative_apath variable or ~/deepmd_iterative_py or in the path file in control")
    logging.critical("Aborting...")
    sys.exit(1)
sys.path.insert(0, str(Path(deepmd_iterative_apath)/"scripts"))
del deepmd_iterative_apath_error
import common_functions as cf

control_apath = training_iterative_apath/"control"
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
jobs_apath = deepmd_iterative_apath/"jobs"/"exploration"
exploration_json = cf.json_read((control_apath/("exploration_"+current_iteration_zfill+".json")),False,True)
previous_iteration = current_iteration-1
previous_iteration_zfill = str(previous_iteration).zfill(3)
prevtraining_json = cf.json_read((control_apath/("training_"+previous_iteration_zfill+".json")),True,True)
if previous_iteration > 0:
    prevexploration_json = cf.json_read((control_apath/("exploration_"+previous_iteration_zfill+".json")),True,True)

exploration_json["deepmd_model_version"] = prevtraining_json["deepmd_model_version"]
exploration_json["nb_nnp"] = config_json["nb_nnp"]
exploration_json["nb_traj"] = 2 if "nb_traj" not in globals() else nb_traj

### Checks
if not prevtraining_json["is_frozen"]:
    logging.critical("Lock found. Previous NNPs aren\"t frozen")
    logging.critical("Aborting...")
    sys.exit(1)

### #35
cluster = cf.check_cluster()
cluster = "jz"

### Set needed variables
exploration_json["cluster"] = cluster
exploration_json["project_name"] = "nvs" if "project_name" not in globals() else project_name
exploration_json["allocation_name"] = "v100" if "allocation_name" not in globals() else allocation_name
exploration_json["arch_name"] = "v100" if "arch_name" not in globals() else arch_name

project_name = exploration_json["project_name"]
allocation_name = exploration_json["allocation_name"]
arch_name = exploration_json["arch_name"]

if arch_name == "v100" or arch_name == "a100":
    arch_type ="gpu"

### Checks
slurm_email = "" if "slurm_email" not in globals() else slurm_email

### Preparation of the exploration
exploration_json["subsys_nr"]={}

for it0_subsys_nr,it_subsys_nr in enumerate(config_json["subsys_nr"]):
    random.seed()
    exploration_json["subsys_nr"][it_subsys_nr]={}

    it_exploration_type = config_json["subsys_nr"][it_subsys_nr]["exploration_type"]
    it_temperature = config_json["subsys_nr"][it_subsys_nr]["temperature_K"] if "temperature_K" not in globals() else temperature_K[it0_subsys_nr]
    it_timestep = config_json["subsys_nr"][it_subsys_nr]["timestep_ps"] if "timestep_ps" not in globals() else timestep_ps[it0_subsys_nr]

    exploration_json["subsys_nr"][it_subsys_nr]["exploration_type"] = it_exploration_type
    exploration_json["subsys_nr"][it_subsys_nr]["temperature_K"] = it_temperature
    exploration_json["subsys_nr"][it_subsys_nr]["timestep_ps"] = it_timestep

    if it_exploration_type == "lammps":

        it_exploration_input = cf.read_file(training_iterative_apath/"inputs"/(it_subsys_nr+".in"))
        it_exploration_input = cf.replace_in_list(it_exploration_input,"_R_TEMPERATURE_",str(it_temperature))
        it_exploration_input = cf.replace_in_list(it_exploration_input,"_R_TIMESTEP_",str(it_timestep))

        if current_iteration == 1:
            it_nb_steps = 20000 if "nb_steps_initial" not in globals() else nb_steps_initial[it0_subsys_nr]
            it_exploration_input = cf.replace_in_list(it_exploration_input,"_R_NUMBER_OF_STEPS_",str(it_nb_steps))
            it_lammps_data_filename = it_subsys_nr+".lmp"
            it_exploration_input = cf.replace_in_list(it_exploration_input,"_R_DATA_FILE_",it_lammps_data_filename)
            it_lammps_data = cf.read_file(training_iterative_apath/"inputs"/it_lammps_data_filename)
            it_approx_time = 10

            ## Get cell and number of atoms
            dim_string = ["xlo xhi", "ylo yhi", "zlo zhi"]
            it_cell=[]
            for n,string in enumerate(dim_string):
                temp = [zzz for zzz in it_lammps_data if string in zzz]
                temp = cf.replace_in_list(temp,"\n","")
                temp = [zzz for zzz in temp[0].split(" ") if zzz]
                it_cell.append(float(temp[1]) - float(temp[0]))
            del n, string
            temp = [zzz for zzz in it_lammps_data if "atoms" in zzz]
            temp = cf.replace_in_list(temp,"\n","")
            temp = [zzz for zzz in temp[0].split(" ") if zzz]
            it_nb_atm = int(temp[0])
            del temp, dim_string

    # Sarting structures (after first iteration)
    if current_iteration > 1:
        starting_point_list_path = [zzz for zzz in (training_iterative_apath/"starting_structures").glob(previous_iteration_zfill+"_"+it_subsys_nr+"_*.lmp")]
        starting_point_list_all = [str(zzz).split("/")[-1] for zzz in starting_point_list_path]
        starting_point_list = [zzz for zzz in starting_point_list_all if "disturbed" not in zzz]
        starting_point_list_disturbed = [zzz for zzz in starting_point_list_all if zzz not in starting_point_list]
        starting_point_list_disturbed_bckp = starting_point_list_disturbed.copy()
        starting_point_list_bckp = starting_point_list.copy()

        if "disturbed_start" not in globals():
            if (
                current_iteration > 1
                and prevexploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"]
                and prevexploration_json["subsys_nr"][it_subsys_nr]["disturbed_min"]
                ):
                starting_point_list = starting_point_list_disturbed.copy()
                starting_point_list_bckp = starting_point_list_disturbed_bckp.copy()
                exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"] = True
            else:
                exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"] = False
        elif disturbed_start[it0_subsys_nr]:
            starting_point_list = starting_point_list_disturbed.copy()
            starting_point_list_bckp = starting_point_list_disturbed_bckp.copy()
            exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"] = disturbed_start[it0_subsys_nr]
        else:
            exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"] = False
        del starting_point_list_path, starting_point_list_all

    ### #35
    cf.check_file(jobs_apath/("job_deepmd_"+it_exploration_type+"_"+arch_type +"_"+cluster+".sh"),True,True,"No SLURM file present for the exploration step on this cluster.")
    slurm_file_master = cf.read_file(jobs_apath/("job_deepmd_"+it_exploration_type+"_"+arch_type +"_"+cluster+".sh"))

    for it_nnp in range(1, config_json["nb_nnp"] + 1 ):
        for it_number in range(1, exploration_json["nb_traj"] + 1):

            local_apath = Path(".").resolve()/str(it_subsys_nr)/str(it_nnp)/(str(it_number).zfill(5))
            local_apath.mkdir(exist_ok=True,parents=True)

            ### Get the model list
            list_nnp = [zzz for zzz in range(1, config_json["nb_nnp"] + 1)]
            reorder_nnp_list = list_nnp[list_nnp.index(it_nnp):] + list_nnp[:list_nnp.index(it_nnp)]
            compress_str = "_compressed" if prevtraining_json["is_compressed"] else ""
            models_list=["graph_"+str(f)+"_"+previous_iteration_zfill+compress_str+".pb" for f in reorder_nnp_list]
            for it_sub_nnp in range(1, config_json["nb_nnp"] + 1 ):
                nnp_apath = (training_iterative_apath/"NNP"/("graph_"+str(it_sub_nnp)+"_"+previous_iteration_zfill+compress_str+".pb")).resolve()
                subprocess.call(["ln","-s", str(nnp_apath), str(local_apath)])
            models_list=" ".join(models_list)
            del list_nnp, it_sub_nnp, nnp_apath, compress_str, reorder_nnp_list

            ### LAMMPS
            if it_exploration_type == "lammps":
                exploration_input = it_exploration_input.copy()
                RAND = random.randrange(0,1000)
                exploration_input = cf.replace_in_list(exploration_input,"_R_SEED_VEL_",str(it_nnp)+str(RAND)+str(it_number)+previous_iteration_zfill)
                RAND = random.randrange(0,1000)
                exploration_input = cf.replace_in_list(exploration_input,"_R_SEED_THER_",str(it_nnp)+str(RAND)+str(it_number)+previous_iteration_zfill)
                exploration_input = cf.replace_in_list(exploration_input,"_R_DCD_OUT_",str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".dcd")
                exploration_input = cf.replace_in_list(exploration_input,"_R_RESTART_OUT_",str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".restart")
                exploration_input = cf.replace_in_list(exploration_input,"_R_MODELS_LIST_",models_list)
                exploration_input = cf.replace_in_list(exploration_input,"_R_DEVI_OUT_","model_devi_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".out")
                del RAND

                ### Get data files (starting points) and number of steps
                if current_iteration > 1:
                    if len(starting_point_list) == 0:
                        starting_point_list = starting_point_list_bckp.copy()
                    RAND = random.randrange(0,len(starting_point_list))
                    it_lammps_data_filename = starting_point_list[RAND]
                    it_lammps_data = cf.read_file(training_iterative_apath/"starting_structures"/it_lammps_data_filename)
                    exploration_input = cf.replace_in_list(exploration_input,"_R_DATA_FILE_",it_lammps_data_filename)

                    ratio_ill_described = (
                        (prevexploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"] + prevexploration_json["subsys_nr"][it_subsys_nr]["nb_rejected"])
                        / prevexploration_json["subsys_nr"][it_subsys_nr]["nb_total"]
                        )
                    it_nb_steps = prevexploration_json["subsys_nr"][it_subsys_nr]["nb_steps"]
                    if ( ratio_ill_described ) < 0.10:
                        it_nb_steps = it_nb_steps * 4
                    elif ( ratio_ill_described ) < 0.20:
                        it_nb_steps = it_nb_steps * 2

                    if it_nb_steps > 400/it_timestep:
                        it_nb_steps = int(400/it_nb_steps)

                    it_nb_steps = it_nb_steps if "nb_steps_exploration" not in globals() else nb_steps_exploration[it0_subsys_nr]
                    exploration_input = cf.replace_in_list(exploration_input,"_R_NUMBER_OF_STEPS_",str(it_nb_steps))

                    it_approx_time = ( prevexploration_json["subsys_nr"][it_subsys_nr]["s_per_step"] * it_nb_steps ) / 3600
                    it_approx_time = it_approx_time * 1.25
                    it_approx_time = int(np.ceil(it_approx_time))

                    del starting_point_list[RAND]
                    del RAND

                ### Write DATA file
                cf.write_file(local_apath/it_lammps_data_filename,it_lammps_data)

                ### Get print freq
                it_print_freq = int(it_nb_steps*0.01) if "print_freq" not in globals() else print_freq[it0_subsys_nr]
                exploration_input = cf.replace_in_list(exploration_input,"_R_PRINT_FREQ_",str(it_print_freq))

                # Plumed files
                if any("plumed" in f for f in exploration_input):
                    list_plumed_files=[x for x in (training_iterative_apath/"inputs").glob("*plumed*_"+it_subsys_nr+".dat")]
                    if len(list_plumed_files) == 0 :
                        logging.critical("Plumed in LAMMPS input but no plumed files")
                        logging.critical("Aborting...")
                        sys.exit(1)
                    plumed_input={}
                    for it_list_plumed_files in list_plumed_files:
                        plumed_input[it_list_plumed_files.name] = cf.read_file(it_list_plumed_files)
                    exploration_input = cf.replace_in_list(exploration_input,"_R_PLUMED_IN_","plumed_"+str(it_subsys_nr)+".dat")
                    exploration_input = cf.replace_in_list(exploration_input,"_R_PLUMED_OUT_","plumed_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".log")
                    for it_plumed_input in plumed_input:
                        plumed_input[it_plumed_input] = cf.replace_in_list(plumed_input[it_plumed_input],"_R_PRINT_FREQ_",str(it_print_freq))
                        cf.write_file(local_apath/it_plumed_input,plumed_input[it_plumed_input])
                    del list_plumed_files, it_list_plumed_files

                exploration_json["subsys_nr"][it_subsys_nr]["nb_steps"] = it_nb_steps
                exploration_json["subsys_nr"][it_subsys_nr]["print_freq"] = it_print_freq

                ### Write INPUT file
                cf.write_file(local_apath/(str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".in"),exploration_input)

                ### #35
                ### Slurm file now
                slurm_file = slurm_file_master.copy()
                slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",project_name)
                slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_",str(it_approx_time)+":00:00")
                slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_MODEL_VERSION_",str(exploration_json["deepmd_model_version"]))
                if allocation_name == "v100":
                    slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",allocation_name)
                    if it_approx_time <= 20:
                        if arch_name == "v100":
                            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
                            slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p13")
                            slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
                        elif arch_name == "a100":
                            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
                            slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p4")
                            slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
                    else:
                        slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t4")
                        slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p13")
                        slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_RSUBPARTITION_")
                elif allocation_name == "a100":
                    slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",allocation_name)
                    if it_approx_time <= 20:
                        slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
                    else:
                        slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t4")
                    slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p5")
                    slurm_file = cf.replace_in_list(slurm_file,"_R_SUBPARTITION_",arch_name)
                else:
                    logging.critical("Unknown error. Please BUG REPORT")
                    logging.critical("Aborting...")
                    sys.exit(1)
                if slurm_email != "":
                    slurm_file = cf.replace_in_list(slurm_file,"##SBATCH --mail-type","#SBATCH --mail-type")
                    slurm_file = cf.replace_in_list(slurm_file,"##SBATCH --mail-user _R_EMAIL_","#SBATCH --mail-user "+slurm_email)

                slurm_file = cf.replace_in_list(slurm_file,"_R_INPUT_",str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill)
                slurm_file = cf.replace_in_list(slurm_file,"_R_DATA_FILE_",it_lammps_data_filename)

                ### Add plumed files
                if any("plumed" in f for f in exploration_input):
                    for n,it_plumed_input in enumerate(plumed_input):
                        if n == 0:
                            slurm_file = cf.replace_in_list(slurm_file,"_R_PLUMED_FILES_LIST_",it_plumed_input)
                        else:
                            slurm_file = cf.replace_in_list(slurm_file,prev_plumed,prev_plumed+"\" \""+it_plumed_input)
                        prev_plumed = it_plumed_input
                    del n, it_plumed_input, plumed_input, prev_plumed
                else:
                    slurm_file = cf.replace_in_list(slurm_file,"_R_PLUMED_FILES_LIST_","")

                models_list_job = models_list.replace(" ","\" \"")
                slurm_file = cf.replace_in_list(slurm_file, "_R_MODELS_LIST_", models_list_job)

                cf.write_file(local_apath/("job_deepmd_"+it_exploration_type+"_"+arch_type+"_"+cluster+".sh"),slurm_file)

                del exploration_input, slurm_file, models_list_job

            del local_apath, models_list

        del it_number

    del it_nnp

    config_json["subsys_nr"][it_subsys_nr]["cell"] = it_cell
    config_json["subsys_nr"][it_subsys_nr]["nb_atm"] = it_nb_atm
    config_json["subsys_nr"][it_subsys_nr]["exploration_type"] = it_exploration_type

    del it_temperature, it_exploration_type, it_cell, it_nb_atm, it_nb_steps, it_exploration_input
    del slurm_file_master, it_lammps_data, it_timestep, it_lammps_data_filename, it_approx_time, it_print_freq

del it0_subsys_nr, it_subsys_nr

exploration_json["is_locked"] = True
exploration_json["is_launched"] = False
exploration_json["is_checked"] = False
exploration_json["is_deviated"] = False
exploration_json["is_extracted"] = False

## Dump the config/training
cf.json_dump(config_json,(control_apath/"config.json"),True)
cf.json_dump(exploration_json,(control_apath/("exploration_"+current_iteration_zfill+".json")),True)

### Cleaning
del config_json, training_iterative_apath, control_apath, jobs_apath
del current_iteration, current_iteration_zfill
del exploration_json
del cluster, arch_type
del project_name, allocation_name, arch_name
del deepmd_iterative_apath
del slurm_email

del previous_iteration, previous_iteration_zfill
del prevtraining_json

del sys, Path, logging, cf
del subprocess, np, random
import gc; gc.collect(); del gc
exit()