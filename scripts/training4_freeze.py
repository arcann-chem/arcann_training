## deepmd_iterative_apath
# deepmd_iterative_apath = ""
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name = "nvs"
# allocation_name = "v100"
# arch_name = "v100"
# slurm_email = ""

###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import subprocess

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
    logging.critical("Can\'t find common_functions.py in usual places:")
    logging.critical("deepmd_iterative_apath variable or ~/deepmd_iterative_py or in the path file in control")
    logging.critical("Aborting...")
    sys.exit(1)
sys.path.insert(0, str(Path(deepmd_iterative_apath)/"scripts"))
del deepmd_iterative_apath_error
import common_functions as cf

### Temp fix before Path/Str pass
training_iterative_apath = str(training_iterative_apath)
deepmd_iterative_apath = str(deepmd_iterative_apath)

### Read what is needed (json files)
config_json_fpath = training_iterative_apath+"/control/config.json"
config_json = cf.json_read(config_json_fpath,True,True)

current_iteration = current_iteration if "current_iteration" in globals() else config_json["current_iteration"]
current_iteration_zfill = str(current_iteration).zfill(3)

training_json_fpath = training_iterative_apath+"/control/training_"+current_iteration_zfill+".json"
training_json = cf.json_read(training_json_fpath,True,True)

### Set needed variables
project_name = project_name if "project_name" in globals() else training_json["project_name"]
allocation_name = allocation_name if "allocation_name" in globals() else training_json["allocation_name"]
arch_name = arch_name if "arch_name" in globals() else training_json["arch_name"]
if arch_name == "v100" or arch_name == "a100":
    arch_type ="gpu"

### Checks
if training_json["is_checked"] is False:
    logging.critical("Maybe check the training before freezing?")
    logging.critical("Aborting...")
    sys.exit(1)

cluster = cf.check_cluster()
cf.check_file(deepmd_iterative_apath+"/jobs/training/job_deepmd_freeze_"+arch_type+"_"+cluster+".sh",0,True,"No SLURM file present for the freezing step on this cluster.")

### Prep and launch DP Freeze
slurm_file_master = cf.read_file(deepmd_iterative_apath+"/jobs/training/job_deepmd_freeze_"+arch_type+"_"+cluster+".sh")
slurm_email = "" if "slurm_email" not in globals() else slurm_email

check = 0
for it_nnp in range(1, config_json["nb_nnp"] + 1):
    cf.change_dir("./"+str(it_nnp))
    cf.check_file("./model.ckpt.index",0,True,"./"+str(it_nnp)+"/model.ckpt.index not present.")
    slurm_file = slurm_file_master
    slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",project_name)
    slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_","01:00:00")
    slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_MODEL_VERSION_",str(training_json["deepmd_model_version"]))
    slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_MODEL_","graph_"+str(it_nnp)+"_"+current_iteration_zfill)
    if allocation_name == "v100":
        slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",allocation_name)
        if arch_name == "v100":
            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
            slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p13")
            slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
        else:
            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
            slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p4")
            slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
    elif allocation_name == "a100":
        slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_","allocation_name")
        slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
        slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p5")
        slurm_file = cf.replace_in_list(slurm_file,"_R_SUBPARTITION_","a100")
    else:
        sys.exit("Unknown error. Please BUG REPORT.\n Aborting...")
    cf.write_file("job_deepmd_freeze_"+arch_type+"_"+cluster+".sh",slurm_file)
    with open("checkpoint", "w") as f:
        f.write("model_checkpoint_path: \"model.ckpt\"\n")
        f.write("all_model_checkpoint_paths: \"model.ckpt\"\n")
        f.close()
    del f
    if Path("job_deepmd_freeze_"+arch_type+"_"+cluster+".sh").is_file():
        subprocess.call(["sbatch","./job_deepmd_freeze_"+arch_type+"_"+cluster+".sh"])
        logging.info("DP Freeze - ./"+str(it_nnp)+" launched")
        check = check + 1
    else:
        logging.critical("DP Freeze - ./"+str(it_nnp)+" NOT launched")
    cf.change_dir("..")
del it_nnp, slurm_file, slurm_file_master

if check == config_json["nb_nnp"]:
    logging.info("Slurm launch of DP Freeze is a success!")
else:
    logging.critical("Some DP Freeze did not launched correctly")
    logging.critical("Please launch manually before continuing to the next step")
del check

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del training_json, training_json_fpath
del cluster, arch_type
del deepmd_iterative_apath
del project_name, allocation_name, arch_name
del slurm_email

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()