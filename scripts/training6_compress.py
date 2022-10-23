## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name: str = "nvs"
# allocation_name: str = "v100"
# arch_name: str = "v100"
# slurm_email: str = ""

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

slurm_email = "" if "slurm_email" not in globals() else slurm_email

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
config_json = cf.json_read((control_apath/"config.json"),True,True)
jobs_apath = deepmd_iterative_apath/"jobs"/"training"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
training_json = cf.json_read((control_apath/("training_"+current_iteration_zfill+".json")),True,True)

### Set needed variables issue#35
project_name = training_json["project_name"] if "project_name" not in globals() else project_name
allocation_name = training_json["allocation_name"] if "allocation_name" not in globals() else allocation_name
arch_name = training_json["arch_name"] if "arch_name" not in globals() else arch_name
if arch_name == "v100" or arch_name == "a100":
    arch_type ="gpu"

### Checks
if training_json["deepmd_model_version"] < 2.0:
    logging.critical("No compression for model < 2.0 and your model is version:"+str(training_json["deepmd_model_version"]))
    logging.critical("Aborting...")
    sys.exit(1)
if not training_json["is_frozen"]:
    logging.critical("Lock found. Run/Check first: training5_checkfreeze.py")
    logging.critical("Aborting...")
    sys.exit(1)

### issue#35
cluster = cf.check_cluster()
cf.check_file(jobs_apath/("job_deepmd_compress_"+arch_type +"_"+cluster+".sh"),True,True,"No SLURM file present for the freezing step on this cluster.")
slurm_file_master = cf.read_file(jobs_apath/("job_deepmd_compress_"+arch_type +"_"+cluster+".sh"))
del jobs_apath

### Prep and launch DP Compress
check = 0
for it_nnp in range(1, config_json["nb_nnp"] + 1):
    local_apath = Path(".").resolve()/str(it_nnp)
    cf.check_file(local_apath/("graph_"+str(it_nnp)+"_"+current_iteration_zfill+".pb"),True,True)
    slurm_file = slurm_file_master
    slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",project_name)
    slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_","02:00:00")
    slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_VERSION_",str(training_json["deepmd_model_version"]))
    slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_MODEL_","graph_"+str(it_nnp)+"_"+current_iteration_zfill)
    if allocation_name == "v100":
        slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_","v100")
        if arch_name == "v100":
            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
            slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p13")
            slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
        else:
            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
            slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p4")
            slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
    elif allocation_name == "a100":
        slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_","a100")
        slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
        slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p5")
        slurm_file = cf.replace_in_list(slurm_file,"_R_SUBPARTITION_","a100")
    else:
        sys.exit("Unknown error. Please BUG REPORT.\n Aborting...")
    cf.write_file(local_apath/("job_deepmd_compress_"+arch_type+"_"+cluster+".sh"),slurm_file)
    if (local_apath/("job_deepmd_compress_"+arch_type+"_"+cluster+".sh")).is_file():
        cf.change_dir(local_apath)
        subprocess.call(["sbatch","./job_deepmd_compress_"+arch_type+"_"+cluster+".sh"])
        cf.change_dir(local_apath.parent)
        logging.info("DP Compress - "+str(it_nnp)+" launched")
        check = check + 1
    else:
        logging.warning("DP Compress - "+str(it_nnp)+" NOT launched")
    del local_apath
del it_nnp, slurm_file, slurm_file_master

if check == config_json["nb_nnp"]:
    logging.info("DP-Freeze: SLURM phase is a success!")
else:
    logging.critical("Some DP Compress did not launched correctly")
    logging.critical("Please launch manually before continuing to the next step")
del check

### Cleaning
del config_json, training_iterative_apath, control_apath
del current_iteration, current_iteration_zfill
del training_json
del cluster, arch_type
del deepmd_iterative_apath
del project_name, allocation_name, arch_name
del slurm_email

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()