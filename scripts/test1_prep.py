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
    if (Path(deepmd_iterative_apath)/"tools"/"common_functions.py").is_file():
        deepmd_iterative_apath = Path(deepmd_iterative_apath)
        deepmd_iterative_apath_error = 0
elif (Path().home()/"deepmd_iterative_py"/"tools"/"common_functions.py").is_file():
    deepmd_iterative_apath = Path().home()/"deepmd_iterative_py"
    deepmd_iterative_apath_error = 0
elif (training_iterative_apath/"control"/"path").is_file():
    deepmd_iterative_apath = Path((training_iterative_apath/"control"/"path").read_text())
    if (deepmd_iterative_apath/"tools"/"common_functions.py").is_file():
        deepmd_iterative_apath_error = 0
if deepmd_iterative_apath_error == 1:
    logging.critical("Can\'t find common_functions.py in usual places:")
    logging.critical("deepmd_iterative_apath variable or ~/deepmd_iterative_py or in the path file in control")
    logging.critical("Aborting...")
    sys.exit(1)
sys.path.insert(0, str(deepmd_iterative_apath/"tools"))
del deepmd_iterative_apath_error
import common_functions as cf

slurm_email = "" if "slurm_email" not in globals() else slurm_email

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
jobs_apath = deepmd_iterative_apath/"jobs"/"test"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
config_json = cf.json_read((control_apath/"config.json"),True,True)
training_json = cf.json_read((control_apath/("training_"+current_iteration_zfill+".json")),True,True)
test_json = cf.json_read((control_apath/("test_"+current_iteration_zfill+".json")),False,True)
current_apath = Path(".").resolve()

if not training_json["is_frozen"]:
    logging.critical("Lock found. Previous NNPs aren\'t frozen")
    logging.critical("Aborting...")
    sys.exit(1)

### Set needed variables
test_json["nb_nnp"] = config_json["nb_nnp"]
test_json["is_compressed"] = training_json["is_compressed"]

### #35
cluster = cf.check_cluster()
test_json["cluster"] = cluster
test_json["project_name"] = "nvs" if "project_name" not in globals() else project_name
test_json["allocation_name"] = "v100" if "allocation_name" not in globals() else allocation_name
test_json["arch_name"] = "v100" if "arch_name" not in globals() else arch_name
project_name = test_json["project_name"]
allocation_name = test_json["allocation_name"]
arch_name = test_json["arch_name"]
if arch_name == "v100" or arch_name == "a100":
    arch_type ="gpu"

### #35
cf.check_file(jobs_apath/("job_deepmd_test_"+arch_type +"_"+cluster+".sh"),True,True)
slurm_master = cf.read_file(jobs_apath/("job_deepmd_test_"+arch_type +"_"+cluster+".sh"))

###
compressed = "_compressed" if test_json["is_compressed"] else ""
for it_nnp in range(1, test_json["nb_nnp"] + 1):
    slurm_file = slurm_master.copy()
    cf.check_file(training_iterative_apath/"NNP"/("graph_"+str(it_nnp)+"_"+current_iteration_zfill+compressed+".pb"),True,True)
    subprocess.call(["rsync","-a", str(training_iterative_apath/"NNP"/("graph_"+str(it_nnp)+"_"+current_iteration_zfill+compressed+".pb")), str(current_apath)])
    slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_MODEL_","graph_"+str(it_nnp)+"_"+current_iteration_zfill+compressed)
    slurm_file = cf.replace_in_list(slurm_file,"_R_NNPNB_","NNP"+str(it_nnp))
    slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",project_name)
    slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_","04:00:00")
    slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_VERSION_",str(training_json["deepmd_model_version"]))

    if allocation_name == "v100":
        slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",allocation_name)
        if arch_name == "v100":
            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
            slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p13")
            slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
        elif arch_name == "a100":
            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
            slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p4")
            slurm_file = cf.replace_in_list(slurm_file,"#SBATCH -C _R_SUBPARTITION_","##SBATCH -C _R_SUBPARTITION_")
    elif allocation_name == "a100":
        slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",allocation_name)
        slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_","qos_gpu-t3")
        slurm_file = cf.replace_in_list(slurm_file,"_R_PARTITION_","gpu_p5")
        slurm_file = cf.replace_in_list(slurm_file,"_R_SUBPARTITION_",arch_name)
    else:
        logging.critical("Unknown error. Please BUG REPORT")
        logging.critical("Aborting...")
        sys.exit(1)
    if slurm_email != "":
        slurm_file = cf.replace_in_list(slurm_file,"##SBATCH --mail-type","#SBATCH --mail-type")
        slurm_file = cf.replace_in_list(slurm_file,"##SBATCH --mail-user _R_EMAIL_","#SBATCH --mail-user "+slurm_email)
    cf.write_file(current_apath/("job_deepmd_test_"+arch_type +"_"+cluster+"_NNP"+str(it_nnp)+".sh"),slurm_file)
    del slurm_file
del it_nnp
del slurm_master, compressed

test_json["is_locked"] = True
test_json["is_launched"] = False
test_json["is_checked"] = False
test_json["is_concatenated"] = False
test_json["is_plotted"] = False

cf.json_dump(test_json,(control_apath/("test_"+current_iteration_zfill+".json")),True)
logging.info("DP-Test: Prep phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath, jobs_apath, current_apath
del training_json
del test_json
del current_iteration, current_iteration_zfill
del cluster, arch_type
del project_name, allocation_name, arch_name
del deepmd_iterative_apath
del slurm_email

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()