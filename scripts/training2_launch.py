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

training_json_fpath = str(Path(training_iterative_apath+"/control/training_"+current_iteration_zfill+".json").resolve())
training_json = cf.json_read(training_json_fpath,True,True)

### Checks
if training_json["is_launched"] is True:
    logging.critical("Already launched.")
    logging.critical("Aborting...")
    sys.exit(1)

if training_json["is_locked"] is False:
    logging.critical("Lock found. Run/Check first: training1_prep.py")
    logging.critical("Aborting...")
    sys.exit(1)

cluster = cf.check_cluster()

if training_json["cluster"] != cluster:
    logging.critical("Different cluster ("+str(cluster)+") than the one for training1_prep.py ("+str(training_json["cluster"])+")")
    logging.critical("Aborting...")
    sys.exit(1)

### Set needed variables
if training_json["arch_name"] == "v100" or training_json["arch_name"] == "a100":
    arch_type="gpu"

### Launch the jobs
check = 0
for it_nnp in range(1, config_json["nb_nnp"] + 1):
    cf.change_dir("./"+str(it_nnp))
    if Path("job_deepmd_train_"+arch_type+"_"+cluster+".sh").is_file():
        subprocess.call(["sbatch","./job_deepmd_train_"+arch_type+"_"+cluster+".sh"])
        logging.info("DP Train - ./"+str(it_nnp)+" launched")
        check = check + 1
    else:
        logging.critical("DP Train - ./"+str(it_nnp)+" NOT launched")
    cf.change_dir("..")
del it_nnp

if check == config_json["nb_nnp"]:
    training_json["is_launched"] = True
    logging.info("Slurm launch of the training is a success!")
else:
    logging.critical("Some DP Train did not launched correctly")
    logging.critical("Please launch manually before continuing to the next step")
    logging.critical("And replace the key \"is_launched\" to True in the corresponding training.json.")
del check

cf.json_dump(training_json,training_json_fpath,True,"training.json")

del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del training_json, training_json_fpath
del cluster, arch_type
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()