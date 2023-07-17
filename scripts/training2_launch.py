#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Very fancy DeepMD-based semi-automatic highly-customizable iterative training procedure. #
#   Copyright 2022-2023 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#

## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""

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

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
training_json = cf.json_read((control_apath/("training_"+current_iteration_zfill+".json")),True,True)

### Checks
if training_json["is_launched"]:
    logging.critical("Already launched.")
    logging.critical("Aborting...")
    sys.exit(1)
if not training_json["is_locked"]:
    logging.critical("Lock found. Run/Check first: training1_prep.py")
    logging.critical("Aborting...")
    sys.exit(1)

###
cluster = cf.check_cluster()
cf.check_same_cluster(cluster,training_json)

### Launch the jobs
check = 0
for it_nnp in range(1, config_json["nb_nnp"] + 1):
    local_apath = Path(".").resolve()/str(it_nnp)
    if (local_apath/("job_deepmd_train_"+training_json["arch_type"]+"_"+cluster+".sh")).is_file():
        cf.change_dir(local_apath)
        subprocess.call(["sbatch","./job_deepmd_train_"+training_json["arch_type"]+"_"+cluster+".sh"])
        cf.change_dir(local_apath.parent)
        logging.info("DP Train - "+str(it_nnp)+" launched")
        check = check + 1
    else:
        logging.critical("DP Train - "+str(it_nnp)+" NOT launched")
    del local_apath
del it_nnp

if check == config_json["nb_nnp"]:
    training_json["is_launched"] = True
    logging.info("DP-Train: SLURM phase is a success!")
else:
    logging.critical("Some DP Train did not launched correctly")
    logging.critical("Please launch manually before continuing to the next step")
    logging.critical("And replace the key \"is_launched\" to True in the corresponding training.json.")
del check

cf.json_dump(training_json,(control_apath/("training_"+current_iteration_zfill+".json")),True)

del config_json, training_iterative_apath, control_apath
del current_iteration, current_iteration_zfill
del training_json
del cluster
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()