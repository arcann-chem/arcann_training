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

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
config_json = cf.json_read((control_apath/"config.json"),True,True)
labeling_json = cf.json_read((control_apath/("labeling_"+current_iteration_zfill+".json")),True,True)

### Checks
if labeling_json["is_launched"]:
    logging.critical("Already launched.")
    logging.critical("Aborting...")
    sys.exit(1)

if not labeling_json["is_locked"]:
    logging.critical("Lock found. Run/Check first: labeling1_prep.py")
    logging.critical("Aborting...")
    sys.exit(1)

###
cluster = cf.check_cluster()
cf.check_same_cluster(cluster,labeling_json)

### Launch of the labeling
check = 0
for it0_subsys_nr,it_subsys_nr in enumerate(config_json["subsys_nr"]):
    subsys_apath = Path(".").resolve()/str(it_subsys_nr)
    if cluster == "jz":
        if (subsys_apath/("job_labeling_array_"+labeling_json["arch_type"]+"_"+cluster+".sh")).is_file():
            cf.change_dir(subsys_apath)
            subprocess.call(["sbatch","./job_labeling_array_"+labeling_json["arch_type"]+"_"+cluster+".sh"])
            cf.change_dir(subsys_apath.parent)
            logging.info("Labeling - "+str(it_subsys_nr)+" launched")
            check = check + 1
        else:
            logging.critical("Labeling - "+str(it_subsys_nr)+" NOT launched")
    elif cluster == "ir":
        if it0_subsys_nr == 0:
            if (subsys_apath/("job_labeling_array_"+labeling_json["arch_type"]+"_"+cluster+".sh")).is_file():
                cf.change_dir(subsys_apath)
                subprocess.call(["ccc_msub","./job_labeling_array_"+labeling_json["arch_type"]+"_"+cluster+".sh"])
                cf.change_dir(subsys_apath.parent)
                logging.info("Labeling - "+str(it_subsys_nr)+" launched")
                check = check + 1
            elif (subsys_apath/("job_labeling_array_"+labeling_json["arch_type"]+"_"+cluster+"_0.sh")).is_file():
                cf.change_dir(subsys_apath)
                subprocess.call(["ccc_msub","./job_labeling_array_"+labeling_json["arch_type"]+"_"+cluster+"_0.sh"])
                cf.change_dir(subsys_apath.parent)
                logging.info("Labeling - "+str(it_subsys_nr)+" - 0 launched")
                check = check + 1
            else:
                logging.critical("Labeling Array - "+str(it_subsys_nr)+" NOT launched")
            logging.warning("Since Irene-Rome does not support more than 300 jobs at a time")
            logging.warning("and SLURM arrays not larger than 1000")
            logging.warning("the labeling array have been split into several jobs")
            logging.warning("and should launch itself automagically until the labeling is complete")
        else:
            ### On Irene-Rome, only the first one is launched
            True
del it_subsys_nr, it0_subsys_nr, subsys_apath

if check == len(config_json["subsys_nr"]):
    labeling_json["is_launched"] = True
    logging.info("Labeling: SLURM phase is a success!")
elif cluster == "ir" and check > 0 :
    labeling_json["is_launched"] = True
    logging.info("Labeling: SLURM phase is a semi-success! (You are on Irene-Rome so who knows what can happen...)")
else:
    logging.critical("Some labeling arrays did not launched correctly")
    logging.critical("Please launch manually before continuing to the next step")
    logging.critical("And replace the key \"is_launched\" to True in the corresponding labeling.json.")
del check

cf.json_dump(labeling_json,(control_apath/("labeling_"+current_iteration_zfill+".json")),True)

### Clean
del config_json, training_iterative_apath, control_apath
del current_iteration, current_iteration_zfill
del labeling_json
del cluster
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()