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
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
exploration_json = cf.json_read((control_apath/("exploration_"+current_iteration_zfill+".json")),True,True)

### Checks
if exploration_json["is_launched"]:
    logging.critical("Already launched.")
    logging.critical("Aborting...")
    sys.exit(1)
if not exploration_json["is_locked"]:
    logging.critical("Lock found. Run/Check first: exploration1_prep.py")
    logging.critical("Aborting...")
    sys.exit(1)

### #35
cluster = cf.check_cluster()
if exploration_json["arch_name"] == "v100" or exploration_json["arch_name"] == "a100":
    arch_type="gpu"

if exploration_json["cluster"] != cluster:
    logging.critical("Different cluster ("+str(cluster)+") than the one for exploration1_prep..py ("+str(exploration_json["cluster"])+")")
    logging.critical("Aborting...")
    sys.exit(1)

exploration_type = exploration_json['exploration_type'] 
### Launch the jobs
check = 0
for it_subsys_nr in config_json["subsys_nr"]:
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        for it_each in range(1, exploration_json["nb_traj"] + 1):
            local_apath = Path(".").resolve()/str(it_subsys_nr)/str(it_nnp)/(str(it_each).zfill(5))
            if (local_apath/("job_deepmd_"+exploration_type+"_"+arch_type+"_"+cluster+".sh")).is_file():
                cf.change_dir(local_apath)
                subprocess.call(["sbatch","./job_deepmd_"+exploration_type+"_"+arch_type+"_"+cluster+".sh"])
                cf.change_dir(((local_apath.parent).parent).parent)
                logging.info("Exploration - "+str(it_subsys_nr)+"/"+str(it_nnp)+"/"+str(it_each).zfill(5)+" launched")
                check = check + 1
            else:
                logging.critical("Exploration - "+str(it_subsys_nr)+"/"+str(it_nnp)+"/"+str(it_each).zfill(5)+" NOT launched")
            del local_apath
        del it_each
    del it_nnp
del it_subsys_nr, exploration_type

if check == (len( exploration_json["subsys_nr"]) * exploration_json["nb_nnp"] * exploration_json["nb_traj"] ):
    exploration_json["is_launched"] = True
    logging.info("Exploration: SLURM phase is a success!")
else:
    logging.critical("Some Exploration did not launched correctly")
    logging.critical("Please launch manually before continuing to the next step")
    logging.critical("And replace the key \"is_launched\" to True in the corresponding exploration.json.")
del check

cf.json_dump(exploration_json,(control_apath/("exploration_"+current_iteration_zfill+".json")),True)

del config_json, training_iterative_apath, control_apath
del current_iteration, current_iteration_zfill
del exploration_json
del cluster, arch_type
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()