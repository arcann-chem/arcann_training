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

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
test_json = cf.json_read((control_apath/("test_"+current_iteration_zfill+".json")),True,True)
current_apath = Path(".").resolve()

### Checks
if test_json["is_launched"]:
    logging.critical("Already launched.")
    logging.critical("Aborting...")
    sys.exit(1)

if not test_json["is_locked"]:
    logging.critical("Lock found. Run/Check first: test1_prep.py")
    logging.critical("Aborting...")
    sys.exit(1)

cluster = cf.check_cluster()
cf.check_same_cluster(cluster,test_json)

### Launch the jobs
check = 0
for it_nnp in range(1, test_json["nb_nnp"] + 1):
    if (current_apath/("job_deepmd_test_"+test_json["test"]["arch_type"]+"_"+cluster+"_NNP"+str(it_nnp)+".sh")).is_file():
        subprocess.call(["sbatch",str(current_apath/("job_deepmd_test_"+test_json["test"]["arch_type"]+"_"+cluster+"_NNP"+str(it_nnp)+".sh"))])
        logging.info("DP Test - "+str(it_nnp)+" launched")
        check = check + 1
    else:
        logging.warning("DP Test - "+str(it_nnp)+" NOT launched")
del it_nnp

if check == test_json["nb_nnp"]:
    test_json["is_launched"] = True
    logging.info("DP-Test: SLURM phase is a success!")
else:
    logging.critical("Some DP-Test did not launched correctly")
    logging.critical("Please launch manually before continuing to the next step")
    logging.critical("And replace the key \"is_launched\" to True in the corresponding training.json.")
del check

cf.json_dump(test_json,(control_apath/("test_"+current_iteration_zfill+".json")),True)

### Cleaning
del training_iterative_apath, control_apath, current_apath
del current_iteration, current_iteration_zfill
del test_json
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()