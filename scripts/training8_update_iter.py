###################################### No change past here
from genericpath import exists
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

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
training_json = cf.json_read((control_apath/("training_"+current_iteration_zfill+".json")),True,True)

### Checks
if not training_json["is_frozen"]:
    logging.critical("Maybe freeze the NNPs before updating the iteration?")
    logging.critical("Aborting...")
    sys.exit(1)

### Prep the next iteration
for it_nnp in range(1, config_json["nb_nnp"] + 1):
    local_apath = Path(".").resolve()/str(it_nnp)
    cf.check_file(local_apath/("graph_"+str(it_nnp)+"_"+current_iteration_zfill+".pb"),True,True)
    if training_json["is_compressed"] is True:
        cf.check_file(local_apath/("graph_"+str(it_nnp)+"_"+current_iteration_zfill+"_compressed.pb"),True,True)
    cf.remove_file(local_apath/("checkpoint"))
    cf.remove_file(local_apath/("input_v2_compat"))
    cf.remove_file_glob(local_apath,"DeepMD_*")
    cf.remove_file_glob(local_apath,"model.ckpt-*")
    if (local_apath/"model-compression").is_dir():
        cf.remove_tree(local_apath/"model-compression")

(training_iterative_apath/(current_iteration_zfill+"-test")).mkdir(exist_ok=True)
cf.check_dir((training_iterative_apath/(current_iteration_zfill+"-test")),True)

subprocess.call(["rsync","-a", str(training_iterative_apath/"data"), str(training_iterative_apath/(current_iteration_zfill+"-test"))])

(training_iterative_apath/"NNP").mkdir(exist_ok=True)
cf.check_dir(training_iterative_apath/"NNP",True)

local_apath = Path(".").resolve()

for it_nnp in range(1, config_json["nb_nnp"] + 1):
    if training_json["is_compressed"] is True:
        subprocess.call(["rsync","-a", str(local_apath/(str(it_nnp)+"/graph_"+str(it_nnp)+"_"+current_iteration_zfill+"_compressed.pb")), str((training_iterative_apath/"NNP"))])
    subprocess.call(["rsync","-a", str(local_apath/(str(it_nnp)+"/graph_"+str(it_nnp)+"_"+current_iteration_zfill+".pb")), str((training_iterative_apath/"NNP"))])
del it_nnp

current_iteration = current_iteration+1
config_json["current_iteration"] = current_iteration
current_iteration_zfill = str(current_iteration).zfill(3)

for it_steps in ["exploration","reactive","labeling","training"]:
    (training_iterative_apath/(current_iteration_zfill+"-"+it_steps)).mkdir(exist_ok=True)
    cf.check_dir(training_iterative_apath/(current_iteration_zfill+"-"+it_steps),True)
del it_steps

cf.json_dump(config_json,(control_apath/"config.json"),True)

if (local_apath/"data").is_dir():
    cf.remove_tree(local_apath/"data")
del local_apath

logging.info("Updating the iteration is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath
del current_iteration, current_iteration_zfill
del training_json
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
print(globals())
exit()