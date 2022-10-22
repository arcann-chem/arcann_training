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

current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)

training_json_fpath = training_iterative_apath+"/control/training_"+current_iteration_zfill+".json"
training_json = cf.json_read(training_json_fpath,True,True)

### Checks
if training_json["is_frozen"] is False:
    logging.critical("Maybe freeze the NNPs before updating the iteration?")
    logging.critical("Aborting...")
    sys.exit(1)

### Prep the next iteration
for it_nnp in range(1, config_json["nb_nnp"] + 1):
    cf.change_dir("./"+str(it_nnp))
    cf.check_file("graph_"+str(it_nnp)+"_"+current_iteration_zfill+".pb",0,True)
    if training_json["is_compressed"] is True:
        cf.check_file("graph_"+str(it_nnp)+"_"+current_iteration_zfill+"_compressed.pb",0,True)
    cf.remove_file_glob(".","DeepMD_*")
    cf.remove_file_glob(".","model.ckpt-*")
    cf.remove_file("checkpoint")
    cf.remove_file("input_v2_compat.json")
    if Path("model-compression").is_dir():
        cf.remove_tree(Path("model-compression"))
    cf.change_dir("../")

cf.create_dir("../"+current_iteration_zfill+"-test")
subprocess.call(["rsync","-a", training_iterative_apath+"/data", "../"+current_iteration_zfill+"-test/"])

cf.create_dir("../NNP")
for it_nnp in range(1, config_json["nb_nnp"] + 1):
    if training_json["is_compressed"] is True:
        subprocess.call(["rsync","-a", "./"+str(it_nnp)+"/graph_"+str(it_nnp)+"_"+current_iteration_zfill+"_compressed.pb", "../NNP/"])
    else:
        subprocess.call(["rsync","-a", "./"+str(it_nnp)+"/graph_"+str(it_nnp)+"_"+current_iteration_zfill+".pb", "../NNP/"])
del it_nnp

current_iteration = current_iteration+1
config_json["current_iteration"] = current_iteration
current_iteration_zfill = str(current_iteration).zfill(3)

cf.create_dir("../"+current_iteration_zfill+"-exploration")
cf.create_dir("../"+current_iteration_zfill+"-reactive")
cf.create_dir("../"+current_iteration_zfill+"-labeling")
cf.create_dir("../"+current_iteration_zfill+"-training")

cf.json_dump(config_json,config_json_fpath,True,"config file")

if Path("data").is_dir():
    cf.remove_tree(Path("data"))

logging.info("Updating the iteration is a success!")

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del training_json, training_json_fpath
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()