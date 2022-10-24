## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

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
if not training_json["is_frozen"]:
    logging.critical("Lock found. Run/Check first: training6_compress.py")
    logging.critical("Aborting...")
    sys.exit(1)

### Check normal termination of DP Compress
check = 0
for it_nnp in range(1, config_json["nb_nnp"] + 1):
    local_apath = Path(".").resolve()/str(it_nnp)
    if (local_apath/("graph_"+str(it_nnp)+"_"+current_iteration_zfill+"_compressed.pb")).is_file():
        check = check + 1
    else:
        logging.critical("DP Compress - "+str(it_nnp)+" not finished/failed")
    del local_apath
del it_nnp

if check == config_json["nb_nnp"]:
    training_json["is_compressed"] = True
else:
    logging.critical("Some DP Compress did not finished correctly")
    logging.critical("Please check manually before relaunching this step")
    logging.critical("Aborting...")
    sys.exit(1)
del check

training_json["is_compressed"] = True

cf.json_dump(training_json,(control_apath/("training_"+current_iteration_zfill+".json")),True)

logging.info("DP-Compress: Check phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath
del current_iteration, current_iteration_zfill
del training_json
del deepmd_iterative_apath

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()