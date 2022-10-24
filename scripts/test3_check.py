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
test_json = cf.json_read((control_apath/("test_"+current_iteration_zfill+".json")),True,True)
current_apath = Path(".").resolve()

### Checks
if not test_json["is_launched"]:
    logging.critical("Lock found. Run/Check first: test2_launch.py")
    logging.critical("Aborting...")
    sys.exit(1)

### Check the normal termination of the test phase
compressed = "_compressed" if test_json["is_compressed"] else ""
check = 0
total = 0
for it_data_folders in (current_apath/"data").iterdir():
    if it_data_folders.is_dir():
        data_name_t = str(it_data_folders.name)
        for it_nnp in range(1,config_json["nb_nnp"]+1):
            total = total + 1
            if (current_apath/("out_NNP"+str(it_nnp)+"/graph_"+str(it_nnp)+"_"+current_iteration_zfill+compressed+"_"+data_name_t+".e.out")).is_file():
                check = check +1
            else:
                logging.warning("No output for NNP "+str(it_nnp)+" and dataset "+data_name_t)
        del it_nnp, data_name_t
del it_data_folders, compressed

if check != total:
    logging.critical("Some jobs failed or are still running.")
    logging.critical("Please check manually before relaunching this step")
    logging.critical("Aborting...")
    sys.exit(1)
else:
    test_json["is_checked"] = True
del check, total

cf.json_dump(test_json,(control_apath/("test_"+current_iteration_zfill+".json")),True)
cf.remove_file_glob(current_apath,"DeepMD_Test.*")
logging.info("The DP-Test: Check phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath, current_apath
del current_iteration, current_iteration_zfill
del test_json
del deepmd_iterative_apath

del sys, Path, logging, cf
import gc; gc.collect(); del gc
print(globals())
exit()