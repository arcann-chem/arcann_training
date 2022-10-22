
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import numpy as np
from collections import defaultdict
import gc

deepmd_iterative_apath = "_DEEPMD_ITERATIVE_APATH_"
sys.path.insert(0, deepmd_iterative_apath+"/scripts/")
import common_functions as cf

training_iterative_apath = str(Path("..").resolve())
config_json_fpath = training_iterative_apath+"/control/config.json"
config_json = cf.json_read(config_json_fpath,True,True)

current_iteration_zfill=Path().resolve().parts[-1].split("-")[0]
current_iteration=int(current_iteration_zfill)

test_json_fpath = training_iterative_apath+"/control/test_"+current_iteration_zfill+".json"
test_json = cf.json_read(test_json_fpath,True,True)

training_json_fpath = training_iterative_apath+"/control/training_"+current_iteration_zfill+".json"
training_json = cf.json_read(training_json_fpath,True,True)

compressed = "_compressed" if test_json["is_compressed"] else ""

energy_sys = defaultdict(dict)
energy_trained = defaultdict(dict)
energy_not_trained = defaultdict(dict)

for it_data_folders in Path("./data").iterdir():
    if it_data_folders.is_dir():
        data_name_t = str(it_data_folders.name)
        for it_nnp in range(1,config_json["nb_nnp"]+1):
            it_nnp = str(it_nnp)
            if Path("./out_NNP"+it_nnp+"/graph_"+it_nnp+"_"+current_iteration_zfill+compressed+"_"+data_name_t+".e.out").is_file():
                data = np.genfromtxt("./out_NNP"+it_nnp+"/graph_"+it_nnp+"_"+current_iteration_zfill+compressed+"_"+data_name_t+".e.out")
                energy_sys[it_nnp][data_name_t]=data
                if data_name_t in training_json["training_data"] :
                    energy_trained[it_nnp]=np.vstack((energy_trained[it_nnp],data)) if it_nnp in energy_trained.keys() else data
                else:
                    energy_not_trained[it_nnp]=np.vstack((energy_not_trained[it_nnp],data)) if it_nnp in energy_not_trained.keys() else data
        del it_nnp
del it_data_folders
np.savez("./energy_sys.npz", **energy_sys)
np.savez("./energy_trained.npz", **energy_trained)
np.savez("./energy_not_trained.npz", **energy_not_trained)
del data, energy_sys, energy_trained, energy_not_trained
gc.collect()

force_sys = defaultdict(dict)
for it_data_folders in Path("./data").iterdir():
    if it_data_folders.is_dir():
        data_name_t = str(it_data_folders.name)
        for it_nnp in range(1,config_json["nb_nnp"]+1):
            it_nnp = str(it_nnp)
            if Path("./out_NNP"+it_nnp+"/graph_"+it_nnp+"_"+current_iteration_zfill+compressed+"_"+data_name_t+".f.out").is_file():
                data = np.genfromtxt("./out_NNP"+it_nnp+"/graph_"+it_nnp+"_"+current_iteration_zfill+compressed+"_"+data_name_t+".f.out")
                data=np.vstack((data[:,(0,3)],data[:,(1,4)],data[:,(2,5)]))
                force_sys[it_nnp][data_name_t]=data
        del it_nnp
del it_data_folders
np.savez("./force_sys.npz", **force_sys)
del data, force_sys
gc.collect()

force_trained = defaultdict(dict)
force_not_trained = defaultdict(dict)
for it_data_folders in Path("./data").iterdir():
    if it_data_folders.is_dir():
        data_name_t = str(it_data_folders.name)
        for it_nnp in range(1,config_json["nb_nnp"]+1):
            it_nnp = str(it_nnp)
            if Path("./out_NNP"+it_nnp+"/graph_"+it_nnp+"_"+current_iteration_zfill+compressed+"_"+data_name_t+".f.out").is_file():
                data = np.genfromtxt("./out_NNP"+it_nnp+"/graph_"+it_nnp+"_"+current_iteration_zfill+compressed+"_"+data_name_t+".f.out")
                data=np.vstack((data[:,(0,3)],data[:,(1,4)],data[:,(2,5)]))
                if data_name_t in training_json["training_data"] :
                    force_trained[it_nnp]=np.vstack((force_trained[it_nnp],data)) if it_nnp in force_trained.keys() else data
                else:
                    force_not_trained[it_nnp]=np.vstack((force_not_trained[it_nnp],data)) if it_nnp in force_not_trained.keys() else data
        del it_nnp, data_name_t
del it_data_folders, compressed
np.savez("./force_trained.npz", **force_trained)
np.savez("./force_not_trained.npz", **force_not_trained)
del data, force_trained, force_not_trained
gc.collect()

test_json["is_concatenated"] = True
cf.json_dump(test_json,test_json_fpath,True,"test.json")
logging.info("The DP-Test: concatenation phase is a success!")

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del test_json, test_json_fpath
del training_json, training_json_fpath
del current_iteration, current_iteration_zfill
del deepmd_iterative_apath

del sys, Path, logging, cf
del np, defaultdict, gc
print(globals())
import gc; gc.collect(); del gc
exit()