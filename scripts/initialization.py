## deepmd_iterative_apath
deepmd_iterative_apath="/gpfs7kw/linkhome/rech/gennsp01/ucf13sj/code/deepmd_iterative_py"
## Set your system name, subsystem ("easy" exploration, standard TEMP, presents from the START of the iterative training) and the number of NNP you want to use
sys_name="NAME"
subsys_name = ["SYSTEM1","SYSTEM2"]
nb_nnp = 3
## These are the default
# temperature_K = [300, 300]
# timestep_ps = [0.0005, 0.0005]
# nb_candidates_max = [500, 500]
# s_low = [0.1, 0.1]
# s_high = [0.8, 0.8]
# s_high_max = [1.0, 1.0]
# ignore_first_x_ps = [0.5, 0.5]

###################################### No change past here
import sys
from pathlib import Path
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

training_iterative_apath = str(Path().resolve())
deepmd_iterative_apath = str(Path(deepmd_iterative_apath).resolve())

sys.path.insert(0, deepmd_iterative_apath+"/scripts/")
import common_functions as cf

### Create the config.json
config_json_fpath = training_iterative_apath+"/control/config.json"
config_json = {}
config_json["training_iterative_apath"] = training_iterative_apath
config_json["deepmd_iterative_apath"] = deepmd_iterative_apath
config_json["system"] = sys_name
config_json["nb_nnp"] = int(nb_nnp)
config_json["current_iteration"] = 0
config_json["subsys_nr"] = {}
del sys_name, nb_nnp

for it0_subsys_nr,it_subsys_nr in enumerate(subsys_name):
    config_json["subsys_nr"][it_subsys_nr] = {}
    config_json["subsys_nr"][it_subsys_nr]["temperature_K"] = 300.0 if "temperature_K" not in globals() else temperature_K[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["timestep_ps"] = 0.0005 if "timestep_ps" not in globals() else timestep_ps[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["nb_candidates_max"] = 500 if "nb_candidates_max" not in globals() else nb_candidates_max[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["s_low"] = 0.1 if "s_low" not in globals() else s_low[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["s_high"] = 0.8 if "temperature" not in globals() else s_high[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["s_high_max"] = 1.0 if "s_high_max" not in globals() else s_high_max[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["ignore_first_x_ps"] = 0.5 if "ignore_first_x_ps" not in globals() else ignore_first_x_ps[it0_subsys_nr]
del it0_subsys_nr, it_subsys_nr, subsys_name

### Create the directories
cf.create_dir(training_iterative_apath+"/control")
cf.create_dir(training_iterative_apath+"/"+str(config_json["current_iteration"]).zfill(3)+"-training")

### Create the installation path file
with open(training_iterative_apath+"/control/path", "w") as f:
    f.write(deepmd_iterative_apath)
f.close()
del f, deepmd_iterative_apath

### Check if data exists, get init_* datasets and extract number of atoms and cell dimensions
cf.check_dir(training_iterative_apath+"/data",True,error_msg="data")
datasets_initial_folders = [ zzz for zzz in Path(training_iterative_apath+"/data").glob("init_*") ]

if len(datasets_initial_folders) == 0 :
    logging.critical("No initial data sets found.")
    logging.critical("Aborting...")
    sys.exit(1)

datasets_initial=[str(zzz).split("/")[-1] for zzz in datasets_initial_folders]
datasets_initial_json={}
for it_datasets_initial in datasets_initial:
    data_path = training_iterative_apath+"/data/"+it_datasets_initial
    cf.check_file(data_path+"/type.raw",0,True,"No type.raw found in "+ data_path)
    cf.check_file(data_path+"/set.000/box.npy",0,True,"No box.npy found in "+ data_path+"/set.000/")
    datasets_initial_json[it_datasets_initial] = np.load(data_path+"/set.000/box.npy").shape[0]
    cf.check_file(data_path+"/set.000/coord.npy",0,True,"No coord.npy found in "+ data_path+"/set.000/")
    cf.check_file(data_path+"/set.000/force.npy",0,True,"No force.npy found in "+ data_path+"/set.000/")
    cf.check_file(data_path+"/set.000/energy.npy",0,True,"No energy.npy found in "+ data_path+"/set.000/")
config_json["datasets_initial"]=[ zzz for zzz in datasets_initial_json.keys() ]
del it_datasets_initial,datasets_initial_folders,data_path,datasets_initial

### Print/Write config and datasets_initial json files
logging.info(config_json)
logging.info(datasets_initial_json)
cf.json_dump(config_json,config_json_fpath,print_log=True,name="configuration file")
cf.json_dump(datasets_initial_json,training_iterative_apath+"/control/datasets_initial.json",print_log=True,name="initial sets info file")

del config_json, config_json_fpath, training_iterative_apath, datasets_initial_json
del sys,np,Path,logging,cf
import gc; gc.collect(); del gc
exit()