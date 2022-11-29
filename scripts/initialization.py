## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Set your system name, subsystem ("easy" exploration, standard TEMP, presents from the START of the iterative training) and the number of NNP you want to use
sys_name: str = "NAME"
subsys_name: list = ["SYSTEM1","SYSTEM2"]
## These are the default
# nb_nnp: int = 3
# exploration_type: str = "lammps"
# temperature_K: list = [298.15, 298.15]
# timestep_ps: list = [0.0005, 0.0005] #float #LAMMPS
# timestep_ps: list = [0.00025, 0.00025] #float #i-PI
# nb_candidates_max = [500, 500]
# s_low: list = [0.1, 0.1]
# s_high: list = [0.8, 0.8]
# s_high_max: list = [1.0, 1.0]
# ignore_first_x_ps: list = [0.5, 0.5]


###################################### No change past here
import sys
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

training_iterative_apath = Path().resolve()
### Check if the deepmd_iterative_apath is defined
deepmd_iterative_apath_error = 1
if "deepmd_iterative_apath" in globals():
    if (Path(deepmd_iterative_apath)/"tools"/"common_functions.py").is_file():
        deepmd_iterative_apath = Path(deepmd_iterative_apath)
        deepmd_iterative_apath_error = 0
elif (Path().home()/"deepmd_iterative_py"/"tools"/"common_functions.py").is_file():
    deepmd_iterative_apath = Path().home()/"deepmd_iterative_py"
    deepmd_iterative_apath_error = 0
if deepmd_iterative_apath_error == 1:
    logging.critical("Can\'t find common_functions.py in usual places:")
    logging.critical("deepmd_iterative_apath variable or ~/deepmd_iterative_py")
    logging.critical("Aborting...")
    sys.exit(1)
sys.path.insert(0, str(deepmd_iterative_apath/"tools"))
del deepmd_iterative_apath_error
import common_functions as cf

### Create the config.json
config_json = {}
#### These two arn't used anywhere, remove ?
config_json["training_iterative_apath"] = str(training_iterative_apath)
config_json["deepmd_iterative_apath"] = str(deepmd_iterative_apath)
####
config_json["system"] = sys_name
config_json["nb_nnp"] = 3 if "nb_nnp" not in globals() else nb_nnp
config_json["exploration_type"] = "lammps" if "exploration_type" not in globals() else exploration_type
config_json["current_iteration"] = 0
config_json["subsys_nr"] = {}
del sys_name

### Sets the default
for it0_subsys_nr,it_subsys_nr in enumerate(subsys_name):
    config_json["subsys_nr"][it_subsys_nr] = {}
    config_json["subsys_nr"][it_subsys_nr]["temperature_K"] = 298.15 if "temperature_K" not in globals() else temperature_K[it0_subsys_nr]
    if config_json["exploration_type"] == "lammps":
        config_json["subsys_nr"][it_subsys_nr]["timestep_ps"] = 0.0005 if "timestep_ps" not in globals() else timestep_ps[it0_subsys_nr]
    elif config_json["exploration_type"] == "i-PI":
        config_json["subsys_nr"][it_subsys_nr]["timestep_ps"] = 0.00025 if "timestep_ps" not in globals() else timestep_ps[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["nb_candidates_max"] = 500 if "nb_candidates_max" not in globals() else nb_candidates_max[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["s_low"] = 0.1 if "s_low" not in globals() else s_low[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["s_high"] = 0.8 if "temperature" not in globals() else s_high[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["s_high_max"] = 1.0 if "s_high_max" not in globals() else s_high_max[it0_subsys_nr]
    config_json["subsys_nr"][it_subsys_nr]["ignore_first_x_ps"] = 0.5 if "ignore_first_x_ps" not in globals() else ignore_first_x_ps[it0_subsys_nr]
del it0_subsys_nr, it_subsys_nr, subsys_name

### Create the control directory
control_apath = training_iterative_apath/"control"
control_apath.mkdir(exist_ok=True)
cf.check_dir(control_apath,True)

### Create the initial training directory
(training_iterative_apath/(str(config_json["current_iteration"]).zfill(3)+"-training")).mkdir(exist_ok=True)
cf.check_dir((training_iterative_apath/(str(config_json["current_iteration"]).zfill(3)+"-training")),True)

### Create the installation path file
(training_iterative_apath/"control"/"path").write_text(str(deepmd_iterative_apath))

### Check if data exists, get init_* datasets and extract number of atoms and cell dimensions
data_apath = training_iterative_apath/"data"
cf.check_dir(data_apath,True,error_msg="data")
initial_datasets_apath = [zzz for zzz in data_apath.glob("init_*")]
if len(initial_datasets_apath) == 0 :
    logging.critical("No initial data sets found.")
    logging.critical("Aborting...")
    sys.exit(1)
del data_apath

initial_datasets_json={}
for it_initial_datasets_apath in initial_datasets_apath:
    cf.check_file(it_initial_datasets_apath/"type.raw",True,True)
    it_initial_datasets_set_apath = (it_initial_datasets_apath/"set.000")
    for it_npy in ["box","coord","energy","force"]:
        cf.check_file(it_initial_datasets_set_apath/(it_npy+".npy"),True,True)
    del it_npy
    initial_datasets_json[it_initial_datasets_apath.name] = np.load(str(it_initial_datasets_set_apath/"box.npy")).shape[0]
del it_initial_datasets_apath, it_initial_datasets_set_apath
del initial_datasets_apath
config_json["initial_datasets"]=[ zzz for zzz in initial_datasets_json.keys() ]

### Print/Write config and datasets_initial json files
logging.info(config_json)
logging.info(initial_datasets_json)
cf.json_dump(config_json,(control_apath/"config.json"),True)
cf.json_dump(initial_datasets_json,(control_apath/"initial_datasets.json"),True)

logging.info("Initialization is a success!")

### Clean-up
del config_json
del control_apath
del initial_datasets_json
del deepmd_iterative_apath, training_iterative_apath

del sys,np,Path,logging,cf
import gc; gc.collect(); del gc
exit()