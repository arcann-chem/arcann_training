## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Either shortcut (machine_file.json) or Project name / allocation / arch
# user_spec = "v100"
# user_spec = ["nvs","v100","v100"]
# slurm_email: str = ""
## Training Parameters (Here are the default defaults)
# use_initial_datasets: bool = True
# use_extra_datasets: bool = False
# start_lr: float = 0.001
# stop_lr: float = 1e-06
# decay_rate: float = 0.90
# decay_steps: int = 5000
# numb_steps: int = 400000
# numb_test: int = 0
# deepmd_model_version: float = 2.1
# deepmd_model_type_descriptor: str = "se_e2_a"
## Guess for initial training walltime
# initial_seconds_per_1000steps: float = 90

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import subprocess
import numpy as np
import random

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

slurm_email = "" if "slurm_email" not in globals() else slurm_email

### Read the config file
control_apath = training_iterative_apath/"control"
config_json = cf.json_read((control_apath/"config.json"),True,True)
jobs_apath = deepmd_iterative_apath/"jobs"/"training"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)

### Read cluster info
if "user_spec" in globals():
    cluster, cluster_spec, cluster_error = cf.clusterize(deepmd_iterative_apath,training_iterative_apath,step="training",user_keyword=user_spec)
else:
    cluster, cluster_spec, cluster_error = cf.clusterize(deepmd_iterative_apath,training_iterative_apath,step="training")
if cluster_error != 0:
    ### #FIXME: Better errors for clusterize
    logging.critical("Error in machine_file.json")
    logging.critical("Aborting...")
    sys.exit(1)

if current_iteration > 0:
    labeling_json = cf.json_read((control_apath/("labeling_"+current_iteration_zfill+".json")),True,True)
    if not labeling_json["is_extracted"]:
        logging.critical("Lock found. Run/Check first: labeling4_extract.py")
        logging.critical("Aborting...")
        sys.exit(1)

### Get/Create training parameters
training_json = cf.json_read((control_apath/("training_"+current_iteration_zfill+".json")),False,True)
training_json["start_lr"] = 0.001 if "start_lr" not in globals() else start_lr
training_json["stop_lr"] = 1e-06 if "stop_lr" not in globals() else stop_lr
training_json["decay_rate"] = 0.90 if "decay_rate" not in globals() else decay_rate
training_json["decay_steps"] = 5000 if "decay_steps" not in globals() else decay_steps
training_json["numb_steps"] = 400000 if "numb_steps" not in globals() else numb_steps
training_json["numb_test"] = 0 if "numb_test" not in globals() else numb_test
training_json["use_initial_datasets"] = True if "use_initial_datasets" not in globals() else use_initial_datasets
training_json["use_extra_datasets"] = False if "use_extra_datasets" not in globals() else use_extra_datasets
training_json["deepmd_model_version"] = 2.1 if "deepmd_model_version" not in globals() else deepmd_model_version
training_json["deepmd_model_type_descriptor"] = "se_e2_a" if "deepmd_model_type_descriptor" not in globals() else deepmd_model_type_descriptor
training_json["cluster"] = cluster
training_json["project_name"] = cluster_spec["project_name"]
training_json["allocation_name"] = cluster_spec["allocation_name"]
training_json["arch_name"] = cluster_spec["arch_name"]
training_json["arch_type"] = cluster_spec["arch_type"]

cf.check_file(jobs_apath/("job_deepmd_train_"+cluster_spec["arch_type"]+"_"+cluster+".sh"),True,True,"No SLURM file present for the training step on this cluster.")
slurm_file_master = cf.read_file(jobs_apath/("job_deepmd_train_"+cluster_spec["arch_type"]+"_"+cluster+".sh"))
del jobs_apath

### Check DeePMD version
if training_json["deepmd_model_version"] not in [1.1, 1.3, 2.0, 2.1]:
    logging.critical("Invalid deepmd model version (1.1, 1.3, 2.0 or 2.1): "+str(training_json["deepmd_model_version"]))
    logging.critical("Aborting...")
    sys.exit(1)
### Check DeePMD descriptor type
if training_json["deepmd_model_type_descriptor"] not in ["se_a", "se_ar", "se_e2_a"]:
    logging.critical("Invalid deepmd type descriptor (se_a (se_e2_a) or se_ar: "+str(training_json["deepmd_model_type_descriptor"]))
    logging.critical("Aborting...")
    sys.exit(1)

### Check mismatch between cluster/arch_name/arch and DeePMD
if training_json["deepmd_model_version"] < 2.0:
    logging.critical("Only version >= 2.0 on Jean Zay!")
    logging.critical("Aborting...")
    sys.exit(1)
if training_json["deepmd_model_version"] < 2.1 and training_json["arch_name"] == "a100":
    logging.critical("Only version >= 2.1 on Jean Zay A100 !")
    logging.critical("Aborting...")
    sys.exit(1)

### Check mismatch between DeePMD version and Descriptor
if ((training_json["deepmd_model_type_descriptor"] == "se_a") and ( training_json["deepmd_model_version"] == 1.1 ))\
or ((training_json["deepmd_model_type_descriptor"] == "se_e2_a") and ( training_json["deepmd_model_version"] == 1.1 ))\
or ((training_json["deepmd_model_type_descriptor"] == "se_ar") and ( training_json["deepmd_model_version"] == 2.0 ))\
or ((training_json["deepmd_model_type_descriptor"] == "se_ar") and ( training_json["deepmd_model_version"] == 2.1 )):
    logging.critical("Invalid DeePMD Version/Descriptor pair: "+str(training_json["deepmd_model_version"])+"/"+str(training_json["deepmd_model_type_descriptor"]))
    logging.critical("Aborting...")
    sys.exit(1)

### Descriptor name equivalence
if ((training_json["deepmd_model_type_descriptor"] == "se_a") and ( training_json["deepmd_model_version"] == 2.0 ))\
    or ((training_json["deepmd_model_type_descriptor"] == "se_a") and ( training_json["deepmd_model_version"] == 2.1 )):
        training_json["deepmd_model_type_descriptor"] = "se_e2_a"
elif ((training_json["deepmd_model_type_descriptor"] == "se_e2_a") and ( training_json["deepmd_model_version"] == 1.3 )):
        training_json["deepmd_model_type_descriptor"] = "se_a"

### Check if the default input json file exists
input_file_fpath = (training_iterative_apath/"inputs"/(str(training_json["deepmd_model_version"])+"_"+str(training_json["deepmd_model_type_descriptor"])+".json")).resolve()
training_input_json = cf.json_read(input_file_fpath,True,True)
config_json["type_map"] = {}
config_json["type_map"] = training_input_json["model"]["type_map"]
del input_file_fpath

### Check the initial sets json file
datasets_initial_json = cf.check_initial_datasets(training_iterative_apath)

### Let us find what is in data
data_apath = training_iterative_apath/"data"
cf.check_dir(data_apath,True)
subsys_name=[]

### #TODO: IMPLEMENT TEST LIST FOR VALIDATION ? If DeepMD version >= 2.0

datasets_extra=[]
datasets_validation=[]
for it_data_folders in data_apath.iterdir():
    if it_data_folders.is_dir():
    ### Escape initial/extra sets, because initial get added first and extra as last, and also escape init_ not in initial_json (in case of removal)
        if it_data_folders.name not in datasets_initial_json.keys() and "extra_" != it_data_folders.name[:6] and "init_" != it_data_folders.name[:5]:
            ### Escape test sets
            if "test_" != it_data_folders.name[:5]:
                ### Escape if set iter is superior as iter, it is only for reprocessing old stuff.
                try:
                    if int(it_data_folders.name.rsplit("_",1)[-1]) <= current_iteration:
                        subsys_name.append(it_data_folders.name.rsplit("_",1)[0])
                except:
                    pass
            else:
                datasets_validation.append(it_data_folders.name)
        ### Get the extra sets !
        elif "extra_" == it_data_folders.name[:6]:
            datasets_extra.append(it_data_folders.name)
del it_data_folders

del datasets_validation

### Training sets list construction
datasets_training=[]
datasets_training_json=[]
### Initial
nb_initial = 0
if training_json["use_initial_datasets"]:
    for it_datasets_initial_json in datasets_initial_json.keys():
        if (data_apath/it_datasets_initial_json).is_dir():
            ### #TODO: Here we don't Path because too complex
            datasets_training.append("data/"+it_datasets_initial_json+"/")
            datasets_training_json.append(it_datasets_initial_json)
            nb_initial = nb_initial+datasets_initial_json[it_datasets_initial_json]
    del it_datasets_initial_json
del datasets_initial_json

### Non-Reactive (aka subsys_nr in the initialization first) && all the others are REACTIVE !
### Total and what is added just for this iteration
nb_added_nr = 0
nb_added_r = 0
nb_added_nr_iter = 0
nb_added_r_iter = 0

### This trick remove duplicates from list via set
subsys_name = list(set(subsys_name))
subsys_name = [i for i in subsys_name if i not in config_json["subsys_nr"]]
subsys_name = [i for i in subsys_name if i not in [zzz + "-disturbed" for zzz in config_json["subsys_nr"]]]
subsys_name = sorted(subsys_name)
config_json["subsys_r"] = subsys_name
del subsys_name

if current_iteration > 0:
    for it_iteration in np.arange(1,current_iteration+1):
        try:
            for system_it in config_json["subsys_nr"]:
                if (data_apath/(system_it+"_"+str(it_iteration).zfill(3))).is_dir():
                    ### #TODO: Here we don't Path because too complex
                    datasets_training.append("data/"+system_it+"_"+str(it_iteration).zfill(3)+"/")
                    datasets_training_json.append(system_it+"_"+str(it_iteration).zfill(3))
                    nb_added_nr = nb_added_nr+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
                    if it_iteration == current_iteration:
                        nb_added_nr_iter = nb_added_nr_iter+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
            del system_it
        except(KeyError,NameError):
            pass
        try:
            for system_it in [zzz + "-disturbed" for zzz in config_json["subsys_nr"]]:
                if (data_apath/(system_it+"_"+str(it_iteration).zfill(3))).is_dir():
                    ### #TODO: Here we don't Path because too complex
                    datasets_training.append("data/"+system_it+"_"+str(it_iteration).zfill(3)+"/")
                    datasets_training_json.append(system_it+"_"+str(it_iteration).zfill(3))
                    nb_added_nr = nb_added_nr+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
                    if it_iteration == current_iteration:
                        nb_added_nr_iter = nb_added_nr_iter+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
            del system_it
        except(KeyError,NameError):
            pass
        try:
            for system_it in config_json["subsys_r"]:
                if (data_apath/(system_it+"_"+str(it_iteration).zfill(3))).is_dir():
                    ### #TODO: Here we don't Path because too complex
                    datasets_training.append("data/"+system_it+"_"+str(it_iteration).zfill(3)+"/")
                    datasets_training_json.append(system_it+"_"+str(it_iteration).zfill(3))
                    nb_added_nr = nb_added_nr+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
                    if it_iteration == current_iteration:
                        nb_added_nr_iter = nb_added_nr_iter+np.load(str(data_apath/(system_it+"_"+str(it_iteration).zfill(3))/"set.000"/"box.npy")).shape[0]
            del system_it
        except(KeyError,NameError):
            pass
    del it_iteration

### Finally the extra sets !
nb_extra = 0
if training_json["use_extra_datasets"]:
    config_json["datasets_extra"] = datasets_extra
    del datasets_extra
    for it_datasets_extra in config_json["datasets_extra"]:
        ### #TODO: Here we don't Path because too complex
        datasets_training.append("data/"+it_datasets_extra+"/")
        datasets_training_json.append(it_datasets_extra)
        nb_extra = nb_extra+np.load(str(data_apath/it_datasets_extra/"set.000"/"box.npy")).shape[0]
    del it_datasets_extra
else:
    del datasets_extra

### Total
nb_trained = nb_initial+nb_added_nr+nb_added_r+nb_extra

### Number of tests
if ( training_json["deepmd_model_version"] < 2.0 ):
    training_input_json["training"]["numb_test"] = training_json["numb_test"]

### #TODO: Auto validation: If there is validation/test sets for 2.0, maybe enforce numb_test to not 0??

### Because changes between version
if ( training_json["deepmd_model_version"] >= 2.0 ):
    training_input_json["training"]["training_data"]["systems"] = datasets_training
else:
    training_input_json["training"]["systems"] = datasets_training

training_json["training_data"] = datasets_training_json
training_json["nb_trained"] = nb_trained
training_json["nb_initial"] = nb_initial
training_json["nb_added_nr"] = nb_added_nr
training_json["nb_added_r"] = nb_added_r
training_json["nb_added_nr_iter"] = nb_added_nr_iter
training_json["nb_added_r_iter"] = nb_added_r_iter
training_json["nb_extra"] = nb_extra

del datasets_training_json
del nb_trained, nb_initial, nb_extra
del nb_added_nr, nb_added_r, nb_added_nr_iter, nb_added_r_iter

### If no override, get decay steps (= nb of trained floored to the nearest 10000 divided by 4)
if "decay_steps" not in globals():
    decay_steps = cf.get_decay_steps(training_json["nb_trained"])

training_json["decay_steps"] = int(decay_steps)
decay_steps = int(decay_steps)

### THE MAGIC IS HERE
### Priority is: GOOD LUCK

### If the decay_rate is overridden and stop_lr is not
if "decay_rate" in globals() and "stop_lr" not in globals():
    ### Here: playing with stop_lr and numb_steps
    ### Calculate the new stop_lr
    stop_lr_new = cf.get_learning_rate(training_json["numb_steps"],training_json["start_lr"],decay_rate,training_json["decay_steps"])
    ### If numb_steps was not overridden, recalculate stop_lr and augment numb_steps if needed to approach the default stop_lr (going up)
    if "numb_steps" not in globals():
        while stop_lr_new > training_json["stop_lr"]:
            numb_steps = numb_steps+1e5
            stop_lr_new = cf.get_learning_rate(numb_steps,training_json["start_lr"],decay_rate,training_json["decay_steps"])
        training_json["numb_steps"] = int(numb_steps)
    training_json["stop_lr"] = stop_lr_new

### If the decay_rate is overridden, as well as stop_lr and numb_steps
elif "decay_rate" in globals() and "stop_lr" in globals() and "numb_steps" in globals():
    ### Here: playing with stop_lr, decay_steps, and decay_rate.
    stop_lr_new = cf.get_learning_rate(numb_steps,training_json["start_lr"],decay_rate,decay_steps)
    if stop_lr_new > stop_lr:
        while stop_lr_new > stop_lr:
            decay_steps = decay_steps-1000
            stop_lr_new = cf.get_learning_rate(numb_steps,training_json["start_lr"],decay_rate,decay_steps)
    else:
        while stop_lr_new < stop_lr:
            decay_steps = decay_steps+1000
            stop_lr_new = cf.get_learning_rate(numb_steps,training_json["start_lr"],decay_rate,decay_steps)
    training_json["decay_steps"] = int(decay_steps)
    decay_rate_new = cf.get_decay_rate(numb_steps,training_json["start_lr"],stop_lr,training_json["decay_steps"])
    training_json["decay_rate"] = decay_rate_new
    del decay_rate_new
### Default case
else:
    ### Here: playing with decay_rate and numb_steps
    ### Overwrite the stop_lr
    if "stop_lr" not in globals():
        stop_lr = training_json["stop_lr"]
    ### Recalculate the decay_rate to be as close as the target (inf), and augment the numb_steps as needed
    numb_steps = training_json["numb_steps"]
    decay_rate_new = cf.get_decay_rate(numb_steps,training_json["start_lr"],stop_lr,training_json["decay_steps"])
    while decay_rate_new < training_json["decay_rate"]:
        numb_steps = numb_steps+1e5
        decay_rate_new = cf.get_decay_rate(numb_steps,training_json["start_lr"],stop_lr,training_json["decay_steps"])
    training_json["numb_steps"] = int(numb_steps)
    training_json["decay_rate"] = decay_rate_new
    del decay_rate_new

del decay_steps, stop_lr

if ( training_json["deepmd_model_version"] >= 2.0 ):
    training_input_json["training"]["numb_steps"] = training_json["numb_steps"]
else:
    training_input_json["training"]["stop_batch"] = training_json["numb_steps"]

training_input_json["learning_rate"]["decay_steps"] = training_json["decay_steps"]

if (training_json["deepmd_model_version"] >= 1.3):
    training_input_json["learning_rate"]["stop_lr"] = training_json["stop_lr"]
else:
    training_input_json["learning_rate"]["decay_rate"] = training_json["decay_rate"]

### Set frozen/compressed bool !
training_json["is_locked"] = True
training_json["is_launched"] = False
training_json["is_checked"] = False
training_json["is_frozen"] = False
training_json["is_compressed"] = False

logging.info(training_json)
logging.info(datasets_training)

### Rsync data to local data
localdata_apath = Path(".").resolve()/"data"
localdata_apath.mkdir(exist_ok=True)
for it_datasets_training in datasets_training:
    subprocess.call(["rsync","-a", str(training_iterative_apath)+"/"+it_datasets_training.rsplit("/",1)[0], str(localdata_apath)])
del it_datasets_training, localdata_apath, datasets_training

### Change some inside output
training_input_json["training"]["disp_file"]="lcurve.out"
training_input_json["training"]["save_ckpt"]="model.ckpt"

### It doesn"t exists anymore :(
if training_json["deepmd_model_version"] < 2.0:
    training_input_json["training"]["load_ckpt"]="model.ckpt"

### Create the inputs/jobfiles for each NNP with random SEED inf the form of NNP_number + random(0,1000) + current_iteration.zfill(3) so between 10000 and unlimited1000999 (at iteration 999 !!)
if current_iteration > 0:
    previous_iteration = current_iteration - 1
    previous_iteration_zfill = str(previous_iteration).zfill(3)
    prevtraining_json = cf.json_read((control_apath/("training_"+previous_iteration_zfill+".json")),True,True)
    walltime_approx_s = int(np.ceil((numb_steps*(prevtraining_json["s_per_step"]*1.50))))
    del previous_iteration, previous_iteration_zfill, prevtraining_json
else:
    initial_seconds_per_1000steps = 90 if "initial_seconds_per_1000steps" not in globals() else initial_seconds_per_1000steps
    walltime_approx_s = int(np.ceil((numb_steps*initial_seconds_per_1000steps/1000)))
del numb_steps

for it_nnp in range(1,config_json["nb_nnp"] + 1):
    local_apath = Path(".").resolve()/str(it_nnp)
    local_apath.mkdir(exist_ok=True)
    cf.check_dir(local_apath,True)

    random.seed()
    RAND = random.randrange(0,1000)
    if training_json["deepmd_model_type_descriptor"] == "se_ar":
        training_input_json["model"]["descriptor"]["a"]["seed"] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)
        training_input_json["model"]["descriptor"]["r"]["seed"] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)
    else:
        training_input_json["model"]["descriptor"]["seed"] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)

    training_input_json["model"]["fitting_net"]["seed"] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)

    training_input_json["training"]["seed"] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)

    training_input_json_fpath = Path(str(it_nnp)+"/training.json").resolve()
    cf.json_dump(training_input_json,training_input_json_fpath,False)

    slurm_file = slurm_file_master.copy()
    slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_VERSION_",str(training_json["deepmd_model_version"]))

    slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",cluster_spec["project_name"])
    slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",cluster_spec["allocation_name"])
    slurm_file = cf.delete_in_list(slurm_file,"_R_PARTITON_") if cluster_spec["partition"] is None else cf.replace_in_list(slurm_file,"_R_PARTITION_",cluster_spec["partition"])
    slurm_file = cf.delete_in_list(slurm_file,"_R_SUBPARTITION_") if cluster_spec["subpartition"] is None else cf.replace_in_list(slurm_file,"_R_SUBPARTITION_",cluster_spec["subpartition"])
    max_qos_time = 0
    max_qos = 0
    for it_qos in cluster_spec["qos"]:
        if cluster_spec["qos"][it_qos] >= walltime_approx_s:
            slurm_file = cf.replace_in_list(slurm_file,"_R_QOS_",it_qos)
            qos_ok = True
        else:
            max_qos = it_qos if cluster_spec["qos"][it_qos] > max_qos_time else max_qos
            qos_ok = False
    del it_qos
    if not qos_ok:
        logging.warning("Approximate wall time superior than the maximun time allowed by the QoS")
        logging.warning("Settign the maximum QoS time as walltime")
        slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_",cf.seconds_to_walltime(max_qos_time)) if cluster != "ir" else cf.replace_in_list(slurm_file,"_R_WALLTIME_",str(max_qos_time))
    else:
        slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_",cf.seconds_to_walltime(walltime_approx_s)) if cluster != "ir" else cf.replace_in_list(slurm_file,"_R_WALLTIME_",str(walltime_approx_s))
    del qos_ok, max_qos_time, max_qos

    if slurm_email != "":
        slurm_file = cf.replace_in_list(slurm_file,"_R_EMAIL_",slurm_email)
    else:
        slurm_file = cf.delete_in_list(slurm_file,"_R_EMAIL_")
        slurm_file = cf.delete_in_list(slurm_file,"mail")

    cf.write_file(local_apath/("job_deepmd_train_"+cluster_spec["arch_type"]+"_"+cluster+".sh"),slurm_file)
    del slurm_file, local_apath, training_input_json_fpath, RAND
del it_nnp, walltime_approx_s, training_input_json

## Dump the config/training
cf.json_dump(config_json,(control_apath/"config.json"),True)
cf.json_dump(training_json,(control_apath/("training_"+current_iteration_zfill+".json")),True)

if "initial_seconds_per_1000steps" in globals():
    del initial_seconds_per_1000steps

logging.info("DP-Train: Prep phase is a success!")

### Cleaning
del data_apath, control_apath
del config_json, training_iterative_apath
del current_iteration, current_iteration_zfill
del training_json
del cluster, cluster_spec
del slurm_email
del slurm_file_master
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess, np, random
import gc; gc.collect(); del gc
exit()