#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Very fancy DeepMD-based semi-automatic highly-customizable iterative training procedure. #
#   Copyright 2022-2023 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#

## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Either shortcut (machine_file.json) or Project name / allocation / arch
# user_spec = "v100"
# user_spec = ["nvs","v100","v100"]
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

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
config_json = cf.json_read((control_apath/"config.json"),True,True)
jobs_apath = deepmd_iterative_apath/"jobs"/"training"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
training_json = cf.json_read((control_apath/("training_"+current_iteration_zfill+".json")),True,True)

### Checks
if training_json["deepmd_model_version"] < 2.0:
    logging.critical("No compression for model < 2.0 and your model is version:"+str(training_json["deepmd_model_version"]))
    logging.critical("Aborting...")
    sys.exit(1)
if not training_json["is_frozen"]:
    logging.critical("Lock found. Run/Check first: training5_checkfreeze.py")
    logging.critical("Aborting...")
    sys.exit(1)

### Read cluster info
if "user_spec" in globals():
    cluster, cluster_spec, cluster_error = cf.clusterize(deepmd_iterative_apath,training_iterative_apath,step="compressing",user_keyword=user_spec)
else:
    cluster, cluster_spec, cluster_error = cf.clusterize(deepmd_iterative_apath,training_iterative_apath,step="compressing")
if cluster_error != 0:
    ### #FIXME: Better errors for clusterize
    logging.critical("Error in machine_file.json")
    logging.critical("Aborting...")
    sys.exit(1)


cf.check_file(jobs_apath/("job_deepmd_compress_"+cluster_spec["arch_type"]+"_"+cluster+".sh"),True,True,"No SLURM file present for the compressing step on this cluster.")
slurm_file_master = cf.read_file(jobs_apath/("job_deepmd_compress_"+cluster_spec["arch_type"]+"_"+cluster+".sh"))
del jobs_apath

### Prep and launch DP Compress
check = 0
for it_nnp in range(1, config_json["nb_nnp"] + 1):
    local_apath = Path(".").resolve()/str(it_nnp)
    cf.check_file(local_apath/("graph_"+str(it_nnp)+"_"+current_iteration_zfill+".pb"),True,True)
    slurm_file = slurm_file_master
    slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_VERSION_",str(training_json["deepmd_model_version"]))
    slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_MODEL_","graph_"+str(it_nnp)+"_"+current_iteration_zfill)

    slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",cluster_spec["project_name"])
    slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",cluster_spec["allocation_name"])
    slurm_file = cf.delete_in_list(slurm_file,"_R_PARTITON_") if cluster_spec["partition"] is None else cf.replace_in_list(slurm_file,"_R_PARTITION_",cluster_spec["partition"])
    slurm_file = cf.delete_in_list(slurm_file,"_R_SUBPARTITION_") if cluster_spec["subpartition"] is None else cf.replace_in_list(slurm_file,"_R_SUBPARTITION_",cluster_spec["subpartition"])
    max_qos_time = 0
    max_qos = 0
    for it_qos in cluster_spec["qos"]:
        if cluster_spec["qos"][it_qos] >= 7200:
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
        slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_",cf.seconds_to_walltime(7200)) if cluster != "ir" else cf.replace_in_list(slurm_file,"_R_WALLTIME_",str(7200))
    del qos_ok, max_qos_time, max_qos
    if slurm_email != "":
        slurm_file = cf.replace_in_list(slurm_file,"_R_EMAIL_",slurm_email)
    else:
        slurm_file = cf.delete_in_list(slurm_file,"_R_EMAIL_")
        slurm_file = cf.delete_in_list(slurm_file,"mail")


    cf.write_file(local_apath/("job_deepmd_compress_"+cluster_spec["arch_type"]+"_"+cluster+".sh"),slurm_file)
    if (local_apath/("job_deepmd_compress_"+cluster_spec["arch_type"]+"_"+cluster+".sh")).is_file():
        cf.change_dir(local_apath)
        subprocess.call(["sbatch","./job_deepmd_compress_"+cluster_spec["arch_type"]+"_"+cluster+".sh"])
        cf.change_dir(local_apath.parent)
        logging.info("DP Compress - "+str(it_nnp)+" launched")
        check = check + 1
    else:
        logging.warning("DP Compress - "+str(it_nnp)+" NOT launched")
    del local_apath
del it_nnp, slurm_file, slurm_file_master

if check == config_json["nb_nnp"]:
    logging.info("DP-Freeze: SLURM phase is a success!")
else:
    logging.critical("Some DP Compress did not launched correctly")
    logging.critical("Please launch manually before continuing to the next step")
del check

### Cleaning
del config_json, training_iterative_apath, control_apath
del current_iteration, current_iteration_zfill
del training_json
del cluster, cluster_spec
del deepmd_iterative_apath
del slurm_email

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()