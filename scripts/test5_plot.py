#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Very fancy DeepMD-based semi-automatic highly-customizable iterative training procedure. #
#   Copyright 2022-2023 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#

## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name: str = "nvs"
# allocation_name: str = "dev"
# arch_name: str = "cpu"
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

slurm_email = "" if "slurm_email" not in globals() else slurm_email

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
jobs_apath = deepmd_iterative_apath/"jobs"/"test"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
config_json = cf.json_read((control_apath/"config.json"),True,True)
test_json = cf.json_read((control_apath/("test_"+current_iteration_zfill+".json")),True,True)
current_apath = Path(".").resolve()
scripts_apath = deepmd_iterative_apath/"tools"

### Remove previous obsolete slurm outputs
cf.remove_file_glob(current_apath,"DeepMD_Test_Concatenation.*")

### Checks
if not test_json["is_concatenated"]:
    logging.critical("Lock found. Run/Check first: test4_concatenation.py")
    logging.critical("Aborting...")
    sys.exit(1)


### Read cluster info
if "user_spec" in globals():
    cluster, cluster_spec, cluster_error = cf.clusterize(deepmd_iterative_apath,training_iterative_apath,step="test_graph",user_keyword=user_spec)
else:
    cluster, cluster_spec, cluster_error = cf.clusterize(deepmd_iterative_apath,training_iterative_apath,step="test_graph")
if cluster_error != 0:
    ### #FIXME: Better errors for clusterize
    logging.critical("Error in machine_file.json: "+str(cluster_error))
    logging.critical("Aborting...")
    sys.exit(1)

cf.check_file(jobs_apath/("job_deepmd_test_plot_"+cluster_spec["arch_type"]+"_"+cluster+".sh"),True,True,"No SLURM file present for the test_graph step on this cluster.")
slurm_file = cf.read_file(jobs_apath/("job_deepmd_test_plot_"+cluster_spec["arch_type"]+"_"+cluster+".sh"))
del jobs_apath

slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",cluster_spec["project_name"])
slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",cluster_spec["allocation_name"])
slurm_file = cf.delete_in_list(slurm_file,"_R_PARTITON_") if cluster_spec["partition"] is None else cf.replace_in_list(slurm_file,"_R_PARTITION_",cluster_spec["partition"])
slurm_file = cf.delete_in_list(slurm_file,"_R_SUBPARTITION_") if cluster_spec["subpartition"] is None else cf.replace_in_list(slurm_file,"_R_SUBPARTITION_",cluster_spec["subpartition"])

walltime_approx_s = 7200
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

cf.write_file(current_apath/("job_deepmd_test_plot_"+cluster_spec["arch_type"]+"_"+cluster+".sh"),slurm_file)
del slurm_file

cf.check_file(scripts_apath/"_deepmd_test_plot.py",True,True)
python_file = cf.read_file(scripts_apath/"_deepmd_test_plot.py")
python_file = cf.replace_in_list(python_file,"_DEEPMD_ITERATIVE_APATH_",str(deepmd_iterative_apath))
cf.write_file(current_apath/"_deepmd_test_plot.py",python_file)
del python_file
logging.info("The DP-Test: plot-prep phase is a success!")

subprocess.call(["sbatch",str(current_apath/("job_deepmd_test_plot_"+cluster_spec["arch_type"]+"_"+cluster+".sh"))])

cf.json_dump(test_json,(control_apath/("test_"+current_iteration_zfill+".json")),True)
logging.info("The DP-Test: plot-SLURM phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath, current_apath, scripts_apath
del test_json
del current_iteration, current_iteration_zfill
del cluster, cluster_spec
del deepmd_iterative_apath
del slurm_email

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()