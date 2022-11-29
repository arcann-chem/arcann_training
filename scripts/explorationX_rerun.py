## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## These are the default
atomsk_fpath: str ="/gpfswork/rech/nvs/commun/programs/apps/atomsk/0.11.2/bin/atomsk"
# vmd_fpath: str=""

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
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
exploration_json = cf.json_read((control_apath/("exploration_"+current_iteration_zfill+".json")),True,True)
scripts_apath = deepmd_iterative_apath/"tools"
jobs_apath = deepmd_iterative_apath/"jobs"/"exploration"

previous_iteration_zfill = str(current_iteration-1).zfill(3)
prevtraining_json = cf.json_read((control_apath/("training_"+previous_iteration_zfill+".json")),True,True)

### Checks
if "i-PI" not in exploration_json["exploration_type"]:
    logging.critical("This is not an i-PI exploration")
    logging.critical("Aborting...")
    sys.exit(1)
if not exploration_json["is_unbeaded"]:
    logging.critical("Lock found. Run/Check first: explorationX_selectbeads.py")
    logging.critical("Aborting...")
    sys.exit(1)

### Read cluster info
if "user_spec" in globals():
    cluster, cluster_spec, cluster_error = cf.clusterize(deepmd_iterative_apath,training_iterative_apath,step="exploration",user_keyword=user_spec)
else:
    cluster, cluster_spec, cluster_error = cf.clusterize(deepmd_iterative_apath,training_iterative_apath,step="exploration")
if cluster_error != 0:
    ### #FIXME: Better errors for clusterize
    logging.critical("Error in machine_file.json: "+str(cluster_error))
    logging.critical("Aborting...")
    sys.exit(1)

master_lammps_in = cf.read_file(scripts_apath/"rerun.in")
slurm_file_master = cf.read_file(jobs_apath/("job_deepmd_lammps_"+cluster_spec["arch_type"]+"_"+cluster+".sh"))
check = 0
skipped = 0
for it0_subsys_nr,it_subsys_nr in enumerate(config_json["subsys_nr"]):

    subsys_walltime_approx_s = 3600
    subsys_lammps_data_fn = it_subsys_nr+".lmp"
    subsys_lammps_data = cf.read_file(training_iterative_apath/"inputs"/subsys_lammps_data_fn)
    subsys_exploration_lammps_input = cf.replace_in_list(master_lammps_in,"_R_DATA_FILE_",subsys_lammps_data_fn)

    for it_nnp in range(1, exploration_json["nb_nnp"] + 1):
        for it_each in range(1, exploration_json["nb_traj"] + 1):

            local_apath = Path(".").resolve()/str(it_subsys_nr)/str(it_nnp)/(str(it_each).zfill(5))

            if not (local_apath/"skip").is_file():

                ### Get the model list
                list_nnp = [zzz for zzz in range(1, config_json["nb_nnp"] + 1)]
                reorder_nnp_list = list_nnp[list_nnp.index(it_nnp):] + list_nnp[:list_nnp.index(it_nnp)]
                compress_str = "_compressed" if prevtraining_json["is_compressed"] else ""
                models_list=["graph_"+str(f)+"_"+previous_iteration_zfill+compress_str+".pb" for f in reorder_nnp_list]
                for it_sub_nnp in range(1, config_json["nb_nnp"] + 1 ):
                    nnp_apath = (training_iterative_apath/"NNP"/("graph_"+str(it_sub_nnp)+"_"+previous_iteration_zfill+compress_str+".pb")).resolve()
                    if not (local_apath/ ("graph_"+str(it_sub_nnp)+"_"+previous_iteration_zfill+compress_str+".pb")).is_file():
                        subprocess.call(["ln","-nsf", str(nnp_apath), str(local_apath)])
                models_string=" ".join(models_list)
                del list_nnp, it_sub_nnp, nnp_apath, compress_str, reorder_nnp_list

                exploration_input = subsys_exploration_lammps_input.copy()
                exploration_input = cf.replace_in_list(exploration_input,"_R_DCD_OUT_",str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".dcd")
                exploration_input = cf.replace_in_list(exploration_input,"_R_MODELS_LIST_",models_string)
                exploration_input = cf.replace_in_list(exploration_input,"_R_DEVI_OUT_","model_devi_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".out")
                exploration_input = cf.replace_in_list(exploration_input,"_R_XYZ_IN_","beads_rerun_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".xyz")

                ### Write DATA file
                cf.write_file(local_apath/subsys_lammps_data_fn,subsys_lammps_data)

                ### Write INPUT file
                cf.write_file(local_apath/(str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+"_rerun.in"),exploration_input)

                ### Now SLURM file
                slurm_file = slurm_file_master.copy()
                slurm_file = cf.replace_in_list(slurm_file,"_R_DEEPMD_VERSION_",str(exploration_json["deepmd_model_version"]))
                slurm_file = cf.replace_in_list(slurm_file,"_R_INPUT_",str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+"_rerun")
                slurm_file = cf.replace_in_list(slurm_file,"_R_DATA_FILE_",subsys_lammps_data_fn)
                slurm_file = cf.replace_in_list(slurm_file,"_R_XYZ_IN_","beads_rerun_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".xyz")

                slurm_file = cf.replace_in_list(slurm_file,"_R_PROJECT_",cluster_spec["project_name"])
                slurm_file = cf.replace_in_list(slurm_file,"_R_ALLOC_",cluster_spec["allocation_name"])
                slurm_file = cf.delete_in_list(slurm_file,"_R_PARTITON_") if cluster_spec["partition"] is None else cf.replace_in_list(slurm_file,"_R_PARTITION_",cluster_spec["partition"])
                slurm_file = cf.delete_in_list(slurm_file,"_R_SUBPARTITION_") if cluster_spec["subpartition"] is None else cf.replace_in_list(slurm_file,"_R_SUBPARTITION_",cluster_spec["subpartition"])
                max_qos_time = 0
                max_qos = 0
                for it_qos in cluster_spec["qos"]:
                    if cluster_spec["qos"][it_qos] >= subsys_walltime_approx_s:
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
                    slurm_file = cf.replace_in_list(slurm_file,"_R_WALLTIME_",cf.seconds_to_walltime(subsys_walltime_approx_s)) if cluster != "ir" else cf.replace_in_list(slurm_file,"_R_WALLTIME_",str(subsys_walltime_approx_s))
                del qos_ok, max_qos_time, max_qos
                if slurm_email != "":
                    slurm_file = cf.replace_in_list(slurm_file,"_R_EMAIL_",slurm_email)
                else:
                    slurm_file = cf.delete_in_list(slurm_file,"_R_EMAIL_")
                    slurm_file = cf.delete_in_list(slurm_file,"mail")

                slurm_file = cf.replace_in_list(slurm_file," \"_R_PLUMED_FILES_LIST_\"","")
                models_list_job = models_string.replace(" ","\" \"")
                slurm_file = cf.replace_in_list(slurm_file, "_R_MODELS_LIST_", models_list_job)

                cf.write_file(local_apath/("job_deepmd_lammps_"+cluster_spec["arch_type"]+"_"+cluster+".sh"),slurm_file)

                if (local_apath/("job_deepmd_lammps_"+cluster_spec["arch_type"]+"_"+cluster+".sh")).is_file():
                    cf.change_dir(local_apath)
                    subprocess.call(["sbatch","./job_deepmd_lammps_"+cluster_spec["arch_type"]+"_"+cluster+".sh"])
                    cf.change_dir(((local_apath.parent).parent).parent)
                    logging.info("Rerun - "+str(it_subsys_nr)+"/"+str(it_nnp)+"/"+str(it_each).zfill(5)+" launched")
                    check = check + 1
                else:
                    logging.info("Rerun - "+str(it_subsys_nr)+"/"+str(it_nnp)+"/"+str(it_each).zfill(5)+" NOT launched")
                del local_apath
            else:
                skipped = skipped + 1
        del it_each
    del it_nnp
del it_subsys_nr

if (check + skipped) == (len( exploration_json["subsys_nr"]) * exploration_json["nb_nnp"] * exploration_json["nb_traj"] ):
    exploration_json["is_reruned"] = True
    cf.json_dump(exploration_json,(control_apath/("exploration_"+current_iteration_zfill+".json")),True)
    logging.info("Exploration: Rerun+Launch phase is a success!")
else:
    logging.critical("Some Rerun did not launched correctly")
    logging.critical("Please launch manually before continuing to the next step")
    logging.critical("And replace the key \"is_reruned\" to True in the corresponding exploration.json.")
del check

### Cleaning
del config_json, training_iterative_apath, scripts_apath, control_apath
del current_iteration_zfill
del exploration_json
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()