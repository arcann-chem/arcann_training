## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name: str = "nvs"
# allocation_name: str = "v100"
# arch_name: str = "v100"
# slurm_email: str = ""
# cp2k_1_walltime_h: list = [0.5, 0.5]
# cp2k_2_walltime_h: list = [1.0, 1.0]
# nb_NODES: list = [1, 1]
# nb_MPI_per_NODE: list = [10, 10]
# nb_OPENMP_per_MPI: list = [1, 1]

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

exploration_json_fpath = training_iterative_apath+"/control/exploration_"+current_iteration_zfill+".json"
exploration_json = cf.json_read(exploration_json_fpath,True,True)

labeling_json_fpath = training_iterative_apath+"/control/labeling_"+current_iteration_zfill+".json"
labeling_json = cf.json_read(labeling_json_fpath,False,True)

### Checks
if exploration_json["is_extracted"] is False:
    logging.critical("Lock found. Run/Check first: exploration5_extract.py")
    logging.critical("Aborting...")
    sys.exit(1)

cluster = cf.check_cluster()

if cluster != "ir" and cluster != "jz":
    logging.critical("Unsupported cluster.")
    logging.critical("Aborting...")
    sys.exit(1)

### Set needed variables
labeling_json["cluster"] = cluster
labeling_json["project_name"] = project_name if "project_name" in globals() else "nvs"
labeling_json["allocation_name"] = allocation_name if "allocation_name" in globals() else "cpu"
labeling_json["arch_name"] = arch_name if "arch_name" in globals() else "cpu"
project_name = labeling_json["project_name"]
allocation_name = labeling_json["allocation_name"]
arch_name = labeling_json["arch_name"]
if arch_name == "cpu":
    arch_type ="cpu"
slurm_email = "" if "slurm_email" not in globals() else slurm_email

### Checks
cf.check_file(deepmd_iterative_apath+"/jobs/labeling/job_labeling_XXXXX_"+arch_type+"_"+cluster+".sh",True,True,"No SLURM file present for the labeling phase on this cluster.")
cf.check_file(deepmd_iterative_apath+"/jobs/labeling/job_labeling_array_"+arch_type+"_"+cluster+".sh",True,True,"No SLURM Array file present for the labeling phase on this cluster.")

### Preparation of the labeling
slurm_file_master = cf.read_file(deepmd_iterative_apath+"/jobs/labeling/job_labeling_XXXXX_"+arch_type+"_"+cluster+".sh")
slurm_file_master = cf.replace_in_list(slurm_file_master,"_R_PROJECT_",project_name)
slurm_file_master = cf.replace_in_list(slurm_file_master,"_R_ALLOC_",allocation_name)
slurm_file_array_master = cf.read_file(deepmd_iterative_apath+"/jobs/labeling/job_labeling_array_"+arch_type+"_"+cluster+".sh")
slurm_file_array_master = cf.replace_in_list(slurm_file_array_master,"_R_PROJECT_",project_name)
slurm_file_array_master = cf.replace_in_list(slurm_file_array_master,"_R_ALLOC_",allocation_name)
if slurm_email != "":
    if cluster == "jz":
        slurm_file_array_master = cf.replace_in_list(slurm_file_array_master,"##SBATCH --mail-type","#SBATCH --mail-type")
        slurm_file_array_master = cf.replace_in_list(slurm_file_array_master,"##SBATCH --mail-user _R_EMAIL_","#SBATCH --mail-user "+slurm_email)
    elif cluster == "ir":
        slurm_file_array_master = cf.replace_in_list(slurm_file_array_master,"##MSUB -@ _R_EMAIL_","##SUB -@ "+slurm_email)

labeling_json["subsys_nr"] = {}
subsys_list=list(config_json["subsys_nr"].keys())
for it0_subsys_nr, it_subsys_nr in enumerate(subsys_list):

    nb_candidates = int(exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_kept"])
    nb_candidates_disturbed = int(exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_kept"]) if exploration_json["subsys_nr"][it_subsys_nr]["disturbed_candidates"] is True else 0
    nb_steps = nb_candidates + nb_candidates_disturbed

    cf.create_dir(it_subsys_nr)
    cf.change_dir("./"+str(it_subsys_nr))

    labeling_json["subsys_nr"][it_subsys_nr] = {}
    labeling_json["subsys_nr"][it_subsys_nr]["cp2k_1_walltime_h"] = cp2k_1_walltime_h[it0_subsys_nr] if "cp2k_1_walltime_h" in globals() else 0.5
    labeling_json["subsys_nr"][it_subsys_nr]["cp2k_2_walltime_h"] = cp2k_2_walltime_h[it0_subsys_nr] if "cp2k_2_walltime_h" in globals() else 1.0
    labeling_json["subsys_nr"][it_subsys_nr]["nb_NODES"] = nb_NODES[it0_subsys_nr] if "nb_NODES" in globals() else 1
    labeling_json["subsys_nr"][it_subsys_nr]["nb_MPI_per_NODE"] = nb_MPI_per_NODE[it0_subsys_nr] if "nb_MPI_per_NODE" in globals() else 10
    labeling_json["subsys_nr"][it_subsys_nr]["nb_OPENMP_per_MPI"] = nb_OPENMP_per_MPI[it0_subsys_nr] if "nb_OPENMP_per_MPI" in globals() else 1

    slurm_file_subsys = cf.replace_in_list(slurm_file_master,"_R_nb_NODES_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_NODES"]))
    slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_nb_OPENMP_per_MPI_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_OPENMP_per_MPI"]))
    if cluster == "jz":
        slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_nb_MPI_per_NODE_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_MPI_per_NODE"]))
    elif cluster == "ir":
        slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_nb_MPI_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_MPI_per_NODE"] * labeling_json["subsys_nr"][it_subsys_nr]["nb_NODES"] ))

    slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_master,"_R_nb_NODES_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_NODES"]))
    slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_nb_OPENMP_per_MPI_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_OPENMP_per_MPI"]))
    if cluster == "jz":
            slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_nb_MPI_per_NODE_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_MPI_per_NODE"]))
    elif cluster == "ir":
         slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_nb_MPI_",str(labeling_json["subsys_nr"][it_subsys_nr]["nb_MPI_per_NODE"] * labeling_json["subsys_nr"][it_subsys_nr]["nb_NODES"] ))
    slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_CP2K_JOBNAME_","CP2K_"+it_subsys_nr+"_"+current_iteration_zfill)

    slurm_walltime_s = (labeling_json["subsys_nr"][it_subsys_nr]["cp2k_1_walltime_h"] + labeling_json["subsys_nr"][it_subsys_nr]["cp2k_2_walltime_h"]) * 3600
    slurm_walltime_s = int(slurm_walltime_s + 0.1 * slurm_walltime_s)

    if cluster == "jz":
        slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_WALLTIME_",cf.seconds_to_walltime(slurm_walltime_s))
        slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_WALLTIME_",cf.seconds_to_walltime(slurm_walltime_s))
        slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_ARRAYCOUNT_",str(nb_steps))
        cf.write_file("./job_labeling_array_"+arch_type+"_"+cluster+".sh",slurm_file_array_subsys)
        slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_WALLTIME_",cf.seconds_to_walltime(slurm_walltime_s))

    elif cluster == "ir":
        slurm_file_subsys = cf.replace_in_list(slurm_file_subsys,"_R_WALLTIME_",str(slurm_walltime_s))
        slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_WALLTIME_",str(slurm_walltime_s))
        if nb_steps <= 1000:
            if nb_steps <= 250:
                slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_ARRAY_START_",str(1))
                slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_ARRAY_END_",str(nb_steps))
                cf.write_file("./job_labeling_array_"+arch_type+"_"+cluster+"_0.sh",slurm_file_array_subsys)
            else:
                slurm_file_array_subsys_dict={}
                quotient = nb_steps // 250
                remainder = nb_steps % 250
                slurm_file_array_subsys = cf.replace_in_list(slurm_file_array_subsys,"_R_NEW_START_","0")
                for i in range(0,quotient+1):
                    if i < quotient:
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys,"_R_ARRAY_START_",str(250*i + 1))
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_ARRAY_END_",str(250 * (i+1)))
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_LAUNCHNEXT_","1")
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_NEXT_JOB_FILE_",str(i+1))
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_CD_WHERE_","${SLURM_SUBMIT_DIR}")
                        cf.write_file("./job_labeling_array_"+arch_type+"_"+cluster+"_"+str(i)+".sh",slurm_file_array_subsys_dict[str(i)])
                    else:
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys,"_R_ARRAY_START_",str(250*i + 1))
                        slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_ARRAY_END_",str(250*i + remainder ))
                        if it0_subsys_nr != len(config_json["subsys_nr"]) - 1:
                            slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_LAUNCHNEXT_","1")
                            slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_NEXT_JOB_FILE_","0")
                            slurm_file_array_subsys_dict[str(i)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(i)],"_R_CD_WHERE_","${SLURM_SUBMIT_DIR}/../"+subsys_list[it0_subsys_nr+1])
                        else:
                            True
                        cf.write_file("./job_labeling_array_"+arch_type+"_"+cluster+"_"+str(i)+".sh",slurm_file_array_subsys_dict[str(i)])
        else:
            slurm_file_array_subsys_dict={}
            quotient = nb_steps // 1000
            remainder = nb_steps % 1000
            m = 0
            for i in range(0,quotient+1):
                if i < quotient:
                    for j in range(0,4):
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys ,"_R_NEW_START_",str(i*1000))
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_START_",str(250*j + 1))
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_END_",str(250 * (j+1)))
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_LAUNCHNEXT_","1")
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_NEXT_JOB_FILE_",str(m+1))
                        slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_CD_WHERE_","${SLURM_SUBMIT_DIR}")
                        cf.write_file("./job_labeling_array_"+arch_type+"_"+cluster+"_"+str(m)+".sh",slurm_file_array_subsys_dict[str(m)])
                        m = m + 1
                else:
                    quotient2 = remainder // 250
                    remainder2 = remainder % 250
                    for j in range(0,quotient2+1):
                        if j < quotient2:
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys ,"_R_NEW_START_",str(i*1000))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_START_",str(250*j + 1))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_END_",str(250 * (j+1)))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_LAUNCHNEXT_","1")
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_NEXT_JOB_FILE_",str(m+1))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_CD_WHERE_","${SLURM_SUBMIT_DIR}")
                            cf.write_file("./job_labeling_array_"+arch_type+"_"+cluster+"_"+str(m)+".sh",slurm_file_array_subsys_dict[str(m)])
                            m = m + 1
                        else:
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys ,"_R_NEW_START_",str(i*1000))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_START_",str(250*j + 1))
                            slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_ARRAY_END_",str(250*j + remainder2))
                            if it0_subsys_nr != len(config_json["subsys_nr"]) - 1:
                                slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_LAUNCHNEXT_","1")
                                slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_NEXT_JOB_FILE_","0")
                                slurm_file_array_subsys_dict[str(m)] = cf.replace_in_list(slurm_file_array_subsys_dict[str(m)],"_R_CD_WHERE_","${SLURM_SUBMIT_DIR}/../"+subsys_list[it0_subsys_nr+1])
                            else:
                                True
                            cf.write_file("./job_labeling_array_"+arch_type+"_"+cluster+"_"+str(m)+".sh",slurm_file_array_subsys_dict[str(m)])
                            m = m + 1
            del slurm_file_array_subsys_dict

    xyz_file=training_iterative_apath+"/"+current_iteration_zfill+"-exploration/"+it_subsys_nr+"/candidates_"+str(it_subsys_nr)+"_"+current_iteration_zfill+".xyz"
    if Path(training_iterative_apath+"/"+current_iteration_zfill+"-exploration/"+it_subsys_nr+"/candidates_"+str(it_subsys_nr)+"_"+current_iteration_zfill+"_disturbed.xyz").is_file():
        xyz_file_disturbed = training_iterative_apath+"/"+current_iteration_zfill+"-exploration/"+it_subsys_nr+"/candidates_"+str(it_subsys_nr)+"_"+current_iteration_zfill+"_disturbed.xyz"

    cp2k_input_file_1 = cf.read_file(training_iterative_apath+"/inputs/1_"+str(it_subsys_nr)+"_labeling_XXXXX_"+cluster+".inp")
    cp2k_input_file_2 = cf.read_file(training_iterative_apath+"/inputs/2_"+str(it_subsys_nr)+"_labeling_XXXXX_"+cluster+".inp")

    cp2k_input_file_1 = cf.replace_in_list(cp2k_input_file_1,"_R_CELL_"," ".join([str(zzz) for zzz in config_json["subsys_nr"][it_subsys_nr]["cell"]]))
    cp2k_input_file_2 = cf.replace_in_list(cp2k_input_file_2,"_R_CELL_"," ".join([str(zzz) for zzz in config_json["subsys_nr"][it_subsys_nr]["cell"]]))
    cp2k_input_file_1 = cf.replace_in_list(cp2k_input_file_1,"_R_WALLTIME_",str(round(labeling_json["subsys_nr"][it_subsys_nr]["cp2k_1_walltime_h"],2) * 3600))
    cp2k_input_file_2 = cf.replace_in_list(cp2k_input_file_2,"_R_WALLTIME_",str(round(labeling_json["subsys_nr"][it_subsys_nr]["cp2k_2_walltime_h"],2) * 3600))

    n_atom, step_atoms, step_coordinates, blank = cf.import_xyz(xyz_file)

    for step_iter in range(1,step_atoms.shape[0]+1):
        step_iter_str = str(step_iter).zfill(5)
        cf.create_dir(step_iter_str)

        cp2k_input_file_t1 = cf.replace_in_list(cp2k_input_file_1,"XXXXX",step_iter_str)
        cp2k_input_file_t2 = cf.replace_in_list(cp2k_input_file_2,"XXXXX",step_iter_str)

        cf.write_file("./"+step_iter_str+"/1_labeling_"+step_iter_str+".inp",cp2k_input_file_t1)
        cf.write_file("./"+step_iter_str+"/2_labeling_"+step_iter_str+".inp",cp2k_input_file_t2)

        slurm_file = cf.replace_in_list(slurm_file_subsys,"XXXXX",step_iter_str)
        slurm_file = cf.replace_in_list(slurm_file,"_R_CP2K_JOBNAME_","CP2K_"+it_subsys_nr+"_"+current_iteration_zfill)

        cf.write_file("./"+step_iter_str+"/job_labeling_"+step_iter_str+"_"+arch_type+"_"+cluster+".sh",slurm_file)

        cf.write_xyz_from_index(step_iter-1,"./"+step_iter_str+"/labeling_"+step_iter_str+".xyz",n_atom,step_atoms,step_coordinates,blank)
        end_step = step_iter

    del n_atom, step_atoms, step_coordinates, blank, step_iter, step_iter_str
    labeling_json["subsys_nr"][it_subsys_nr]["candidates"] = end_step

    if Path(training_iterative_apath+"/"+current_iteration_zfill+"-exploration/"+it_subsys_nr+"/candidates_"+str(it_subsys_nr)+"_"+current_iteration_zfill+"_disturbed.xyz").is_file():
        n_atom, step_atoms, step_coordinates, blank = cf.import_xyz(xyz_file_disturbed)
        for d_step_iter in range(end_step+1,step_atoms.shape[0]+end_step+1):
            d_step_iter_str = str(d_step_iter).zfill(5)

            cf.create_dir(d_step_iter_str)

            cp2k_input_file_t1 = cf.replace_in_list(cp2k_input_file_1,"XXXXX",d_step_iter_str)
            cp2k_input_file_t2 = cf.replace_in_list(cp2k_input_file_2,"XXXXX",d_step_iter_str)

            cf.write_file("./"+d_step_iter_str+"/1_labeling_"+d_step_iter_str+".inp",cp2k_input_file_t1)
            cf.write_file("./"+d_step_iter_str+"/2_labeling_"+d_step_iter_str+".inp",cp2k_input_file_t2)

            slurm_file=cf.replace_in_list(slurm_file_subsys,"XXXXX",d_step_iter_str)
            slurm_file=cf.replace_in_list(slurm_file,"_R_CP2K_JOBNAME_","CP2K_"+it_subsys_nr+"_"+current_iteration_zfill)

            cf.write_file("./"+d_step_iter_str+"/job_labeling_"+d_step_iter_str+"_"+arch_type+"_"+cluster+".sh",slurm_file)

            cf.write_xyz_from_index(d_step_iter-end_step-1,"./"+d_step_iter_str+"/labeling_"+d_step_iter_str+".xyz",n_atom,step_atoms,step_coordinates,blank)

            end_step_disturbed=d_step_iter

        labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"] = d_step_iter-end_step
        del n_atom, step_atoms, step_coordinates, blank, d_step_iter, d_step_iter_str, end_step, end_step_disturbed
    else:
        labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"] = 0

    labeling_json["is_locked"] = True
    labeling_json["is_launched"] = False
    labeling_json["is_checked"] = False
    labeling_json["is_extracted"] = False

    cf.json_dump(labeling_json,labeling_json_fpath,True,"labeling.json")

    cf.change_dir("../")

logging.info("Preparation of the labeling is a success!")

del it0_subsys_nr, it_subsys_nr
del xyz_file, xyz_file_disturbed
del cp2k_input_file_1, cp2k_input_file_2, cp2k_input_file_t1, cp2k_input_file_t2
del slurm_file_master, slurm_file_subsys, slurm_file
del slurm_file_array_master, slurm_file_array_subsys

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del labeling_json, labeling_json_fpath
del exploration_json, exploration_json_fpath
del cluster, arch_type
del project_name, allocation_name, arch_name
del deepmd_iterative_apath
del slurm_email

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()
