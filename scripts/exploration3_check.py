###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

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

current_iteration = current_iteration if "current_iteration" in globals() else config_json["current_iteration"]
current_iteration_zfill = str(current_iteration).zfill(3)

exploration_json_fpath = training_iterative_apath+"/control/exploration_"+current_iteration_zfill+".json"
exploration_json = cf.json_read(exploration_json_fpath,True,True)

### Checks
if exploration_json["is_launched"] is False:
    logging.critical("Lock found. Run/Check first: exploration2_launch.py")
    logging.critical("Aborting...")
    sys.exit(1)

### Check the normal termination of the exploration phase
check = 0
skiped = 0
forced = 0
for it_subsys_nr in exploration_json["subsys_nr"]:
    average_per_step = 0
    subsys_count = 0
    timings_sum = 0
    timings=[]
    for it_nnp in range(1, exploration_json["nb_nnp"] + 1):
        for it_each in range(1, exploration_json["nb_traj"] + 1):
            it_each_zfill = str(it_each).zfill(5)
            check_path="./"+str(it_subsys_nr)+"/"+str(it_nnp)+"/"+it_each_zfill
            lammps_output_file = check_path+"/"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".log"
            if Path(lammps_output_file).is_file():
                lammps_output = cf.read_file(check_path+"/"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".log")
                if any("Total wall time:" in f for f in lammps_output):
                    subsys_count = subsys_count + 1
                    check = check + 1
                    timings=[zzz for zzz in lammps_output if "Loop time of" in zzz]
                    timings_sum = timings_sum+float(timings[0].split(" ")[3])
                elif ( Path(check_path+"/skip").is_file() ):
                    skiped = skiped + 1
                    logging.warning(lammps_output_file+" skipped")
                elif ( Path(check_path+"/force").is_file() ):
                    forced = forced + 1
                    logging.warning(lammps_output_file+" forced")
                else:
                    logging.critical(lammps_output_file+" failed. Check manually")
                del lammps_output
            elif ( Path(check_path+"/skip").is_file() ):
                    skiped = skiped + 1
                    logging.warning(lammps_output_file+" skipped")
            else:
                logging.critical(lammps_output_file+" not found. Check manually")
            del lammps_output_file, check_path
    timings = timings_sum/subsys_count
    average_per_step = timings/exploration_json["subsys_nr"][it_subsys_nr]["nb_steps"]
    exploration_json["subsys_nr"][it_subsys_nr]["avg_seconds_per_step"] = average_per_step
    del timings,average_per_step,subsys_count,timings_sum

del it_subsys_nr, it_nnp, it_each, it_each_zfill,  current_iteration_zfill

if (check + skiped + forced) != (len( exploration_json["subsys_nr"]) * exploration_json["nb_nnp"] * exploration_json["nb_traj"] ):
    logging.critical("Some jobs failed or are still running.")
    logging.critical("Please check manually before relaunching this step")
    logging.critical("Or create files named \"skip\" or \"force\" to skip or force")
    logging.critical("Aborting...")
    sys.exit(1)
else:
    exploration_json["is_checked"] = True
    cf.json_dump(exploration_json,exploration_json_fpath,True,"exploration.json")
    for it_subsys_nr in exploration_json["subsys_nr"]:
        for it_nnp in range(1, exploration_json["nb_nnp"] + 1):
            for it_each in range(1, exploration_json["nb_traj"] + 1):
                it_each_zfill = str(it_each).zfill(5)
                check_path="./"+str(it_subsys_nr)+"/"+str(it_nnp)+"/"+it_each_zfill
                cf.remove_file_glob(check_path+"/","LAMMPS_*")
                cf.remove_file_glob(check_path+"/","*.pb")
                del check_path, it_each_zfill
            del it_each
        del it_nnp
    del it_subsys_nr
del check, exploration_json, exploration_json_fpath
if (skiped + forced) != 0:
    logging.warning(str(skiped)+" systems were skipped")
    logging.warning(str(forced)+" systems were forced")
logging.info("The exploration phase is a success!")

### Cleaning
##TODO

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()