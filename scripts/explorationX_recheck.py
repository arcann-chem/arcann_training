## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import numpy as np

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
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
exploration_json = cf.json_read((control_apath/("exploration_"+current_iteration_zfill+".json")),True,True)

### Checks
if "i-PI" not in exploration_json["exploration_type"]:
    logging.critical("This is not an i-PI exploration")
    logging.critical("Aborting...")
    sys.exit(1)
if not exploration_json["is_reruned"]:
    logging.critical("Lock found. Run/Check first: explorationX_rerun.py")
    logging.critical("Aborting...")
    sys.exit(1)

exploration_type = exploration_json['exploration_type']
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
            local_apath = Path(".").resolve()/str(it_subsys_nr)/str(it_nnp)/(str(it_each).zfill(5))
            lammps_output_file = local_apath/(str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+"_rerun.log")
            if lammps_output_file.is_file():
                lammps_output = cf.read_file(lammps_output_file)
                if any("Total wall time:" in f for f in lammps_output):
                    subsys_count = subsys_count + 1
                    check = check + 1
                    timings=[zzz for zzz in lammps_output if "Loop time of" in zzz]
                    timings_sum = timings_sum+float(timings[0].split(" ")[3])
                elif (local_apath/"skip").is_file():
                    skiped = skiped + 1
                    logging.warning(str(lammps_output_file)+" skipped")
                elif (local_apath/"force").is_file():
                    forced = forced + 1
                    logging.warning(str(lammps_output_file)+" forced")
                else:
                    logging.critical(str(lammps_output_file)+" failed. Check manually")
                del lammps_output
            elif (local_apath/"skip").is_file():
                    skiped = skiped + 1
                    logging.warning(str(lammps_output_file)+" skipped")
            else:
                logging.critical(str(lammps_output_file)+" not found. Check manually")
            del lammps_output_file
            del local_apath
    timings = timings_sum/subsys_count
    average_per_step = timings/exploration_json["subsys_nr"][it_subsys_nr]["nb_steps"]
    exploration_json["subsys_nr"][it_subsys_nr]["s_per_step_rerun"] = average_per_step
    del timings,average_per_step,subsys_count,timings_sum

del it_subsys_nr, it_nnp, it_each

if (check + skiped + forced) != (len( exploration_json["subsys_nr"]) * exploration_json["nb_nnp"] * exploration_json["nb_traj"] ):
    logging.critical("Some jobs failed or are still running.")
    logging.critical("Please check manually before relaunching this step")
    logging.critical("Or create files named \"skip\" or \"force\" to skip or force")
    logging.critical("Aborting...")
    sys.exit(1)
else:
    exploration_json["is_rechecked"] = True
    cf.json_dump(exploration_json,(control_apath/("exploration_"+current_iteration_zfill+".json")),True)
    for it_subsys_nr in exploration_json["subsys_nr"]:
        for it_nnp in range(1, exploration_json["nb_nnp"] + 1):
            for it_each in range(1, exploration_json["nb_traj"] + 1):
                local_apath = Path(".").resolve()/str(it_subsys_nr)/str(it_nnp)/(str(it_each).zfill(5))
                logging.info("Deleting SLURM out/error files...")
                cf.remove_file_glob(local_apath,"LAMMPS_*")
                logging.info("Deleting NNP PB files...")
                cf.remove_file_glob(local_apath,"*.pb")
                logging.info("Cleaning done!")
                del local_apath
            del it_each
        del it_nnp
    del it_subsys_nr
del check
if (skiped + forced) != 0:
    logging.warning(str(skiped)+" systems were skipped")
    logging.warning(str(forced)+" systems were forced")
del skiped, forced

logging.info("Exploration: Check phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath
del current_iteration_zfill
del exploration_json
del deepmd_iterative_apath

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()