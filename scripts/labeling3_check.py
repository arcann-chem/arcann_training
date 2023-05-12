## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

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
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
config_json = cf.json_read((control_apath/"config.json"),True,True)
labeling_json = cf.json_read((control_apath/("labeling_"+current_iteration_zfill+".json")),True,True)

### Checks
if not labeling_json["is_launched"]:
    logging.critical("Lock found. Run/Check first: labeling2_launch.py")
    logging.critical("Aborting...")
    sys.exit(1)

### Check the normal termination of the labeling phase
total_steps = 0
step_1 = 0
step_2 = 0
skipped = 0

for it_subsys_nr in labeling_json["subsys_nr"]:
    total_steps = total_steps + labeling_json["subsys_nr"][it_subsys_nr]["candidates"] + labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"]
    average_per_step = 0
    timings_sum_1 = 0
    timings_1 = []
    timings_sum_2 = 0
    timings_2 = []
    not_converged_list_1 = []
    not_converged_list_2 = []
    failed_list_1 = []
    failed_list_2 = []
    skipped_list = []
    skipped_subsys_candidates = 0
    skipped_subsys_disturbed = 0

    for it_step in range(1, labeling_json["subsys_nr"][it_subsys_nr]["candidates"] + 1):
        it_step_zfill = str(it_step).zfill(5)
        local_apath = Path(".").resolve()/it_subsys_nr/it_step_zfill

        if (local_apath/"skip").is_file():
            # configuration skipped
            skipped += 1
            skipped_subsys_candidates += 1
            skipped_list.append(str(local_apath)+"\n")
        else:
            cp2k_output_file_1 = local_apath/("1_labeling_"+it_step_zfill+".out")
            if cp2k_output_file_1.is_file():
                cp2k_output_1 = cf.read_file(cp2k_output_file_1)
                if any("SCF run converged in " in f for f in cp2k_output_1):
                    step_1 = step_1 + 1
                    timings_1 = [zzz for zzz in cp2k_output_1 if "CP2K                                 1  1.0" in zzz]
                    timings_sum_1 = timings_sum_1 + float(timings_1[0].split(" ")[-1])
                elif any("SCF run NOT converged in " in f for f in cp2k_output_1):
                    not_converged_list_1.append(str(local_apath/("1_labeling_"+it_step_zfill+".out"))+"\n")
                else:
                    failed_list_1.append(str(local_apath/("1_labeling_"+it_step_zfill+".out"))+"\n")
                del cp2k_output_1
            else:
                failed_list_1.append(str(local_apath/("1_labeling_"+it_step_zfill+".out"))+"\n")
            del cp2k_output_file_1

            cp2k_output_file_2 = local_apath/("2_labeling_"+it_step_zfill+".out")
            if cp2k_output_file_2.is_file():
                cp2k_output_2 = cf.read_file(cp2k_output_file_2)
                if any("SCF run converged in " in f for f in cp2k_output_2):
                    step_2 = step_2 + 1
                    timings_2 = [zzz for zzz in cp2k_output_2 if "CP2K                                 1  1.0" in zzz]
                    timings_sum_2 = timings_sum_2 + float(timings_2[0].split(" ")[-1])
                elif any("SCF run NOT converged in " in f for f in cp2k_output_2):
                    not_converged_list_2.append(str(local_apath/("2_labeling_"+it_step_zfill+".out"))+"\n")
                else:
                    failed_list_2.append(str(local_apath/("2_labeling_"+it_step_zfill+".out"))+"\n")
                del cp2k_output_2
            else:
                failed_list_2.append(str(local_apath/("2_labeling_"+it_step_zfill+".out"))+"\n")
            del cp2k_output_file_2
    
    for it_step in range(labeling_json["subsys_nr"][it_subsys_nr]["candidates"] + 1, labeling_json["subsys_nr"][it_subsys_nr]["candidates"] + labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"] + 1):
        it_step_zfill = str(it_step).zfill(5)
        local_apath = Path(".").resolve()/str(it_subsys_nr)/it_step_zfill

        if (local_apath/"skip").is_file():
            # configuration skipped
            skipped += 1
            skipped_subsys_disturbed += 1
            skipped_list.append(str(local_apath)+"\n")
        else:
            cp2k_output_file_1 = local_apath/("1_labeling_"+it_step_zfill+".out")
            if cp2k_output_file_1.is_file():
                cp2k_output_1 = cf.read_file(cp2k_output_file_1)
                if any("SCF run converged in " in f for f in cp2k_output_1):
                    step_1 = step_1 + 1
                    timings_1 = [zzz for zzz in cp2k_output_1 if "CP2K                                 1  1.0" in zzz]
                    timings_sum_1 = timings_sum_1 + float(timings_1[0].split(" ")[-1])
                elif any("SCF run NOT converged in " in f for f in cp2k_output_1):
                    not_converged_list_1.append(str(local_apath/("1_labeling_"+it_step_zfill+".out"))+"\n")
                else:
                    failed_list_1.append(str(local_apath/("1_labeling_"+it_step_zfill+".out"))+"\n")
                del cp2k_output_1
            else:
                failed_list_1.append(str(local_apath/("1_labeling_"+it_step_zfill+".out"))+"\n")
            del cp2k_output_file_1

            cp2k_output_file_2 = local_apath/("2_labeling_"+it_step_zfill+".out")
            if cp2k_output_file_2.is_file():
                cp2k_output_2 = cf.read_file(cp2k_output_file_2)
                if any("SCF run converged in " in f for f in cp2k_output_2):
                    step_2 = step_2 + 1
                    timings_2 = [zzz for zzz in cp2k_output_2 if "CP2K                                 1  1.0" in zzz]
                    timings_sum_2 = timings_sum_2 + float(timings_2[0].split(" ")[-1])
                elif any("SCF run NOT converged in " in f for f in cp2k_output_2):
                    not_converged_list_2.append(str(local_apath/("2_labeling_"+it_step_zfill+".out"))+"\n")
                else:
                    failed_list_2.append(str(local_apath/("2_labeling_"+it_step_zfill+".out"))+"\n")
                del cp2k_output_2
            else:
                failed_list_2.append(str(local_apath/("2_labeling_"+it_step_zfill+".out"))+"\n")
            del cp2k_output_file_2
    del it_step, it_step_zfill

    if (step_1 == 0 or step_2 == 0) and skipped == 0:
        logging.critical("ALL jobs have failed/not converged/still running (second step).")
        logging.critical("Please check manually before relaunching this step")
        logging.critical("Or create files named \"skip\" to skip some configurations")
        logging.critical("Aborting...")
        sys.exit(1)

    timings_1 = timings_sum_1/step_1
    timings_2 = timings_sum_2/step_2
    labeling_json["subsys_nr"][it_subsys_nr]["timing_s"] = [timings_1, timings_2]
    labeling_json["subsys_nr"][it_subsys_nr]["candidates_skipped"] = skipped_subsys_candidates
    labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed_skipped"] = skipped_subsys_disturbed
    cf.remove_file(local_apath.parent/(str(it_subsys_nr)+"_1_not_converged.txt"))
    cf.remove_file(local_apath.parent/(str(it_subsys_nr)+"_2_not_converged.txt"))
    cf.remove_file(local_apath.parent/(str(it_subsys_nr)+"_1_failed.txt"))
    cf.remove_file(local_apath.parent/(str(it_subsys_nr)+"_2_failed.txt"))
    cf.remove_file(local_apath.parent/(str(it_subsys_nr)+"_skipped.txt"))
    cf.write_file(local_apath.parent/(str(it_subsys_nr)+"_1_not_converged.txt"),not_converged_list_1) if len(not_converged_list_1) != 0 else True
    cf.write_file(local_apath.parent/(str(it_subsys_nr)+"_2_not_converged.txt"),not_converged_list_2) if len(not_converged_list_2) != 0 else True
    logging.warning("Not converged calculations in sub-system "+str(it_subsys_nr)) if len(not_converged_list_1)+len(not_converged_list_2) != 0 else True
    cf.write_file(local_apath.parent/(str(it_subsys_nr)+"_1_failed.txt"),failed_list_1) if len(failed_list_1) != 0 else True
    cf.write_file(local_apath.parent/(str(it_subsys_nr)+"_2_failed.txt"),failed_list_2) if len(failed_list_2) != 0 else True
    logging.warning("Failed calculations in sub-system "+str(it_subsys_nr)) if len(failed_list_1)+len(failed_list_2) != 0 else True
    cf.write_file(local_apath.parent/(str(it_subsys_nr)+"_skipped.txt"),skipped_list) if len(skipped_list) != 0 else True
    logging.warning("Skipped calculations in sub-system "+str(it_subsys_nr)) if len(skipped_list) != 0 else True
    del timings_1, timings_sum_1, not_converged_list_1, failed_list_1
    del timings_2, timings_sum_2, not_converged_list_2, failed_list_2
    del average_per_step, local_apath

if total_steps != (step_1+skipped):
    logging.warning("Some jobs have failed/not converged/still running (first step). Check manually")
    logging.warning("See 1_not_converged.txt / 1_failed.txt")
if total_steps != (step_2+skipped):
    logging.critical("Some jobs have failed/not converged/still running (second step). Check manually")
    logging.critical("Or create files named \"skip\" to skip some configurations")
    logging.critical("See 2_not_converged.txt / 2_failed.txt")
    sys.exit(1)
else:
    labeling_json["is_checked"] = True
    cf.json_dump(labeling_json,(control_apath/("labeling_"+current_iteration_zfill+".json")),True)
    for it_subsys_nr in labeling_json["subsys_nr"]:
        local_apath = Path(".").resolve()/str(it_subsys_nr)
        logging.info("Deleting SLURM out/error files...")
        cf.remove_file_glob(local_apath,"**/CP2K.*")
        cf.remove_file_glob(local_apath,"CP2K.*")
        logging.info("Cleaning done!")
    del it_subsys_nr, local_apath
del total_steps, step_1, step_2

logging.info("Labeling: Check phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath
del current_iteration, current_iteration_zfill
del labeling_json
del deepmd_iterative_apath

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()
