## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## These are the default
# nb_candidates_max = [500, 500]
# s_low: list = [0.1, 0.1]
# s_high: list = [0.8, 0.8]
# s_high_max: list = [1.0, 1.0]
# ignore_first_x_ps: list = [0.5, 0.5]

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO,format="%(levelname)s: %(message)s")

import numpy as np

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

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
exploration_json = cf.json_read((control_apath/("exploration_"+current_iteration_zfill+".json")),True,True)

### Checks
if not exploration_json["is_checked"]:
    logging.critical("Lock found. Run/Check first: exploration3_check.py")
    logging.critical("Aborting...")
    sys.exit(1)
if exploration_json["exploration_type"] == "i-PI" and not exploration_json["is_rechecked"]:
    ### #12
    logging.critical("Lock found. Run/Check first: XXX")
    logging.critical("Aborting...")
    sys.exit(1)

### Running the Query-by-commitee
for it0_subsys_nr,it_subsys_nr in enumerate(config_json["subsys_nr"]):

    exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_max"] = config_json["subsys_nr"][it_subsys_nr]["nb_candidates_max"] if "nb_candidates_max" not in globals() else nb_candidates_max[it0_subsys_nr]
    exploration_json["subsys_nr"][it_subsys_nr]["s_low"] = config_json["subsys_nr"][it_subsys_nr]["s_low"] if "s_low" not in globals() else s_low[it0_subsys_nr]
    exploration_json["subsys_nr"][it_subsys_nr]["s_high"] = config_json["subsys_nr"][it_subsys_nr]["s_high"] if "s_high" not in globals() else s_high[it0_subsys_nr]
    exploration_json["subsys_nr"][it_subsys_nr]["s_high_max"] = config_json["subsys_nr"][it_subsys_nr]["s_high_max"] if "s_high_max" not in globals() else s_high_max[it0_subsys_nr]
    exploration_json["subsys_nr"][it_subsys_nr]["ignore_first_x_ps"] = config_json["subsys_nr"][it_subsys_nr]["ignore_first_x_ps"] if "ignore_first_x_ps" not in globals() else ignore_first_x_ps[it0_subsys_nr]
    exploration_json["subsys_nr"][it_subsys_nr]["avg_max_devi_f"] = 0
    exploration_json["subsys_nr"][it_subsys_nr]["std_max_devi_f"] = 0
    exploration_json["subsys_nr"][it_subsys_nr]["nb_total"] = 0
    exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"] = 0
    exploration_json["subsys_nr"][it_subsys_nr]["nb_rejected"] = 0
    skipped_traj = 0

    it_s_low = exploration_json["subsys_nr"][it_subsys_nr]["s_low"]
    it_s_high = exploration_json["subsys_nr"][it_subsys_nr]["s_high"]
    it_s_high_max = exploration_json["subsys_nr"][it_subsys_nr]["s_high_max"]

    it_first_frame = 0
    while it_first_frame * exploration_json["subsys_nr"][it_subsys_nr]["print_freq"] * exploration_json["subsys_nr"][it_subsys_nr]["timestep_ps"] < exploration_json["subsys_nr"][it_subsys_nr]["ignore_first_x_ps"]:
        it_first_frame = it_first_frame + 1

    for it_nnp in range(1, exploration_json["nb_nnp"] + 1):
        for it_each in range(1, exploration_json["nb_traj"]+1):

            local_apath = Path(".").resolve()/str(it_subsys_nr)/str(it_nnp)/(str(it_each).zfill(5))
            devi_filename = "model_devi_"+it_subsys_nr+"_"+str(it_nnp)+"_"+current_iteration_zfill+".out"
            devi_info_json = cf.json_read(local_apath/"devi_info.json",False,False)
            devi_index_json = cf.json_read(local_apath/"devi_index.json",False,False)

            devi_info_json["s_low"] = it_s_low
            devi_info_json["s_high"] = it_s_high
            devi_info_json["s_high_max"] = it_s_high_max

            expected =  int( exploration_json["subsys_nr"][it_subsys_nr]["nb_steps"] / exploration_json["subsys_nr"][it_subsys_nr]["print_freq"] + 1 ) - it_first_frame

            if not (local_apath/"skip").is_file():

                devi = np.genfromtxt(str(local_apath/devi_filename))

                if expected > ( devi.shape[0] - it_first_frame ):
                    devi_info_json["nb_total"] = expected
                    logging.warning("Exploration "+ str(it_subsys_nr)+" / "+str(it_nnp)+" / "+str(it_each))
                    logging.warning("mismatch between expected and actual number in the deviation file")
                    if (local_apath/"forced").is_file():
                        logging.warning("but it has been forced, so it should be ok")
                elif expected == ( devi.shape[0] - it_first_frame ):
                    devi_info_json["nb_total"] = devi.shape[0] - it_first_frame
                else:
                    logging.critical("Unknown error. Please BUG REPORT")
                    logging.critical("Aborting...")
                    sys.exit(1)

                ### Skip the first frame if from disturbed
                if int(current_iteration_zfill) == 1 :
                    start = 0
                elif exploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"]:
                    start = 1
                else:
                    start = 0

                ### Check if is blows up
                if np.any(devi[start:,4] >= it_s_high_max):
                    end = np.argmax(devi[start:,4] >= it_s_high_max)
                else:
                    end = -1

                ### This is when the system is OK for the full traj
                if end == -1:
                    devi_info_json["avg_max_devi_f"] = np.average(devi[it_first_frame:,4])
                    devi_info_json["std_max_devi_f"] = np.std(devi[it_first_frame:,4])
                    filter_candidates = np.logical_and(devi[it_first_frame:,4]>it_s_low,devi[it_first_frame:,4]<it_s_high)
                    candidates_ind = devi[it_first_frame:,0][filter_candidates]
                    filter_good = devi[it_first_frame:,4] <= it_s_low
                    good_ind = devi[it_first_frame:,0][filter_good]
                    filter_rejected = devi[it_first_frame:,4] >= it_s_high
                    rejected_ind = devi[it_first_frame:,0][filter_rejected]

                ### This is when the system blows up right away (SKIP everything for stats)
                elif end <= it_first_frame:
                    ####TOTHINK Maybe replace by 999 like skiped traj?
                    devi_info_json["avg_max_devi_f"] = np.average(devi[it_first_frame:,4])
                    devi_info_json["std_max_devi_f"] = np.std(devi[it_first_frame:,4])
                    ### And there is no filter
                    candidates_ind = np.array([])
                    good_ind = np.array([])
                    rejected_ind = devi[it_first_frame:,0]
                    ### It is a skip
                    skipped_traj = skipped_traj + 1

                ### This is when the system blows up somewhere (even if get back good after) (Get relevents stats only before BOOM)
                else:
                    devi_info_json["avg_max_devi_f"] = np.average(devi[it_first_frame:end,4])
                    devi_info_json["std_max_devi_f"] = np.std(devi[it_first_frame:end,4])
                    filter_candidates = np.logical_and(devi[it_first_frame:end,4]>it_s_low,devi[it_first_frame:end,4]<it_s_high)
                    candidates_ind = devi[it_first_frame:end,0][filter_candidates]
                    filter_good = devi[it_first_frame:end,4] <= it_s_low
                    good_ind = devi[it_first_frame:end,0][filter_good]
                    filter_rejected = devi[it_first_frame:end,4] >= it_s_high
                    rejected_ind = devi[it_first_frame:end,0][filter_rejected]
                    ### Sum with the rejected-discarded after BOOM
                    rejected_ind=np.hstack((rejected_ind,devi[end:,4]))

                ###
                devi_info_json["nb_good"] = good_ind.shape[0]
                devi_index_json["good_ind"] = good_ind.tolist()
                devi_info_json["nb_rejected"] = rejected_ind.shape[0]
                devi_index_json["rejected_ind"] = rejected_ind.tolist()
                devi_info_json["nb_candidates"] = candidates_ind.shape[0]
                devi_index_json["candidates_ind"] = candidates_ind.tolist()

                ### If the traj is smaller than expected (forced case) add the missing as rejected
                if ( devi_info_json["nb_good"] + devi_info_json["nb_rejected"] + devi_info_json["nb_candidates"] ) < expected:
                    devi_info_json["nb_rejected"] = devi_info_json["nb_rejected"] + expected - ( devi_info_json["nb_good"] + devi_info_json["nb_rejected"] + devi_info_json["nb_candidates"] )

                ### Only if we have corect stats, add it
                if  (end > it_first_frame) or (end == -1) :
                    exploration_json["subsys_nr"][it_subsys_nr]["avg_max_devi_f"] = exploration_json["subsys_nr"][it_subsys_nr]["avg_max_devi_f"] + devi_info_json["avg_max_devi_f"]
                    exploration_json["subsys_nr"][it_subsys_nr]["std_max_devi_f"] = exploration_json["subsys_nr"][it_subsys_nr]["std_max_devi_f"] + devi_info_json["std_max_devi_f"]

                exploration_json["subsys_nr"][it_subsys_nr]["nb_total"] = exploration_json["subsys_nr"][it_subsys_nr]["nb_total"] + devi_info_json["nb_total"]
                exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"] = exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"] + devi_info_json["nb_candidates"]
                exploration_json["subsys_nr"][it_subsys_nr]["nb_rejected"] = exploration_json["subsys_nr"][it_subsys_nr]["nb_rejected"] + devi_info_json["nb_rejected"]

                # cf.json_dump(devi_info_json,"./selection_candidates.json",False,"selection_candidates.json")
                # cf.json_dump(devi_index_json,"./selection_candidates_index.json",False,"selection_candidates_index.json")

                if  (end > it_first_frame) or (end == -1) :
                    del filter_candidates, filter_good, filter_rejected
                del devi, end, candidates_ind, good_ind, rejected_ind, start

            else:
                ### If the trajectory was used skiped, count everything as a failure
                skipped_traj = skipped_traj + 1
                devi_info_json["avg_max_devi_f"] = 999
                devi_info_json["std_max_devi_f"] = 999
                devi_info_json["nb_total"] = expected
                devi_info_json["nb_good"] = 0
                devi_index_json["good_ind"] = []
                devi_info_json["nb_rejected"] = expected
                devi_index_json["rejected_ind"] = []
                devi_info_json["nb_candidates"] = 0
                devi_index_json["candidates_ind"] = []

                exploration_json["subsys_nr"][it_subsys_nr]["nb_total"] = exploration_json["subsys_nr"][it_subsys_nr]["nb_total"] + devi_info_json["nb_total"]
                exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"] = exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"] + devi_info_json["nb_candidates"]
                exploration_json["subsys_nr"][it_subsys_nr]["nb_rejected"] = exploration_json["subsys_nr"][it_subsys_nr]["nb_rejected"] + devi_info_json["nb_rejected"]

            cf.json_dump(devi_info_json,local_apath/"devi_info.json",False)
            cf.json_dump(devi_index_json,local_apath/"devi_index.json",False)

            del devi_filename, devi_info_json, devi_index_json, expected, local_apath
        del it_each
    del it_nnp, it_first_frame, it_s_low, it_s_high, it_s_high_max

    ### Average for the subsys (with adjustment, remove the skipped ones)
    exploration_json["subsys_nr"][it_subsys_nr]["avg_max_devi_f"] = exploration_json["subsys_nr"][it_subsys_nr]["avg_max_devi_f"] / ( exploration_json["nb_nnp"] +  len(range(1, exploration_json["nb_traj"] + 1)) - skipped_traj )
    exploration_json["subsys_nr"][it_subsys_nr]["std_max_devi_f"] = exploration_json["subsys_nr"][it_subsys_nr]["std_max_devi_f"] / ( exploration_json["nb_nnp"] +  len(range(1, exploration_json["nb_traj"] + 1)) - skipped_traj )

del it_subsys_nr

for it0_subsys_nr,it_subsys_nr in enumerate(exploration_json["subsys_nr"]):

    exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_kept"] = 0
    exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_discarded"] = 0

    it_nb_candidates_max = exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_max"] if "nb_candidates_max" not in globals() else nb_candidates_max[it0_subsys_nr]

    for it_nnp in range(1, exploration_json["nb_nnp"] + 1):
        for it_each in range(1,exploration_json["nb_traj"]+1):

            local_apath = Path(".").resolve()/str(it_subsys_nr)/str(it_nnp)/(str(it_each).zfill(5))
            devi_info_json = cf.json_read(local_apath/"devi_info.json",True,False)
            devi_index_json = cf.json_read(local_apath/"devi_index.json",True,False)

            if not (local_apath/"skip").is_file():

                ### This is the "weight" in case of to much candidates per the limits
                ### Enough for all
                if exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"] <= it_nb_candidates_max:
                    selection_factor = 1
                ### Cut is needed
                else:
                    selection_factor = devi_info_json["nb_candidates"] / exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"]

                devi_info_json["selection_factor"] = selection_factor
                it_nb_candidates_max_local = int(np.floor(it_nb_candidates_max*selection_factor))
                if selection_factor == 1:
                    devi_info_json["nb_candidates_max_local"] = -1
                else:
                    devi_info_json["nb_candidates_max_local"] = it_nb_candidates_max_local

                candidates_ind = np.array(devi_index_json["candidates_ind"])
                ### Selection of candidates (as linearly as possible, keeping the first and the last ones)
                if devi_info_json["nb_candidates"] > it_nb_candidates_max_local:
                    candidates_ind_kept = candidates_ind[np.round(np.linspace(0, len(candidates_ind)-1, it_nb_candidates_max_local)).astype(int)]
                else:
                    candidates_ind_kept = candidates_ind

                candidates_ind_disc = np.array([zzz for zzz in candidates_ind if zzz not in candidates_ind_kept])

                devi_info_json["nb_candidates_kept"] = candidates_ind_kept.shape[0]
                devi_index_json["candidates_kept_ind"] = candidates_ind_kept.tolist()
                devi_info_json["nb_candidates_discarded"] = candidates_ind_disc.shape[0]
                devi_index_json["candidates_discarded_ind"] = candidates_ind_disc.tolist()

                ### Selection of the next starting point
                if candidates_ind_kept.shape[0] > 0:
                    ### Min of candidates
                    devi_filename = "model_devi_"+it_subsys_nr+"_"+str(it_nnp)+"_"+current_iteration_zfill+".out"
                    devi = np.genfromtxt(str(local_apath/devi_filename))
                    min_val = 1e30
                    for it_ind_kept in devi_index_json["candidates_kept_ind"]:
                        temp_min = devi[:,4][np.where(devi[:,0] == it_ind_kept)]
                        if temp_min < min_val:
                            min_val = temp_min
                            min_index = it_ind_kept
                    devi_info_json["min_index"] = it_ind_kept
                    del devi_filename, devi, min_val, min_index, it_ind_kept, temp_min
                elif len(devi_index_json["good_ind"]) > 0:
                    ### Last of good
                    devi_info_json["min_index"] = devi_index_json["good_ind"][-1]
                else:
                    ### Nothing
                    devi_info_json["min_index"] = -1

                exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_kept"] = exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_kept"] + candidates_ind_kept.shape[0]
                exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_discarded"] = exploration_json["subsys_nr"][it_subsys_nr]["nb_candidates_discarded"] + candidates_ind_disc.shape[0]
                del it_nb_candidates_max_local, selection_factor, candidates_ind_kept, candidates_ind, candidates_ind_disc

            else:
                ### In case of skipped trajectories
                devi_info_json["nb_selection_factor"] = 0
                devi_info_json["nb_candidates_max_weighted"] = 0
                devi_info_json["nb_candidates_kept"] = 0
                devi_index_json["candidates_kept_ind"] = []
                devi_info_json["nb_candidates_discarded"] = 0
                devi_index_json["candidates_discarded_ind"] = []
                devi_info_json["min_index"] = -1

            cf.json_dump(devi_info_json,local_apath/"devi_info.json",False)
            cf.json_dump(devi_index_json,local_apath/"devi_index.json",False)

            del devi_info_json, devi_index_json, local_apath

        del it_each
    del it_nnp, it_nb_candidates_max
del it0_subsys_nr,it_subsys_nr

exploration_json["is_deviated"] = True

cf.json_dump(exploration_json,(control_apath/("exploration_"+current_iteration_zfill+".json")),True)

del skipped_traj

logging.info("Exploration-Deviation phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath
del current_iteration_zfill
del exploration_json
del deepmd_iterative_apath

del sys, Path, logging, cf
del np
import gc; gc.collect(); del gc
exit()