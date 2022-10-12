## deepmd_iterative_apath
# deepmd_iterative_apath = ''
## These are the default
# nb_candidates_max = [500, 500]
# s_low = [0.1, 0.1]
# s_high = [0.8, 0.8]
# s_high_max = [1.0, 1.0]
# ignore_first_n_frames = [10, 10]

###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level = logging.INFO,format='%(levelname)s: %(message)s')

import numpy as np

training_iterative_apath = str(Path('..').resolve())
### Check if the deepmd_iterative_apath is defined
if 'deepmd_iterative_apath' in globals():
    True
elif Path(training_iterative_apath+'/control/path').is_file():
    with open(training_iterative_apath+'/control/path', "r") as f:
        deepmd_iterative_apath = f.read()
    f.close()
    del f
else:
    if 'deepmd_iterative_apath' not in globals() :
        logging.critical(training_iterative_apath+'/control/path not found and deepmd_iterative_apath not defined.')
        logging.critical('Aborting...')
        sys.exit(1)
sys.path.insert(0, deepmd_iterative_apath+'/scripts/')
import common_functions as cf

### Read what is needed (json files)
config_json_fpath = training_iterative_apath+'/control/config.json'
config_json = cf.json_read(config_json_fpath,True,True)

current_iteration = current_iteration if 'current_iteration' in globals() else config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

exploration_json_fpath = training_iterative_apath+'/control/exploration_'+current_iteration_zfill+'.json'
exploration_json = cf.json_read(exploration_json_fpath,True,True)

### Checks
if exploration_json['is_checked'] is False:
    logging.critical('Lock found. Run/Check first: exploration3_check.py')
    logging.critical('Aborting...')
    sys.exit(1)

### Running the Query-by-commitee
for it0_subsys_nr,it_subsys_nr in enumerate(config_json['subsys_nr']):
    cf.change_dir('./'+str(it_subsys_nr))

    exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates_max'] = nb_candidates_max[it0_subsys_nr] if 'nb_candidates_max' in globals() else config_json['subsys_nr'][it_subsys_nr]['nb_candidates_max']
    exploration_json['subsys_nr'][it_subsys_nr]['s_low'] = s_low[it0_subsys_nr] if 's_low' in globals() else config_json['subsys_nr'][it_subsys_nr]['s_low']
    exploration_json['subsys_nr'][it_subsys_nr]['s_high'] = s_high[it0_subsys_nr] if 's_high' in globals() else config_json['subsys_nr'][it_subsys_nr]['s_high']
    exploration_json['subsys_nr'][it_subsys_nr]['s_high_max'] = s_high_max[it0_subsys_nr] if 's_high_max' in globals() else config_json['subsys_nr'][it_subsys_nr]['s_high_max']
    exploration_json['subsys_nr'][it_subsys_nr]['ignore_first_n_frames'] = ignore_first_n_frames[it0_subsys_nr] if 'ignore_first_n_frames' in globals() else config_json['subsys_nr'][it_subsys_nr]['ignore_first_n_frames']
    exploration_json['subsys_nr'][it_subsys_nr]['avg_max_devi_f'] = 0
    exploration_json['subsys_nr'][it_subsys_nr]['std_max_devi_f'] = 0
    exploration_json['subsys_nr'][it_subsys_nr]['nb_total'] = 0
    exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates'] = 0
    exploration_json['subsys_nr'][it_subsys_nr]['nb_rejected'] = 0
    full_skip = 0

    s_low_local = exploration_json['subsys_nr'][it_subsys_nr]['s_low']
    s_high_local = exploration_json['subsys_nr'][it_subsys_nr]['s_high']
    s_high_max_local =  exploration_json['subsys_nr'][it_subsys_nr]['s_high_max']
    ignore_first_n_frames_local = exploration_json['subsys_nr'][it_subsys_nr]['ignore_first_n_frames'] 

    for it_nnp in range(1, exploration_json['nb_nnp'] + 1):
        cf.change_dir('./'+str(it_nnp))

        for it_each in range(1, exploration_json['nb_traj']+1):
            cf.change_dir('./'+str(it_each).zfill(5))

            filename_str = it_subsys_nr+'_'+str(it_nnp)+'_'+current_iteration_zfill
            devi_json = cf.json_read('./selection_candidates.json',False,False)
            devi_json_index = cf.json_read('./selection_candidates_index.json',False,False)

            devi_json['s_low'] = s_low_local
            devi_json['s_high'] = s_high_local
            devi_json['s_high_max'] = s_high_max_local

            devi = np.genfromtxt('model_devi_'+filename_str+'.out')
            devi_json['nb_total'] = devi.shape[0]-ignore_first_n_frames_local

            # Skip the first frame if from disturbed
            if current_iteration == 1 :
                start = 0
            elif exploration_json['subsys_nr'][it_subsys_nr]['disturbed_start']:
                start = 1
            else:
                start = 0

            if np.any(devi[start:,4] >= s_high_max_local):
                end = np.argmax(devi[start:,4] >= s_high_max_local)
            else:
                end = -1

            if end == -1:
                devi_json['avg_max_devi_f'] = np.average(devi[ignore_first_n_frames_local:,4])
                devi_json['std_max_devi_f'] = np.std(devi[ignore_first_n_frames_local:,4])
                filter_candidates = np.logical_and(devi[ignore_first_n_frames_local:,4]>s_low_local,devi[ignore_first_n_frames_local:,4]<s_high_local)
                candidates_ind = devi[ignore_first_n_frames_local:,0][filter_candidates]
                filter_good = devi[ignore_first_n_frames_local:,4] <= s_low_local
                good_ind = devi[ignore_first_n_frames_local:,0][filter_good]
                filter_rejected = devi[ignore_first_n_frames_local:,4] >= s_high_local
                rejected_ind = devi[ignore_first_n_frames_local:,0][filter_rejected]

            elif end <= ignore_first_n_frames_local:
                candidates_ind=np.array([])
                good_ind=np.array([])
                rejected_ind = devi[ignore_first_n_frames_local:,0]
                devi_json['avg_max_devi_f']=np.average(devi[ignore_first_n_frames_local:,4])
                devi_json['std_max_devi_f']=np.std(devi[ignore_first_n_frames_local:,4])
                full_skip = full_skip + 1

            else:
                devi_json['avg_max_devi_f'] = np.average(devi[ignore_first_n_frames_local:end,4])
                devi_json['std_max_devi_f'] = np.std(devi[ignore_first_n_frames_local:end,4])

                filter_candidates = np.logical_and(devi[ignore_first_n_frames_local:end,4]>s_low_local,devi[ignore_first_n_frames_local:end,4]<s_high_local)
                candidates_ind = devi[ignore_first_n_frames_local:end,0][filter_candidates]
                filter_good = devi[ignore_first_n_frames_local:end,4] <= s_low_local
                good_ind = devi[ignore_first_n_frames_local:end,0][filter_good]
                filter_rejected = devi[ignore_first_n_frames_local:end,4] >= s_high_local
                rejected_ind = devi[ignore_first_n_frames_local:end,0][filter_rejected]
                rejected_ind=np.hstack((rejected_ind,devi[end:,4]))

            devi_json['nb_good'] = good_ind.shape[0]
            devi_json_index['good_ind'] = good_ind.tolist()
            devi_json['nb_rejected'] = rejected_ind.shape[0]
            devi_json_index['rejected_ind'] = rejected_ind.tolist()
            devi_json['nb_candidates'] = candidates_ind.shape[0]
            devi_json_index['candidates_ind'] = candidates_ind.tolist()

            if  (end > ignore_first_n_frames_local) or (end == -1) :
                exploration_json['subsys_nr'][it_subsys_nr]['avg_max_devi_f'] = exploration_json['subsys_nr'][it_subsys_nr]['avg_max_devi_f'] + devi_json['avg_max_devi_f']
                exploration_json['subsys_nr'][it_subsys_nr]['std_max_devi_f'] = exploration_json['subsys_nr'][it_subsys_nr]['std_max_devi_f'] + devi_json['std_max_devi_f']

            exploration_json['subsys_nr'][it_subsys_nr]['nb_total'] = exploration_json['subsys_nr'][it_subsys_nr]['nb_total'] + devi_json['nb_total']
            exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates'] = exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates'] + devi_json['nb_candidates']
            exploration_json['subsys_nr'][it_subsys_nr]['nb_rejected'] = exploration_json['subsys_nr'][it_subsys_nr]['nb_rejected'] + devi_json['nb_rejected']

            cf.json_dump(devi_json,'./selection_candidates.json',False,'selection_candidates.json')
            cf.json_dump(devi_json_index,'./selection_candidates_index.json',False,'selection_candidates_index.json')

            if  (end > ignore_first_n_frames_local) or (end == -1) :
                del filter_candidates, filter_good, filter_rejected

            del filename_str, devi, end, devi_json, devi_json_index, candidates_ind, good_ind, rejected_ind
            cf.change_dir('..')

        del it_each
        cf.change_dir('..')

    del it_nnp, ignore_first_n_frames_local, s_low_local, s_high_local, s_high_max_local
    cf.change_dir('..')

    exploration_json['subsys_nr'][it_subsys_nr]['avg_max_devi_f'] = exploration_json['subsys_nr'][it_subsys_nr]['avg_max_devi_f'] / ( exploration_json['nb_nnp'] +  len(range(1, exploration_json['nb_traj'] + 1)) - full_skip )
    exploration_json['subsys_nr'][it_subsys_nr]['std_max_devi_f'] = exploration_json['subsys_nr'][it_subsys_nr]['std_max_devi_f'] / ( exploration_json['nb_nnp'] +  len(range(1, exploration_json['nb_traj'] + 1)) - full_skip )

del it_subsys_nr

for it0_subsys_nr,it_subsys_nr in enumerate(exploration_json['subsys_nr']):
    cf.change_dir('./'+str(it_subsys_nr))

    exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates_kept'] = 0
    exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates_discarded'] = 0

    nb_candidates_max_local = nb_candidates_max[it0_subsys_nr] if 'nb_candidates_max' in globals() else exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates_max']

    for it_nnp in range(1, exploration_json['nb_nnp'] + 1):
        cf.change_dir('./'+str(it_nnp))

        for it_each in range(1,exploration_json['nb_traj']+1):
            cf.change_dir('./'+str(it_each).zfill(5))

            devi_json = cf.json_read('./selection_candidates.json',True,False)
            devi_json_index = cf.json_read('./selection_candidates_index.json',False,False)

            if exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates'] <= nb_candidates_max_local:
                nb_selection_factor = 1
            else:
                nb_selection_factor =  devi_json['nb_candidates']  / exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates']

            devi_json['nb_selection_factor'] = nb_selection_factor
            nb_candidates_max_weighted = int(np.floor(nb_candidates_max_local*nb_selection_factor))
            devi_json['nb_candidates_max_weighted'] = nb_candidates_max_weighted

            candidates_ind = np.array(devi_json_index['candidates_ind'])

            if devi_json['nb_candidates'] > nb_candidates_max_weighted:
                candidates_ind_kept = candidates_ind[np.round(np.linspace(0, len(candidates_ind)-1, nb_candidates_max_weighted)).astype(int)]
            else:
                candidates_ind_kept = candidates_ind

            candidates_ind_disc = np.array([zzz for zzz in candidates_ind if zzz not in candidates_ind_kept])

            devi_json['nb_candidates_kept'] = candidates_ind_kept.shape[0]
            devi_json_index['candidates_kept_ind'] = candidates_ind_kept.tolist()
            devi_json['nb_candidates_discarded'] = candidates_ind_disc.shape[0]
            devi_json_index['candidates_discarded_ind'] = candidates_ind_disc.tolist()


            if candidates_ind_kept.shape[0] > 0:
                filename_str = it_subsys_nr+'_'+str(it_nnp)+'_'+current_iteration_zfill
                devi = np.genfromtxt('model_devi_'+filename_str+'.out')
                min_val = 1e30
                for it_ind_kept in devi_json_index['candidates_kept_ind']:
                    temp_min = devi[:,4][np.where(devi[:,0] == it_ind_kept)]
                    if temp_min < min_val:
                        min_val = temp_min
                        min_index = it_ind_kept
                devi_json['min_index'] = it_ind_kept
                del filename_str, devi, min_val, min_index, it_ind_kept, temp_min
            else:
                devi_json['min_index'] = devi_json_index['good_ind'][-1]


            cf.json_dump(devi_json,'./selection_candidates.json',False,'selection_candidates.json')
            cf.json_dump(devi_json_index,'./selection_candidates_index.json',False,'selection_candidates_index.json')

            exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates_kept'] = exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates_kept'] + candidates_ind_kept.shape[0]
            exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates_discarded'] = exploration_json['subsys_nr'][it_subsys_nr]['nb_candidates_discarded'] + candidates_ind_disc.shape[0]

            del devi_json, devi_json_index, nb_selection_factor, candidates_ind_kept, nb_candidates_max_weighted, candidates_ind, candidates_ind_disc
            cf.change_dir('..')
        del it_each
        cf.change_dir('..')
    del it_nnp, nb_candidates_max_local
    cf.change_dir('..')
del it0_subsys_nr,it_subsys_nr

exploration_json['is_deviated'] = True
cf.json_dump(exploration_json,exploration_json_fpath,True,'exploration.json')

del full_skip

logging.info('The exploration deviation phase is a success!')

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del exploration_json, exploration_json_fpath
del deepmd_iterative_apath

del sys, Path, logging, cf
del np
import gc; gc.collect(); del gc
exit()