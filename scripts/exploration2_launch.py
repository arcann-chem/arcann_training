###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')

import subprocess

training_iterative_apath = Path('..').resolve()
### Check if the deepmd_iterative_apath is defined
deepmd_iterative_apath_error = 1
if 'deepmd_iterative_apath' in globals():
    if (Path(deepmd_iterative_apath)/'scripts'/'common_functions.py').is_file():
        deepmd_iterative_apath_error = 0
elif (Path().home()/'deepmd_iterative_py'/'scripts'/'common_functions.py').is_file():
    deepmd_iterative_apath = Path().home()/'deepmd_iterative_py'
    deepmd_iterative_apath_error = 0
elif (training_iterative_apath/'control'/'path').is_file():
    deepmd_iterative_apath = Path((training_iterative_apath/'control'/'path').read_text())
    if (deepmd_iterative_apath/'scripts'/'common_functions.py').is_file():
        deepmd_iterative_apath_error = 0
if deepmd_iterative_apath_error == 1:
    logging.critical('Can\'t find common_functions.py in usual places:')
    logging.critical('deepmd_iterative_apath variable or ~/deepmd_iterative_py or in the path file in control')
    logging.critical('Aborting...')
    sys.exit(1)
sys.path.insert(0, str(Path(deepmd_iterative_apath)/'scripts'))
del deepmd_iterative_apath_error
import common_functions as cf

### Temp fix before Path/Str pass
training_iterative_apath = str(training_iterative_apath)
deepmd_iterative_apath = str(deepmd_iterative_apath)

### Read what is needed (json files)
config_json_fpath = training_iterative_apath+'/control/config.json'
config_json = cf.json_read(config_json_fpath,True,True)

current_iteration = current_iteration if 'current_iteration' in globals() else config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

exploration_json_fpath = training_iterative_apath+'/control/exploration_'+current_iteration_zfill+'.json'
exploration_json = cf.json_read(exploration_json_fpath,True,True)

### Checks
if exploration_json['is_locked'] is False:
    logging.critical('Lock found. Run/Check first: exploration1_prep.py')
    logging.critical('Aborting...')
    sys.exit(1)

cluster = cf.check_cluster()

if exploration_json['cluster'] != cluster:
    logging.critical('Different cluster ('+str(cluster)+') than the one for exploration1_prep..py ('+str(exploration_json['cluster'])+')')
    logging.critical('Aborting...')
    sys.exit(1)

### Set needed variables
if exploration_json['arch_name'] == 'v100' or exploration_json['arch_name'] == 'a100':
    arch_type='gpu'

### Launch the jobs
check = 0
for it_subsys_nr in config_json['subsys_nr']:
    cf.change_dir('./'+str(it_subsys_nr))
    for it_nnp in range(1, config_json['nb_nnp'] + 1):
        cf.change_dir('./'+str(it_nnp))
        for it_each in range(1, exploration_json['nb_traj'] + 1):
            cf.change_dir('./'+str(it_each).zfill(5))
            if Path('./job_deepmd_lammps_'+arch_type+'_'+cluster+'.sh').is_file():
                subprocess.call(['sbatch','./job_deepmd_lammps_'+arch_type+'_'+cluster+'.sh'])
                logging.info('Exploration - '+str(it_subsys_nr)+'/'+str(it_nnp)+'/'+str(it_each).zfill(5)+' launched')
                check = check + 1
            else:
                logging.critical('Exploration - '+str(it_subsys_nr)+'/'+str(it_nnp)+'/'+str(it_each).zfill(5)+' NOT launched')
            cf.change_dir('..')
        del it_each
        cf.change_dir('..')
    del it_nnp
    cf.change_dir('..')
del it_subsys_nr

if check == (len( exploration_json['subsys_nr']) * exploration_json['nb_nnp'] * exploration_json['nb_traj'] ):
    exploration_json['is_launched'] = True
    logging.info('Slurm launch of the exploration is a success!')
else:
    logging.critical('Some Exploration did not launched correctly')
    logging.critical('Please launch manually before continuing to the next step')
    logging.critical('And replace the key \'is_launched\' to True in the corresponding exploration.json.')
del check

cf.json_dump(exploration_json,exploration_json_fpath,True,'exploration.json')

del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del exploration_json, exploration_json_fpath
del cluster, arch_type
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()