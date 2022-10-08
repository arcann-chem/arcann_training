###################################### No change past here
import sys
from pathlib import Path
import subprocess
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')


training_iterative_apath = str(Path('..').resolve())
### Check if the DeePMD Iterative PY path is defined
if Path(training_iterative_apath+'/control/path').is_file():
    with open(training_iterative_apath+'/control/path', "r") as f:
        deepmd_iterative_path = f.read()
    f.close()
    del f
else:
    if 'deepmd_iterative_path' not in globals() :
        logging.critical(training_iterative_apath+'/control/path not found and deepmd_iterative_path not defined.')
        logging.critical('Aborting...')
        sys.exit(1)
sys.path.insert(0, deepmd_iterative_path+'/scripts/')
import common_functions as cf

### Read what is needed
config_json_fpath = training_iterative_apath+'/control/config.json'
config_json = cf.json_read(config_json_fpath, abort=True)

config_json['current_iteration'] = current_iteration if 'current_iteration' in globals() else cf.check_if_in_dict(config_json,'current_iteration',False,0)
current_iteration_zfill = str(config_json['current_iteration']).zfill(3)
del config_json_fpath

exploration_json_fpath = training_iterative_apath+'/control/exploration_'+current_iteration_zfill+'.json'
exploration_json = cf.json_read(exploration_json_fpath, abort=True)
del current_iteration_zfill, training_iterative_apath

if exploration_json['arch_name'] == 'v100' or exploration_json['arch_name'] == 'a100':
    arch_type='gpu'

cluster = cf.check_cluster()
if exploration_json['cluster'] != cluster:
    logging.critical('Different cluster ('+str(cluster)+') than the one for exploration1.py ('+str(exploration_json['cluster'])+')')
    logging.critical('Aborting...')
    sys.exit(1)

cluster = str(exploration_json['cluster'])

for it_subsys_nr in config_json['subsys_nr']:
    cf.change_dir('./'+str(it_subsys_nr))
    for it_nnp in range(1, config_json['nb_nnp'] + 1):
        cf.change_dir('./'+str(it_nnp))
        for it_each in range(1, exploration_json['nb_traj'] + 1):
            cf.change_dir('./'+str(it_each).zfill(5))
            cf.check_file('job_deepmd_lammps_'+arch_type+'_'+cluster+'.sh',0,1,'Exploration - '+str(it_subsys_nr)+'/'+str(it_nnp)+'/'+str(it_each)+' not lauched. Job file missing.')
            subprocess.call(['sbatch','./job_deepmd_lammps_'+arch_type+'_'+cluster+'.sh'])
            cf.change_dir('..')
        del it_each
        cf.change_dir('..')
    del it_nnp
    cf.change_dir('..')
del it_subsys_nr, config_json, cluster

exploration_json['is_launched'] = True

cf.json_dump(exploration_json,exploration_json_fpath, True, 'exploration config file')

del exploration_json, exploration_json_fpath, arch_type
del deepmd_iterative_path

del sys, Path, subprocess, logging, cf
import gc; gc.collect(); del gc
exit()