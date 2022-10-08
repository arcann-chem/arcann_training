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

labeling_json_fpath = training_iterative_apath+'/control/labeling_'+current_iteration_zfill+'.json'
labeling_json = cf.json_read(labeling_json_fpath, abort=True)
del current_iteration_zfill, training_iterative_apath

cluster = cf.check_cluster()
if labeling_json['cluster'] != cluster:
    logging.critical('Different cluster ('+str(cluster)+') than the one for exploration1.py ('+str(labeling_json['cluster'])+')')
    logging.critical('Aborting...')
    sys.exit(1)

if labeling_json['arch_name'] == 'cpu':
    arch_type='cpu'

cluster = str(labeling_json['cluster'])

for it_subsys_nr in config_json['subsys_nr']:
    cf.change_dir('./'+str(it_subsys_nr))
    cf.check_file('job_labeling_array_'+arch_type+'_'+cluster+'.sh',0,1,'Labeling - '+str(it_subsys_nr)+' not lauched. Job file missing.')
    subprocess.call(['sbatch','./job_labeling_array_'+arch_type+'_'+cluster+'.sh'])
    cf.change_dir('..')
del it_subsys_nr, config_json, cluster

labeling_json['is_launched'] = True

cf.json_dump(labeling_json,labeling_json_fpath, True, 'exploration config file')

del labeling_json, labeling_json_fpath, arch_type
del deepmd_iterative_path

del sys, Path, subprocess, logging, cf
import gc; gc.collect(); del gc
exit()