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

del deepmd_iterative_path

### Read what is needed
config_json_fpath = training_iterative_apath+'/control/config.json'
config_json = cf.json_read(config_json_fpath, abort=True)

config_json['current_iteration'] = current_iteration if 'current_iteration' in globals() else cf.check_if_in_dict(config_json,'current_iteration',False,0)
current_iteration = config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

training_json_fpath = str(Path(training_iterative_apath+'/control/training_'+current_iteration_zfill+'.json').resolve())
training_json = cf.json_read(training_json_fpath, abort=True)
deepmd_model_version = str(training_json['deepmd_model_version'])

del training_iterative_apath, current_iteration, current_iteration_zfill

if training_json['arch_name'] == 'v100' or training_json['arch_name'] == 'a100':
    arch_type='gpu'

cluster = cf.check_cluster()
training_json['cluster'] = cluster

del config_json_fpath
check = 0
for it_nnp in range(1, config_json['nb_nnp'] + 1):
    cf.change_dir('./'+str(it_nnp))
#-----# The check serves no purpose now, the job file in training1 always exists or abort !
    cf.check_file('job_deepmd_train_'+arch_type+'_'+cluster+'.sh',0,1,'Training of NNP '+str(it_nnp)+' was not lauched. No job file found.')
    if Path('job_deepmd_train_'+arch_type+'_'+cluster+'.sh').is_file():
        subprocess.call(['sbatch','./job_deepmd_train_'+arch_type+'_'+cluster+'.sh'])
        check = check + 1
    cf.change_dir('..')

if check == config_json['nb_nnp']:
    training_json['is_launched'] = True
del check

cf.json_dump(training_json, training_json_fpath, print_log=True, name='training config file')

del training_json, training_json_fpath
del deepmd_model_version, cluster, it_nnp, config_json, arch_type

del sys, Path, subprocess, logging, cf
import gc; gc.collect(); del gc
exit()