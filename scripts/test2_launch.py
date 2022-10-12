## deepmd_iterative_apath
# deepmd_iterative_apath = ''
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name = 'nvs'
# allocation_name = 'v100'
# arch_name = 'v100'
# slurm_email = ''

###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')

import subprocess

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

config_json_fpath = training_iterative_apath+'/control/config.json'
config_json = cf.json_read(config_json_fpath,True,True)

current_iteration_zfill=Path().resolve().parts[-1].split('-')[0]
current_iteration=int(current_iteration_zfill)

test_json_fpath = training_iterative_apath+'/control/test_'+current_iteration_zfill+'.json'
test_json = cf.json_read(test_json_fpath,True,True)

### Checks
if test_json['is_launched'] is True:
    logging.critical('Already launched.')
    logging.critical('Aborting...')
    sys.exit(1)

if test_json['is_locked'] is False:
    logging.critical('Lock found. Run/Check first: test1_prep.py')
    logging.critical('Aborting...')
    sys.exit(1)

cluster = cf.check_cluster()

if test_json['cluster'] != cluster:
    logging.critical('Different cluster ('+str(cluster)+') than the one for test1_prep.py ('+str(test_json['cluster'])+')')
    logging.critical('Aborting...')
    sys.exit(1)

### Set needed variables
if test_json['arch_name'] == 'cpu':
    arch_type='cpu'

### Launch the jobs
check = 0
for it_nnp in range(1, test_json['nb_nnp'] + 1):
    if Path('job_deepmd_test_'+arch_type+'_'+cluster+'_NNP'+str(it_nnp)+'.sh').is_file():
        subprocess.call(['sbatch','./job_deepmd_test_'+arch_type+'_'+cluster+'_NNP'+str(it_nnp)+'.sh'])
        logging.info('DP Test - ./'+str(it_nnp)+' launched')
        check = check + 1
    else:
        logging.warning('DP Test - ./'+str(it_nnp)+' NOT launched')
    cf.change_dir('..')
del it_nnp

if check == config_json['nb_nnp']:
    test_json['is_launched'] = True
    logging.info('Slurm launch of the training is a success!')
else:
    logging.critical('Some DP Test did not launched correctly')
    logging.critical('Please launch manually before continuing to the next step')
    logging.critical('And replace the key \'is_launched\' to True in the corresponding training.json.')
del check

cf.json_dump(test_json,test_json_fpath,True,'training.json')

del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del test_json, test_json_fpath
del cluster, arch_type
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()