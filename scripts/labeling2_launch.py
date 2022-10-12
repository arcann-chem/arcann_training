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

### Read what is needed (json files)
config_json_fpath = training_iterative_apath+'/control/config.json'
config_json = cf.json_read(config_json_fpath,True,True)

current_iteration = current_iteration if 'current_iteration' in globals() else config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

labeling_json_fpath = training_iterative_apath+'/control/labeling_'+current_iteration_zfill+'.json'
labeling_json = cf.json_read(labeling_json_fpath,True,True)

### Checks
if labeling_json['is_launched'] is True:
    logging.critical('Already launched.')
    logging.critical('Aborting...')
    sys.exit(1)

if labeling_json['is_locked'] is False:
    logging.critical('Lock found. Run/Check first: labeling1_prep.py')
    logging.critical('Aborting...')
    sys.exit(1)

cluster = cf.check_cluster()

if labeling_json['cluster'] != cluster:
    logging.critical('Different cluster ('+str(cluster)+') than the one for labeling1_prep.py ('+str(labeling_json['cluster'])+')')
    logging.critical('Aborting...')
    sys.exit(1)

### Set needed variables
if labeling_json['arch_name'] == 'cpu':
    arch_type='cpu'

### Launch of the labeling
check = 0
for it0_subsys_nr,it_subsys_nr in enumerate(config_json['subsys_nr']):
    cf.change_dir('./'+str(it_subsys_nr))
    if cluster == 'jz':
        if Path('job_labeling_array_'+arch_type+'_'+cluster+'.sh').is_file():
            subprocess.call(['sbatch','./job_labeling_array_'+arch_type+'_'+cluster+'.sh'])
            logging.info('Labeling Array - '+str(it_subsys_nr)+' launched')
            check = check + 1
        else:
            logging.critical('Labeling Array - '+str(it_subsys_nr)+' NOT launched')
    elif cluster == 'ir':
        if it0_subsys_nr == 0:
            if Path('job_labeling_array_'+arch_type+'_'+cluster+'.sh').is_file():
                subprocess.call(['ccc_msub','./job_labeling_array_'+arch_type+'_'+cluster+'.sh'])
                logging.info('Labeling Array - '+str(it_subsys_nr)+' launched')
            elif Path('job_labeling_array_'+arch_type+'_'+cluster+'_0.sh').is_file():
                subprocess.call(['ccc_msub','./job_labeling_array_'+arch_type+'_'+cluster+'_0.sh'])
                logging.info('Labeling Array - '+str(it_subsys_nr)+' - 0 launched')
            else:
                logging.critical('Labeling Array - '+str(it_subsys_nr)+' NOT launched')

            logging.warning('Since Irene-Rome does not support more than 300 jobs at a time')
            logging.warning('and SLURM arrays not larger than 1000')
            logging.warning('the labeling array have been split into several jobs')
            logging.warning('and should launch itself automagically until the labeling is complete')
            logging.warning('The labeling job launch phase is a semi-success! (You are on Irene-Rome so who knows what can happen...)')
        else:
            True
    cf.change_dir('..')
del it_subsys_nr, it0_subsys_nr

if check == len(config_json['subsys_nr']):
    labeling_json['is_launched'] = True
    logging.info('The labeling job launch phase is a success!')
elif cluster == 'ir':
    True
else:
    logging.critical('Some labeling arrays did not launched correctly')
    logging.critical('Please launch manually before continuing to the next step')
    logging.critical('And replace the key \'is_launched\' to True in the corresponding labeling.json.')
del check

cf.json_dump(labeling_json,labeling_json_fpath,True,'labeling.json')

### Clean
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del labeling_json, labeling_json_fpath
del cluster, arch_type
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()