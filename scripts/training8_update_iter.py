###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')

import subprocess

training_iterative_apath = str(Path('..').resolve())
### Check if the deepmd_iterative_apath is defined
if 'deepmd_iterative_path' in globals():
    True
elif Path(training_iterative_apath+'/control/path').is_file():
    with open(training_iterative_apath+'/control/path', 'r') as f:
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
config_json = cf.json_read(config_json_fpath, abort=True)

current_iteration = current_iteration if 'current_iteration' in globals() else config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

training_json_fpath = training_iterative_apath+'/control/training_'+current_iteration_zfill+'.json'
training_json = cf.json_read(training_json_fpath, abort=True)

### Checks
if training_json['is_frozen'] is False:
    logging.critical('Maybe freeze the NNPs before updating the iteration?')
    logging.critical('Aborting...')
    sys.exit(1)

### Prep the next iteration
for it_nnp in range(1, config_json['nb_nnp'] + 1):
    cf.change_dir('./'+str(it_nnp))
    cf.check_file('graph_'+str(it_nnp)+'_'+current_iteration_zfill+'.pb',0,True)
    if training_json['is_compressed'] is True:
        cf.check_file('graph_'+str(it_nnp)+'_'+current_iteration_zfill+'_compressed.pb',0,True)
    cf.remove_file_glob('.','DeepMD_*')
    cf.remove_file_glob('.','model.ckpt-*')
    cf.remove_file('checkpoint')
    cf.remove_file('input_v2_compat.json')
    if Path('model-compression').is_dir():
        cf.remove_tree(Path('model-compression'))
    cf.change_dir('../')

cf.create_dir('../'+current_iteration_zfill+'-test')
subprocess.call(['rsync','-a', training_iterative_apath+'/data', '../'+current_iteration_zfill+'-test/'])

cf.create_dir('../NNP')
for it_nnp in range(1, config_json['nb_nnp'] + 1):
    if training_json['is_compressed'] is True:
        subprocess.call(['rsync','-a', './'+str(it_nnp)+'/graph_'+str(it_nnp)+'_'+current_iteration_zfill+'_compressed.pb', '../NNP/'])
    else:
        subprocess.call(['rsync','-a', './'+str(it_nnp)+'/graph_'+str(it_nnp)+'_'+current_iteration_zfill+'.pb', '../NNP/'])
del it_nnp

current_iteration = current_iteration+1
config_json['current_iteration'] = current_iteration
current_iteration_zfill = str(current_iteration).zfill(3)

cf.create_dir('../'+current_iteration_zfill+'-exploration')
cf.create_dir('../'+current_iteration_zfill+'-reactive')
cf.create_dir('../'+current_iteration_zfill+'-labeling')
cf.create_dir('../'+current_iteration_zfill+'-training')

cf.json_dump(config_json,config_json_fpath, True, 'config file')

if Path('data').is_dir():
    cf.remove_tree(Path('data'))

logging.info('Updating the iteration is a success!')

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del training_json, training_json_fpath
del deepmd_iterative_apath

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()