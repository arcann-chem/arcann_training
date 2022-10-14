###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')

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

training_json_fpath = training_iterative_apath+'/control/training_'+current_iteration_zfill+'.json'
training_json = cf.json_read(training_json_fpath,True,True)

### Checks
if training_json['is_frozen'] is False:
    logging.critical('Maybe check the freezing before checking the compressing?')
    logging.critical('Aborting...')
    sys.exit(1)

### Check normal termination of DP Compress
check = 0
for it_nnp in range(1, config_json['nb_nnp'] + 1 ):
    cf.change_dir('./'+str(it_nnp))
    if Path('graph_'+str(it_nnp)+'_'+current_iteration_zfill+'_compressed.pb').is_file():
        check = check + 1
    else:
        logging.critical('DP Compress - ./'+str(it_nnp)+' not finished/failed')
    cf.change_dir('../')
del it_nnp

if check == config_json['nb_nnp']:
    training_json['is_compressed'] = True
else:
    logging.critical('Some DP Compress did not finished correctly')
    logging.critical('Please check manually before relaunching this step')
    logging.critical('Aborting...')
    sys.exit(1)
del check

training_json['is_compressed'] = True

cf.json_dump(training_json,training_json_fpath,True,'training.json')

logging.info('DP Compress is a success!')

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del training_json, training_json_fpath
del deepmd_iterative_apath

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()