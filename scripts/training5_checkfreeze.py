###################################### No change past here
import sys
from pathlib import Path
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
current_iteration = config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

training_json_fpath = training_iterative_apath+'/control/training_'+current_iteration_zfill+'.json'
training_json = cf.json_read(training_json_fpath, abort=True)
deepmd_model_version = str(training_json['deepmd_model_version'])

for it_nnp in range(1, config_json['nb_nnp'] + 1):
    cf.change_dir('./'+str(it_nnp))
    cf.check_file('graph_'+str(it_nnp)+'_'+current_iteration_zfill+'.pb',0,0)
    cf.change_dir('../')

training_json['is_frozen'] = True

cf.json_dump(training_json,training_json_fpath, True, 'training config file')

logging.info('DP Freeze success')

del it_nnp, config_json, config_json_fpath, deepmd_model_version, deepmd_iterative_path, training_json, training_json_fpath
del training_iterative_apath, current_iteration, current_iteration_zfill

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()