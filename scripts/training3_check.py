###################################### No change past here
import logging
import sys
from pathlib import Path
import numpy as np
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

if training_json['is_launched'] is False:
    logging.critical('Maybe launch the training before checking?')
    logging.critical('Aborting')
    sys.exit(1)

del current_iteration, current_iteration_zfill

time_per_step=[]
check = 0
for it_nnp in range(1, config_json['nb_nnp'] + 1):
    cf.change_dir('./'+str(it_nnp))
    cf.check_file('training.out',0,1,'No out file found for '+str(it_nnp)+'.')
    if Path('training.out').is_file():
        training_out = cf.read_file('training.out')
        if any('finished training' in s for s in training_out):
            training_out_time=[s for s in training_out if "training time" in s]
            training_out_time_split=[]
            for n in range(0,len(training_out_time)):
                training_out_time_split.append(training_out_time[n].split(' '))
                training_out_time_split[n]=' '.join(training_out_time_split[n]).split()
            if Path('model.ckpt-'+str(training_out_time_split[-1][3])+'.index').is_file():
                Path('model.ckpt-'+str(training_out_time_split[-1][3])+'.index').rename('model.ckpt.index')
                Path('model.ckpt-'+str(training_out_time_split[-1][3])+'.meta').rename('model.ckpt.meta')
                Path('model.ckpt-'+str(training_out_time_split[-1][3])+'.data-00000-of-00001').rename('model.ckpt.data-00000-of-00001')
            for n in range(0,len(training_out_time_split)):
                time_per_step.append(float(training_out_time_split[n][6]))
            del n
            step_size = float(training_out_time_split[-1][3])-float(training_out_time_split[-2][3])
            check = check + 1
        else:
            logging.warning('Training of '+str(it_nnp)+' not finished/failed.')
        del training_out, training_out_time, training_out_time_split
    cf.change_dir('..')

if check == config_json['nb_nnp']:
    training_json['is_checked'] = True
del check

del it_nnp, config_json, config_json_fpath

if ( 'time_per_step' in globals() ) and ( 'step_size' in globals() ):
    training_json['avg_seconds_per_step']=np.average(time_per_step)/(step_size)
    del time_per_step, step_size

cf.json_dump(training_json, training_json_fpath, True, 'deepmd training input file')

logging.info('DP Train success')

del training_json, training_json_fpath, deepmd_iterative_path, training_iterative_apath

del sys, Path, np, logging, cf
import gc; gc.collect(); del gc
exit()