## deepmd_iterative_apath
# deepmd_iterative_apath = ''
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name = 'nvs'
# allocation_name = 'v100'
# arch_name = 'v100'
# slurm_email = ''
## Training Parameters (Here are the default defaults)
# use_datasets_initial = True
# use_datasets_extra = False
# start_lr = 0.001
# stop_lr = 1e-06
# decay_rate = 0.90
# decay_steps = 5000
# stop_batch = 400000
# numb_test = 0
# deepmd_model_version = 2.1
# deepmd_model_type_descriptor = 'se_e2_a'
## Guess for initial training walltime
# initial_seconds_per_1000steps = 90

###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')

import subprocess
import numpy as np
import random

training_iterative_apath = str(Path('..').resolve())
### Check if the deepmd_iterative_apath is defined
if 'deepmd_iterative_apath' in globals():
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

### Read the config file
config_json_fpath = training_iterative_apath+'/control/config.json'
config_json = cf.json_read(config_json_fpath, abort=True)

### Get current iteration
config_json['current_iteration'] = training_iterative_apath if 'current_iteration' in globals() else cf.check_if_in_dict(config_json,'current_iteration',False,0)
current_iteration = config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

if 'arch_name' in globals() and ( arch_name != 'v100' or arch_name != 'a100' ):
    logging.critical('Invalid arch_name: '+ arch_name)
    logging.critical('Aborting...')
    sys.exit(1)

if current_iteration > 0:
    labeling_json_fpath = training_iterative_apath+'/control/labeling_'+current_iteration_zfill+'.json'
    labeling_json = cf.json_read(labeling_json_fpath, abort=True)
    if labeling_json['is_extracted'] is False:
        logging.critical('Lock found. Run/Check first: labeling4_extract.py')
        logging.critical('Aborting...')
        sys.exit(1)

### Check the cluster name
cluster = cf.check_cluster()

### Get/Create training parameters
training_json_path = training_iterative_apath+'/control/training_'+current_iteration_zfill+'.json'
training_json = cf.json_read(training_json_path, abort = False)
#TODO Is check in dict really needed?
training_json['start_lr'] = start_lr if 'start_lr' in globals() else cf.check_if_in_dict(training_json,'start_lr',0.001,1)
training_json['stop_lr'] = stop_lr if 'stop_lr' in globals() else cf.check_if_in_dict(training_json,'stop_lr',1e-06,1)
training_json['decay_rate'] = decay_rate if 'decay_rate' in globals() else cf.check_if_in_dict(training_json,'decay_rate',0.90,1)
training_json['decay_steps'] = int(decay_steps) if 'decay_steps' in globals() else cf.check_if_in_dict(training_json,'decay_steps',5000,1)
training_json['stop_batch'] = int(stop_batch) if 'stop_batch' in globals() else cf.check_if_in_dict(training_json,'stop_batch',400000,1)
training_json['numb_test'] = numb_test if 'numb_test' in globals() else cf.check_if_in_dict(training_json,'numb_test',0,1)
training_json['use_datasets_initial'] = use_datasets_initial if 'use_datasets_initial' in globals() else cf.check_if_in_dict(training_json,'use_datasets_initial',True,1)
training_json['use_datasets_extra'] = use_datasets_extra if 'use_datasets_extra' in globals() else cf.check_if_in_dict(training_json,'use_datasets_extra',False,1)
training_json['deepmd_model_version'] = deepmd_model_version if 'deepmd_model_version' in globals() else cf.check_if_in_dict(training_json,'deepmd_model_version',2.1,1)
training_json['deepmd_model_type_descriptor'] = deepmd_model_type_descriptor if 'deepmd_model_type_descriptor' in globals() else cf.check_if_in_dict(training_json,'deepmd_model_type_descriptor','se_e2_a',1)
training_json['cluster'] = cluster
training_json['project_name'] = project_name if 'project_name' in globals() else cf.check_if_in_dict(training_json,'project_name','nvs',1)
training_json['allocation_name'] = allocation_name if 'allocation_name' in globals() else cf.check_if_in_dict(training_json,'allocation_name','v100',1)
training_json['arch_name'] = arch_name if 'arch_name' in globals() else cf.check_if_in_dict(training_json,'arch_name','v100',1)

project_name = training_json['project_name']
allocation_name = training_json['allocation_name']
arch_name = training_json['arch_name']
if arch_name == 'v100' or arch_name == 'a100':
    arch_type ='gpu'

cf.check_file(deepmd_iterative_apath+'/jobs/training/job_deepmd_train_'+arch_type +'_'+cluster+'.sh',0,True,'No SLURM file present for the training step on this cluster.')
slurm_file_master = cf.read_file(deepmd_iterative_apath+'/jobs/training/job_deepmd_train_'+arch_type+'_'+cluster+'.sh')
slurm_email = '' if 'slurm_email' not in globals() else slurm_email

### Check DeePMD version
if training_json['deepmd_model_version'] not in [1.1, 1.3, 2.0, 2.1]:
    logging.critical('Invalid deepmd model version (1.1, 1.3, 2.0 or 2.1): '+str(training_json['deepmd_model_version']))
    logging.critical('Aborting...')
    sys.exit(1)
### Check DeePMD descriptor type
if training_json['deepmd_model_type_descriptor'] not in ['se_a', 'se_ar', 'se_e2_a']:
    logging.critical('Invalid deepmd type descriptor (se_a (se_e2_a) or se_ar: '+str(training_json['deepmd_model_type_descriptor']))
    logging.critical('Aborting...')
    sys.exit(1)

### Check mismatch between cluster/arch_name/arch and DeePMD
if cluster != 'jz':
    logging.critical('Only on Jean Zay !')
    logging.critical('Aborting...')
    sys.exit(1)
if training_json['deepmd_model_version'] < 2.0:
    logging.critical('Only version >= 2.0 on Jean Zay!')
    logging.critical('Aborting...')
    sys.exit(1)
if training_json['deepmd_model_version'] < 2.1 and training_json['arch_name'] == 'a100':
    logging.critical('Only version >= 2.1 on Jean Zay A100 !')
    logging.critical('Aborting...')
    sys.exit(1)

### Check mismatch between DeePMD version and Descriptor
if ((training_json['deepmd_model_type_descriptor'] == 'se_a') and ( training_json['deepmd_model_version'] == 1.1 ))\
or ((training_json['deepmd_model_type_descriptor'] == 'se_e2_a') and ( training_json['deepmd_model_version'] == 1.1 ))\
or ((training_json['deepmd_model_type_descriptor'] == 'se_ar') and ( training_json['deepmd_model_version'] == 2.0 ))\
or ((training_json['deepmd_model_type_descriptor'] == 'se_ar') and ( training_json['deepmd_model_version'] == 2.1 )):
    logging.critical('Invalid DeePMD Version/Descriptor pair: '+str(training_json['deepmd_model_version'])+'/'+str(training_json['deepmd_model_type_descriptor']))
    logging.critical('Aborting...')
    sys.exit(1)

### Descriptor name equivalence
if ((training_json['deepmd_model_type_descriptor'] == 'se_a') and ( training_json['deepmd_model_version'] == 2.0 ))\
    or ((training_json['deepmd_model_type_descriptor'] == 'se_a') and ( training_json['deepmd_model_version'] == 2.1 )):
        training_json['deepmd_model_type_descriptor'] = 'se_e2_a'
elif ((training_json['deepmd_model_type_descriptor'] == 'se_e2_a') and ( training_json['deepmd_model_version'] == 1.3 )):
        training_json['deepmd_model_type_descriptor'] = 'se_a'

### Check if the default input json file exists
input_file_fpath = str(Path(training_iterative_apath+'/inputs/'+str(training_json['deepmd_model_version'])+'_'+str(training_json['deepmd_model_type_descriptor'])+'.json').resolve())
training_input_json = cf.json_read(input_file_fpath, abort = False)

### Check the initial sets json file
datasets_initial_json = cf.check_datasets_initial(training_iterative_apath)

### Let us find what is in data
subsys_name=[]

#TODO IMPLEMENT TEST LIST FOR VALIDATION ? If DeepMD version >= 2.0

datasets_extra=[]
datasets_validation=[]
for it_data_folders in Path(training_iterative_apath+'/data').iterdir():
    if it_data_folders.is_dir():
    ### Escape initial/extra sets, because initial get added first and extra as last
        if it_data_folders.name not in datasets_initial_json.keys() and 'extra_' != it_data_folders.name[:6]:
            ### Escape test sets
            if 'test_' != it_data_folders.name[:5]:
                ### Escape if set iter is superior as iter, it is only for reprocessing old stuff.
                try:
                    if int(it_data_folders.name.rsplit('_',1)[-1]) <= current_iteration:
                        subsys_name.append(it_data_folders.name.rsplit('_',1)[0])
                except:
                    pass
            else:
                datasets_validation.append(it_data_folders.name)
        ### Get the extra sets !
        elif 'extra_' == it_data_folders.name[:6]:
            datasets_extra.append(it_data_folders.name)

### Training sets list construction
datasets_training=[]

### Initial
structures_initial_total = 0
if training_json['use_datasets_initial']:
    for it_datasets_initial_json in datasets_initial_json.keys():
        if Path(training_iterative_apath+'/data/'+it_datasets_initial_json).is_dir():
            datasets_training.append('data/'+it_datasets_initial_json+'/')
            structures_initial_total = structures_initial_total+datasets_initial_json[it_datasets_initial_json]

### Non-Reactive (aka subsys_nr in the initialization first) && all the others are REACTIVE !
### Total and what is added just for this iteration
structures_added_nr_total = 0
structures_added_r_total = 0
structures_added_nr_iter = 0
structures_added_r_iter = 0

### This trick remove duplicates from list via set
subsys_name = list(set(subsys_name))
subsys_name = [i for i in subsys_name if i not in config_json['subsys_nr']]
subsys_name = [i for i in subsys_name if i not in [zzz + '-disturbed' for zzz in config_json['subsys_nr']]]
subsys_name = sorted(subsys_name)
config_json['subsys_r'] = subsys_name

if current_iteration > 0:
    for it_iteration in np.arange(1,current_iteration+1):
        try:
            for system_it in config_json['subsys_nr']:
                if Path(training_iterative_apath+'/data/'+system_it+'_'+str(it_iteration).zfill(3)).is_dir():
                    datasets_training.append('data/'+system_it+'_'+str(it_iteration).zfill(3)+'/')
                    structures_added_nr_total = structures_added_nr_total+np.load(training_iterative_apath+'/data/'+system_it+'_'+str(it_iteration).zfill(3)+'/set.000/box.npy').shape[0]
                    if it_iteration == current_iteration:
                        structures_added_nr_iter = structures_added_nr_iter+np.load(training_iterative_apath+'/data/'+system_it+'_'+str(it_iteration).zfill(3)+'/set.000/box.npy').shape[0]
        except(KeyError,NameError):
            pass
        try:
            for system_it in [zzz + '-disturbed' for zzz in config_json['subsys_nr']]:
                if Path(training_iterative_apath+'/data/'+system_it+'_'+str(it_iteration).zfill(3)).is_dir():
                    datasets_training.append('data/'+system_it+'_'+str(it_iteration).zfill(3)+'/')
                    structures_added_nr_total = structures_added_nr_total+np.load(training_iterative_apath+'/data/'+system_it+'_'+str(it_iteration).zfill(3)+'/set.000/box.npy').shape[0]
                    if it_iteration == current_iteration:
                        structures_added_nr_iter = structures_added_nr_iter+np.load(training_iterative_apath+'/data/'+system_it+'_'+str(it_iteration).zfill(3)+'/set.000/box.npy').shape[0]
        except(KeyError,NameError):
            pass
        try:
            for system_it in config_json['subsys_r']:
                if Path(training_iterative_apath+'/data/'+system_it+'_'+str(it_iteration).zfill(3)).is_dir():
                    datasets_training.append('data/'+system_it+'_'+str(it_iteration).zfill(3)+'/')
                    structures_added_r_total = structures_added_r_total+np.load(training_iterative_apath+'/data/'+system_it+'_'+str(it_iteration).zfill(3)+'/set.000/box.npy').shape[0]
                    if it_iteration == current_iteration:
                        structures_added_r_iter = structures_added_r_iter+np.load(training_iterative_apath+'/data/'+system_it+'_'+str(it_iteration).zfill(3)+'/set.000/box.npy').shape[0]
        except(KeyError,NameError):
            pass

### Finally the extra sets !
structures_extra_total = 0
if training_json['use_datasets_extra']:
    config_json['datasets_extra'] = datasets_extra
    for it_datasets_extra in config_json['datasets_extra']:
        datasets_training.append('data/'+it_datasets_extra+'/')
        structures_extra_total = structures_extra_total+np.load(training_iterative_apath+'/data/'+it_datasets_extra+'/set.000/box.npy').shape[0]

### Total
structures_trained_total = structures_initial_total+structures_added_nr_total+structures_added_r_total+structures_extra_total

### Number of tests
if ( training_json['deepmd_model_version'] < 2.0 ):
    training_input_json['training']['numb_test'] = training_json['numb_test']

#TODO IMPLEMENT If there is validation sets because >= 2.0, maybe enforce numb_test to not 0??
#TODO IMPLEMENT If they appeared ? Maybe in the exploration extract if discarded ones, keep 10/20 to grow a validation ?

### Because changes beteween version
if ( training_json['deepmd_model_version'] >= 2.0 ):
    training_input_json['training']['training_data']['systems'] = datasets_training
else:
    training_input_json['training']['systems'] = datasets_training

training_json['structures_trained_total'] = structures_trained_total
training_json['structures_initial_total'] = structures_initial_total
training_json['structures_added_nr_total'] = structures_added_nr_total
training_json['structures_added_r_total'] = structures_added_r_total
training_json['structures_added_nr_iter'] = structures_added_nr_iter
training_json['structures_added_r_iter'] = structures_added_r_iter
training_json['structures_extra_total'] = structures_extra_total

### If no override, get decay steps (= nb of trained floored to the nearest 10000 divided by 4)
if 'decay_steps' not in globals():
    decay_steps = cf.get_decay_steps(structures_trained_total)

training_json['decay_steps'] = int(decay_steps)
decay_steps = int(decay_steps)

### THE MAGIC IS HERE
### Priority is: GOOD LUCK
if 'decay_rate' in globals() and 'stop_lr' not in globals():
    if 'stop_batch' not in globals():
        stop_batch = training_json['stop_batch']
    stop_lr_new = cf.get_learning_rate(stop_batch,training_json['start_lr'],decay_rate,training_json['decay_steps'])
    if 'stop_batch' not in globals():
        while stop_lr_new > training_json['stop_lr']:
            stop_batch = stop_batch+1e5
            stop_lr_new = cf.get_learning_rate(stop_batch,training_json['start_lr'],decay_rate,training_json['decay_steps'])
    training_json['stop_batch'] = int(stop_batch)
    training_json['stop_lr'] = stop_lr_new
elif 'decay_rate' in globals() and 'stop_lr' in globals() and 'stop_batch' in globals():
    stop_lr_new = cf.get_learning_rate(stop_batch,training_json['start_lr'],decay_rate,decay_steps)
    if stop_lr_new > stop_lr:
        while stop_lr_new > stop_lr:
            decay_steps = decay_steps-1000
            stop_lr_new = cf.get_learning_rate(stop_batch,training_json['start_lr'],decay_rate,decay_steps)
    else:
        while stop_lr_new < stop_lr:
            decay_steps = decay_steps+1000
            stop_lr_new = cf.get_learning_rate(stop_batch,training_json['start_lr'],decay_rate,decay_steps)
    training_json['decay_steps'] = int(decay_steps)
    decay_rate_new = cf.get_decay_rate(stop_batch,training_json['start_lr'],stop_lr,training_json['decay_steps'])
    training_json['decay_rate'] = decay_rate_new
else:
    if 'stop_lr' not in globals():
        stop_lr = training_json['stop_lr']
    stop_batch = training_json['stop_batch']
    decay_rate_new = cf.get_decay_rate(stop_batch,training_json['start_lr'],stop_lr,training_json['decay_steps'])
    while decay_rate_new < training_json['decay_rate']:
        stop_batch = stop_batch+1e5
        decay_rate_new = cf.get_decay_rate(stop_batch,training_json['start_lr'],stop_lr,training_json['decay_steps'])
    training_json['stop_batch'] = int(stop_batch)
    training_json['decay_rate'] = decay_rate_new

if ( training_json['deepmd_model_version'] >= 2.0 ):
    training_input_json['training']['numb_steps'] = training_json['stop_batch']
else:
    training_input_json['training']['stop_batch'] = training_json['stop_batch']

training_input_json['learning_rate']['decay_steps'] = training_json['decay_steps']

if (training_json['deepmd_model_version'] >= 1.3):
    training_input_json['learning_rate']['stop_lr'] = training_json['stop_lr']
else:
    training_input_json['learning_rate']['decay_rate'] = training_json['decay_rate']

### Set frozen/compressed bool !
training_json['is_locked'] = True
training_json['is_launched'] = False
training_json['is_checked'] = False
training_json['is_frozen'] = False
training_json['is_compressed'] = False

logging.info(training_json)
logging.info(datasets_training)

### Rsync data to local data
cf.create_dir('./data')
for it_datasets_training in datasets_training:
    subprocess.call(['rsync','-a', training_iterative_apath+'/'+it_datasets_training.rsplit('/',1)[0], './data'])

### Change some inside output
training_input_json['training']['disp_file']='lcurve.out'
training_input_json['training']['save_ckpt']='model.ckpt'

### It doesn't exists anymore :(
if training_json['deepmd_model_version'] < 2.0:
    training_input_json['training']['load_ckpt']='model.ckpt'

## Dump the config/training
cf.json_dump(config_json,config_json_fpath, True, 'config file')
cf.json_dump(training_json,training_json_path, True, 'training config file')

### Create the inputs/jobfiles for each NNP with random SEED inf the form of NNP_number + random(0,1000) + current_iteration.zfil(3) so between 10000 and unlimited1000999 (at iteration 999 !!)
if current_iteration > 0:
    previous_iteration = current_iteration - 1
    previous_iteration_zfill = str(previous_iteration).zfill(3)
    prevtraining_json_fpath = training_iterative_apath+'/control/training_'+previous_iteration_zfill+'.json'
    prevtraining_json = cf.json_read(prevtraining_json_fpath, abort=True)
    approx_time = int(np.ceil((stop_batch*(prevtraining_json['avg_seconds_per_step']+0.25*prevtraining_json['avg_seconds_per_step'])/3600)))
    del previous_iteration, previous_iteration_zfill, prevtraining_json_fpath, prevtraining_json
else:
    initial_seconds_per_1000steps = initial_seconds_per_1000steps if 'initial_seconds_per_1000steps' in globals() else 90
    approx_time = int(np.ceil((stop_batch*initial_seconds_per_1000steps/1000/3600)))

if approx_time > 100:
    approx_time = 100

for it_nnp in range(1,config_json['nb_nnp'] + 1):
    cf.create_dir(str(it_nnp))
    random.seed()
    RAND = random.randrange(0,1000)
    if training_json['deepmd_model_type_descriptor'] == 'se_ar':
        training_input_json['model']['descriptor']['a']['seed'] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)
        training_input_json['model']['descriptor']['r']['seed'] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)
    else:
        training_input_json['model']['descriptor']['seed'] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)

    training_input_json['model']['fitting_net']['seed'] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)
    training_input_json['training']['seed'] = int(str(it_nnp)+str(RAND)+current_iteration_zfill)

    training_input_json_fpath = str(Path(str(it_nnp)+'/training.json').resolve())
    cf.json_dump(training_input_json,training_input_json_fpath, True, 'deepmd training input file')

    slurm_file = slurm_file_master
    slurm_file = cf.replace_in_list(slurm_file,'_PROJECT_',project_name)
    slurm_file = cf.replace_in_list(slurm_file,'_WALLTIME_',str(approx_time)+':00:00')
    slurm_file = cf.replace_in_list(slurm_file,'SET_DEEPMD_MODEL_VERSION',str(training_json['deepmd_model_version']))
    if allocation_name == 'v100':
        slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_',allocation_name)
        if approx_time <= 20:
            if arch_name == 'v100':
                slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t3')
                slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p13')
                slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
            else:
                slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t3')
                slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p4')
                slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
        else:
            slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t4')
            slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p13')
            slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
    elif allocation_name == 'a100':
        slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_',allocation_name)
        if approx_time <= 20:
            slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t3')
        else:
            slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t4')
        slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p5')
        slurm_file = cf.replace_in_list(slurm_file,'_SUBPARTITION_','a100')
    else:
        logging.critical('Unknown error. Please BUG REPORT')
        logging.critical('Aborting')
        sys.exit(1)
    if slurm_email != '':
        slurm_file = cf.replace_in_list(slurm_file,'##SBATCH --mail-type','#SBATCH --mail-type')
        slurm_file = cf.replace_in_list(slurm_file,'##SBATCH --mail-user _EMAIL_','#SBATCH --mail-user '+slurm_email)

    cf.write_file(str(it_nnp)+'/job_deepmd_train_'+arch_type+'_'+cluster+'.sh',slurm_file)

logging.info('Preparation of the training is a success!')

del sys, Path, logging, cf
del subprocess, np, random
import gc; gc.collect(); del gc
exit()