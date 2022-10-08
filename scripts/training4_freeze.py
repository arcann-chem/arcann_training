####### Project name / AllocType/GPUType -- > v100/v100 or v100/a100 or a100/a100
project_name='nvs'
allocation_name='v100'
arch_name='v100'

####### Default no email
#slurm_email=''

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
current_iteration = config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

training_json_fpath = training_iterative_apath+'/control/training_'+current_iteration_zfill+'.json'
training_json = cf.json_read(training_json_fpath, abort=True)
deepmd_model_version = str(training_json['deepmd_model_version'])

if training_json['is_checked'] is False:
    logging.critical('Maybe check the training before freezing?')
    logging.critical('Aborting...')
    sys.exit(1)
    
if arch_name == 'v100' or arch_name == 'a100':
    arch_type='gpu'

### Check the cluster name
cluster = cf.check_cluster()
cf.check_file(deepmd_iterative_path+'/jobs/training/job_deepmd_freeze_'+arch_type+'_'+cluster+'.sh',0,0)
slurm_file_master = cf.read_file(deepmd_iterative_path+'/jobs/training/job_deepmd_freeze_'+arch_type+'_'+cluster+'.sh')
slurm_email = '' if 'slurm_email' not in globals() else slurm_email

for it_nnp in range(1, config_json['nb_nnp'] + 1):
    cf.change_dir('./'+str(it_nnp))
    cf.check_file('./model.ckpt.index',0,0)
    slurm_file = slurm_file_master
    slurm_file = cf.replace_in_list(slurm_file,'_PROJECT_',project_name)
    slurm_file = cf.replace_in_list(slurm_file,'_WALLTIME_','01:00:00')
    slurm_file = cf.replace_in_list(slurm_file,'SET_DEEPMD_MODEL_VERSION',str(deepmd_model_version))
    slurm_file = cf.replace_in_list(slurm_file,'DeepMD_Freeze','DeepMD_Freeze')
    slurm_file = cf.replace_in_list(slurm_file,'DeepMD_PB_F','graph_'+str(it_nnp)+'_'+current_iteration_zfill)
    if allocation_name == 'v100':
        slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_','v100')
        if arch_name == 'v100':
            slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t3')
            slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p13')
            slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
        else:
            slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t3')
            slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p4')
            slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
    elif allocation_name == 'a100':
        slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_','a100')
        slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t3')
        slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p5')
        slurm_file = cf.replace_in_list(slurm_file,'_SUBPARTITION_','a100')
    else:
        sys.exit('Unknown error. Please BUG REPORT.\n Aborting...')
    cf.write_file('job_deepmd_freeze_'+arch_type+'_'+cluster+'.sh',slurm_file)
    with open('checkpoint', 'w') as f:
        f.write('model_checkpoint_path: \"model.ckpt\"\n')
        f.write('all_model_checkpoint_paths: \"model.ckpt\"\n')
        f.close()
    del f
    cf.check_file('job_deepmd_freeze_'+arch_type+'_'+cluster+'.sh',0,1,'Training NNP '+str(it_nnp)+' was not lauched. No job file found.')
    if Path('job_deepmd_freeze_'+arch_type+'_'+cluster+'.sh').is_file():
        subprocess.call(['sbatch','./job_deepmd_freeze_'+arch_type+'_'+cluster+'.sh'])
    cf.change_dir('..')

del it_nnp, config_json, config_json_fpath, deepmd_model_version, deepmd_iterative_path, slurm_email, slurm_file, slurm_file_master, allocation_name, arch_name, project_name, training_json, training_json_fpath
del cluster, training_iterative_apath, current_iteration, current_iteration_zfill, arch_type

del sys, Path, subprocess, logging, cf
import gc; gc.collect(); del gc
exit()