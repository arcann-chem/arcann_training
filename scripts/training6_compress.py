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
if Path(training_iterative_apath+'/control/path').is_file():
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

### Set needed variables
project_name = project_name if 'project_name' in globals() else training_json['project_name']
allocation_name = allocation_name if 'allocation_name' in globals() else training_json['allocation_name']
arch_name = arch_name if 'arch_name' in globals() else training_json['arch_name']
if arch_name == 'v100' or arch_name == 'a100':
    arch_type ='gpu'

### Checks
if training_json['is_frozen'] is False:
    logging.critical('Maybe freeze the NNPs before compressing?')
    logging.critical('Aborting...')
    sys.exit(1)

cluster = cf.check_cluster()
cf.check_file(deepmd_iterative_apath+'/jobs/training/job_deepmd_freeze_'+arch_type+'_'+cluster+'.sh',0,True,'No SLURM file present for the compressing step on this cluster.')

### Prep and launch the jobs
slurm_file_master = cf.read_file(deepmd_iterative_apath+'/jobs/training/job_deepmd_compress_'+arch_type+'_'+cluster+'.sh')
slurm_email = '' if 'slurm_email' not in globals() else slurm_email

for it_nnp in range(1, config_json['nb_nnp'] + 1):
    cf.change_dir('./'+str(it_nnp))
    cf.check_file('graph_'+str(it_nnp)+'_'+current_iteration_zfill+'.pb',0,0)
    slurm_file = slurm_file_master
    slurm_file = cf.replace_in_list(slurm_file,'_PROJECT_',project_name)
    slurm_file = cf.replace_in_list(slurm_file,'_WALLTIME_','02:00:00')
    slurm_file = cf.replace_in_list(slurm_file,'SET_DEEPMD_MODEL_VERSION',str(training_json['deepmd_model_version']))
    slurm_file = cf.replace_in_list(slurm_file,'DeepMD_Compress','DeepMD_Compress')
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
    cf.write_file('job_deepmd_compress_'+arch_type+'_'+cluster+'.sh',slurm_file)
    if Path('job_deepmd_compress_'+arch_type+'_'+cluster+'.sh').is_file():
        subprocess.call(['sbatch','./job_deepmd_compress_'+arch_type+'_'+cluster+'.sh'])
        logging.info('Compressing of NNP '+str(it_nnp)+' was lauched.')
    else:
        logging.warning('Compressing of NNP '+str(it_nnp)+' was not lauched. No job file found.')

    cf.change_dir('..')
del it_nnp, slurm_file, slurm_file_master

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del training_json, training_json_fpath
del cluster, arch_type
del deepmd_iterative_apath
del project_name, allocation_name, arch_name
del slurm_email

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()