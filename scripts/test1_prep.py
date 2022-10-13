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

training_json_fpath = training_iterative_apath+'/control/training_'+current_iteration_zfill+'.json'
training_json = cf.json_read(training_json_fpath,True,True)

test_json_fpath = training_iterative_apath+'/control/test_'+current_iteration_zfill+'.json'
test_json = cf.json_read(test_json_fpath,False,True)

if training_json['is_frozen'] is False:
    logging.critical('Lock found. Previous NNPs aren\'t frozen')
    logging.critical('Aborting...')
    sys.exit(1)

cluster = cf.check_cluster()

### Set needed variables
test_json['nb_nnp'] = config_json['nb_nnp']
test_json['is_compressed'] = training_json['is_compressed']
test_json['cluster'] = cluster
test_json['project_name'] = project_name if 'project_name' in globals() else 'nvs'
test_json['allocation_name'] = allocation_name if 'allocation_name' in globals() else 'v100'
test_json['arch_name'] = arch_name if 'arch_name' in globals() else 'v100'
project_name = test_json['project_name']
allocation_name = test_json['allocation_name']
arch_name = test_json['arch_name']
if arch_name == 'v100' or arch_name == 'a100':
    arch_type ='gpu'
slurm_email = '' if 'slurm_email' not in globals() else slurm_email

cf.check_file(deepmd_iterative_apath+'/jobs/test/job_deepmd_test_'+arch_type+'_'+cluster+'.sh',0,True,'No SLURM file present for the exploration phase on this cluster.')
slurm_file_master = cf.read_file(deepmd_iterative_apath+'/jobs/test/job_deepmd_test_'+arch_type+'_'+cluster+'.sh')

###
compressed = '_compressed' if test_json['is_compressed'] else ''
for it_nnp in range(1, test_json['nb_nnp'] + 1):
    slurm_file = slurm_file_master
    cf.check_file('../NNP/graph_'+str(it_nnp)+'_'+current_iteration_zfill+compressed+'.pb',0,True)
    subprocess.call(['rsync','-a', '../NNP/graph_'+str(it_nnp)+'_'+current_iteration_zfill+compressed+'.pb', './'])
    slurm_file = cf.replace_in_list(slurm_file,'_DEEPMD_PB_','graph_'+str(it_nnp)+'_'+current_iteration_zfill+compressed)
    slurm_file = cf.replace_in_list(slurm_file,'_NNPNB_','NNP'+str(it_nnp))
    slurm_file = cf.replace_in_list(slurm_file,'_PROJECT_',project_name)
    slurm_file = cf.replace_in_list(slurm_file,'_WALLTIME_','04:00:00')
    slurm_file = cf.replace_in_list(slurm_file,'_DEEPMD_MODEL_VERSION_',str(training_json['deepmd_model_version']))

    if allocation_name == 'v100':
        slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_',allocation_name)
        if arch_name == 'v100':
            slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t3')
            slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p13')
            slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
        elif arch_name == 'a100':
            slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t3')
            slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p4')
            slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
    elif allocation_name == 'a100':
        slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_',allocation_name)
        slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t3')
        slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p5')
        slurm_file = cf.replace_in_list(slurm_file,'_SUBPARTITION_',arch_name)
    else:
        logging.critical('Unknown error. Please BUG REPORT')
        logging.critical('Aborting...')
        sys.exit(1)
    if slurm_email != '':
        slurm_file = cf.replace_in_list(slurm_file,'##SBATCH --mail-type','#SBATCH --mail-type')
        slurm_file = cf.replace_in_list(slurm_file,'##SBATCH --mail-user _EMAIL_','#SBATCH --mail-user '+slurm_email)
    cf.write_file('./job_deepmd_test_'+arch_type+'_'+cluster+'_NNP'+str(it_nnp)+'.sh',slurm_file)

    del slurm_file
del it_nnp

test_json['is_locked'] = True
test_json['is_launched'] = False
test_json['is_checked'] = False
test_json['is_concatenated'] = False
test_json['is_plotted'] = False

cf.json_dump(test_json,test_json_fpath,True,'test.json')
logging.info('The DP-Test: prep phase is a success!')

del slurm_file_master

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del training_json, training_json_fpath
del test_json, test_json_fpath
del current_iteration, current_iteration_zfill
del cluster, arch_type
del project_name, allocation_name, arch_name
del deepmd_iterative_apath
del slurm_email

del sys, Path, logging, cf
del subprocess
import gc; gc.collect(); del gc
exit()