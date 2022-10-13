## deepmd_iterative_apath
# deepmd_iterative_apath = ''
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name = 'nvs'
# allocation_name = 'dev'
# arch_name = 'cpu'
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

### Remove previous obsolete slurm outputs
cf.remove_file_glob('./','DeepMD_Test_Concatenation.*')

### Checks
if test_json['is_concatenated'] is False:
    logging.critical('Lock found. Run/Check first: test4_concatenation.py')
    logging.critical('Aborting...')
    sys.exit(1)
cluster = cf.check_cluster()

### Set needed variables
test_json['cluster_2'] = cluster
test_json['project_name_2'] = project_name if 'project_name' in globals() else test_json['project_name_2']
test_json['allocation_name_2'] = allocation_name if 'allocation_name' in globals() else test_json['allocation_name_2']
test_json['arch_name_2'] = arch_name if 'arch_name' in globals() else test_json['arch_name_2']
project_name = test_json['project_name_2']
allocation_name = test_json['allocation_name_2']
arch_name = test_json['arch_name_2']
if arch_name == 'cpu':
    arch_type ='cpu'
slurm_email = '' if 'slurm_email' not in globals() else slurm_email

cf.check_file(deepmd_iterative_apath+'/jobs/test/job_deepmd_test_plot_'+arch_type+'_'+cluster+'.sh',0,True,'No SLURM file present for the plotting phase on this cluster.')
slurm_file = cf.read_file(deepmd_iterative_apath+'/jobs/test/job_deepmd_test_plot_'+arch_type+'_'+cluster+'.sh')
slurm_file = cf.replace_in_list(slurm_file,'_PROJECT_',project_name)
slurm_file = cf.replace_in_list(slurm_file,'_WALLTIME_','02:00:00')
if allocation_name == 'prepost':
    slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_','cpu')
    slurm_file = cf.replace_in_list(slurm_file,'#SBATCH --qos=_QOS_','##SBATCH --qos=_QOS_')
    slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','prepost')
    slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
elif allocation_name == 'dev':
    slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_','cpu')
    slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_cpu-dev')
    slurm_file = cf.replace_in_list(slurm_file,'#SBATCH --partition=_PARTITION_','##SBATCH --partition=_PARTITION_')
    slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
elif allocation_name == 'cpu':
    slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_','cpu')
    slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_cpu-t3')
    slurm_file = cf.replace_in_list(slurm_file,'#SBATCH --partition=_PARTITION_','##SBATCH --partition=_PARTITION_')
    slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
else:
    logging.critical('Unknown error. Please BUG REPORT')
    logging.critical('Aborting...')
    sys.exit(1)
if slurm_email != '':
    slurm_file = cf.replace_in_list(slurm_file,'##SBATCH --mail-type','#SBATCH --mail-type')
    slurm_file = cf.replace_in_list(slurm_file,'##SBATCH --mail-user _EMAIL_','#SBATCH --mail-user '+slurm_email)
cf.write_file('./job_deepmd_test_plot_'+arch_type+'_'+cluster+'.sh',slurm_file)
del slurm_file

cf.check_file(deepmd_iterative_apath+'/scripts/_deepmd_test_plot.py',0,True)
python_file = cf.read_file(deepmd_iterative_apath+'/scripts/_deepmd_test_plot.py')
python_file = cf.replace_in_list(python_file,'_DEEPMD_ITERATIVE_APATH_',str(deepmd_iterative_apath))
cf.write_file('./_deepmd_test_plot.py',python_file)
del python_file
logging.info('The DP-Test: plot-prep phase is a success!')

subprocess.call(['sbatch','./job_deepmd_test_plot_'+arch_type+'_'+cluster+'.sh'])
logging.info('The DP-Test: plot-SLURM phase is a success!')

cf.json_dump(test_json,test_json_fpath,True,'test.json')

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
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