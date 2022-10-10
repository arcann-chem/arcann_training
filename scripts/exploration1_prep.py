## deepmd_iterative_apath
# deepmd_iterative_apath = ''
## Project name / allocation / arch (nvs/v100/a100 or gen7156/rome/cpu)
# project_name = 'nvs'
# allocation_name = 'v100'
# arch_name = 'v100'
# slurm_email = ''
## These are the default
# temperature_K = [300, 300]
# timestep_ps = [0.0005, 0.0005]
## print_freq is every 1% / nb_steps_exploration is initial/auto-calculated (local subsys)
## These are the default
# nb_steps_exploration = [20000, 20000]
# print_freq = [200, 200]
# nb_steps_exploration_initial = 20000
# nb_traj = 2

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
config_json = cf.json_read(config_json_fpath, abort=True)

current_iteration = current_iteration if 'current_iteration' in globals() else config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

previous_iteration = current_iteration-1
previous_iteration_zfill = str(previous_iteration).zfill(3)

prevtraining_json_fpath = training_iterative_apath+'/control/training_'+previous_iteration_zfill+'.json'
prevtraining_json = cf.json_read(prevtraining_json_fpath,abort=True)

exploration_json_fpath = training_iterative_apath+'/control/exploration_'+current_iteration_zfill+'.json'
exploration_json = cf.json_read(exploration_json_fpath,abort=False)

### Checks
if prevtraining_json['is_frozen'] is False:
    logging.critical('Lock found. Previous NNPs aren\'t frozen')
    logging.critical('Aborting...')
    sys.exit(1)

cluster = cf.check_cluster()

### Set needed variables
exploration_json['cluster'] = cluster
exploration_json['project_name'] = project_name if 'project_name' in globals() else 'nvs'
exploration_json['allocation_name'] = allocation_name if 'allocation_name' in globals() else 'v100'
exploration_json['arch_name'] = arch_name if 'arch_name' in globals() else 'v100'
exploration_json['deepmd_model_version'] = prevtraining_json['deepmd_model_version']
exploration_json['nb_subsys_nr'] = len(config_json['subsys_nr'])
exploration_json['nb_nnp'] = config_json['nb_nnp']
project_name = exploration_json['project_name']
allocation_name = exploration_json['allocation_name']
arch_name = exploration_json['arch_name']

if arch_name == 'cpu':
    arch_type ='cpu'
slurm_email = '' if 'slurm_email' not in globals() else slurm_email

### Checks
cf.check_file(deepmd_iterative_apath+'/jobs/exploration/job_deepmd_lammps_'+arch_type+'_'+cluster+'.sh',0,True,'No SLURM file present for the exploration phase on this cluster.')

### Preparation of the exploration
if previous_iteration > 0:
    prevexploration_json_fpath = training_iterative_apath+'/control/exploration_'+previous_iteration_zfill+'.json'
    prevexploration_json = cf.json_read(prevexploration_json_fpath,abort=True)
    prevdeepmd_model_version = prevexploration_json['deepmd_model_version']

nb_steps_exploration_initial = 20000 if 'nb_steps_exploration_initial' not in globals() else nb_steps_exploration_initial
nb_traj = 2 if 'nb_traj' not in globals() else int(nb_traj)
exploration_json['nb_traj'] = int(nb_traj)
slurm_email = '' if 'slurm_email' not in globals() else slurm_email

exploration_json['subsys_nr']={}
for it0_subsys_nr,it_subsys_nr in enumerate(config_json['subsys_nr']):
    cf.create_dir('./'+it_subsys_nr)
    random.seed()
    exploration_json['subsys_nr'][it_subsys_nr]={}
    temperature = temperature_K[it0_subsys_nr] if 'temperature_K' in globals() else config_json['subsys_nr'][it_subsys_nr]['temperature_K']
    timestep = timestep_ps[it0_subsys_nr] if 'timestep_ps' in globals() else config_json['subsys_nr'][it_subsys_nr]['timestep_ps']
    exploration_json['subsys_nr'][it_subsys_nr]['temperature_K'] = temperature
    exploration_json['subsys_nr'][it_subsys_nr]['timestep_ps'] = timestep
    if current_iteration == 1:
        if Path(training_iterative_apath+'/inputs/'+it_subsys_nr+'.lmp').is_file():
            exploration_type = 'lammps'
    else:
        exploration_type = config_json['subsys_nr'][it_subsys_nr]['exploration_type']
        starting_point_list_path = [zzz for zzz in Path(training_iterative_apath+'/starting_structures').glob(previous_iteration_zfill+'_'+it_subsys_nr+'_*.lmp')]
        starting_point_list = [str(zzz).split('/')[-1] for zzz in starting_point_list_path]
        starting_point_list_bckp = starting_point_list
        del starting_point_list_path

    exploration_json['subsys_nr'][it_subsys_nr]['exploration_type'] = exploration_type
    slurm_file_master = cf.read_file(deepmd_iterative_apath+'/jobs/exploration/job_deepmd_'+exploration_type+'_'+arch_type+'_'+cluster+'.sh')
    for it_nnp in range(1, config_json['nb_nnp'] + 1 ):
        cf.create_dir('./'+it_subsys_nr+'/'+str(it_nnp))
        for it_number in range(1, nb_traj + 1):
            local_path='./'+it_subsys_nr+'/'+str(it_nnp)+'/'+str(it_number).zfill(5)
            cf.create_dir(local_path)
            list_nnp = [zzz for zzz in range(1, config_json['nb_nnp'] + 1)]
            reorder_nnp_list = list_nnp[list_nnp.index(it_nnp):] + list_nnp[:list_nnp.index(it_nnp)]
            del list_nnp
            if prevtraining_json['is_compressed'] == True:
                models_list=['graph_'+str(f)+'_'+previous_iteration_zfill+'_compressed.pb' for f in reorder_nnp_list]
                for it_sub_nnp in range(1, config_json['nb_nnp'] + 1 ):
                    nnp_apath = Path(training_iterative_apath+'/NNP/graph_'+str(it_sub_nnp)+'_'+previous_iteration_zfill+'_compressed.pb').resolve()
                    subprocess.call(['ln','-s', str(nnp_apath), local_path+'/'])
                del it_sub_nnp,nnp_apath
            else:
                models_list=['graph_'+str(f)+'_'+previous_iteration_zfill+'.pb' for f in reorder_nnp_list]
                for it_sub_nnp in range(1, config_json['nb_nnp'] + 1 ):
                    nnp_apath = Path(training_iterative_apath+'/NNP/graph_'+str(it_sub_nnp)+'_'+previous_iteration_zfill+'.pb').resolve()
                    subprocess.call(['ln','-s', str(nnp_apath), local_path+'/'])
                del it_sub_nnp,nnp_apath
            models_list=" ".join(models_list)

            if exploration_type == 'lammps':
                lammps_input = cf.read_file(training_iterative_apath+'/inputs/'+it_subsys_nr+'.in')
                #lammps_input = cf.read_file(deepmd_iterative_apath+'/inputs/exploration/'+it_subsys_nr+'.in')
                RAND = random.randrange(0,1000)
                lammps_input = cf.replace_in_list(lammps_input,'_SEED_VEL_',str(it_nnp)+str(RAND)+str(it_number)+previous_iteration_zfill)
                RAND = random.randrange(0,1000)
                lammps_input = cf.replace_in_list(lammps_input,'_SEED_THER_',str(it_nnp)+str(RAND)+str(it_number)+previous_iteration_zfill)
                lammps_input = cf.replace_in_list(lammps_input,'_DCD_OUT_',str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+'.dcd')
                lammps_input = cf.replace_in_list(lammps_input,'_RESTART_OUT_',str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+'.restart')
                lammps_input = cf.replace_in_list(lammps_input,'_MODELS_LIST_',models_list)
                lammps_input = cf.replace_in_list(lammps_input,'_DEVI_OUT_','model_devi_'+str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+'.out')
                lammps_input = cf.replace_in_list(lammps_input,'_TEMPERATURE_',str(temperature))
                lammps_input = cf.replace_in_list(lammps_input,'_TIMESTEP_',str(timestep))

            #### Get DATA files and number of steps
                if current_iteration == 1:
                    lammps_data = cf.read_file(training_iterative_apath+'/inputs/'+it_subsys_nr+'.lmp')
                    lammps_input = cf.replace_in_list(lammps_input,'_DATA_FILE_',it_subsys_nr+'.lmp')
                    nb_steps = nb_steps_exploration_initial
                    lammps_input = cf.replace_in_list(lammps_input,'_NUMBER_OF_STEPS_',str(nb_steps))
                    data_file_name = it_subsys_nr+'.lmp'
                # elif ( prevexploration_json['percent_of_bad'] + prevexploration_json['percent_of_new'] ) > 0.75:
                #     lammps_input = cf.replace_in_list(lammps_input,'_DATA_FILE_',it_subsys_nr+'.lmp')
                #     nb_steps = nb_steps_exploration_initial
                #     lammps_input = cf.replace_in_list(lammps_input,'_NUMBER_OF_STEPS_',str(nb_steps))
                else:
                    if len(starting_point_list) == 0:
                        starting_point_list = starting_point_list_bckp
                    RAND = random.randrange(0,len(starting_point_list))
                    previous_data_file = starting_point_list[RAND]
                    del starting_point_list[RAND]
                    lammps_data = cf.read_file(training_iterative_apath+'/starting_structures/'+previous_data_file)
                    data_file_name = previous_data_file
                    ratio_ill_described = (prevexploration_json['subsys_nr'][it_subsys_nr]['nb_candidates'] + prevexploration_json['subsys_nr'][it_subsys_nr]['nb_rejected']) / prevexploration_json['subsys_nr'][it_subsys_nr]['nb_total']

                    if ( ratio_ill_described ) < 0.10:
                        lammps_input = cf.replace_in_list(lammps_input,'_DATA_FILE_',previous_data_file)
                        nb_steps = nb_steps_exploration[it0_subsys_nr] if 'nb_steps_exploration' in globals() else prevexploration_json['subsys_nr'][it_subsys_nr]['nb_steps']
                        nb_steps = nb_steps * 4
                        if nb_steps > 400/timestep:
                            nb_steps = int(400/timestep)
                        lammps_input = cf.replace_in_list(lammps_input,'_NUMBER_OF_STEPS_',str(nb_steps))
                    elif ( ratio_ill_described ) < 0.20:
                        lammps_input = cf.replace_in_list(lammps_input,'_DATA_FILE_',previous_data_file)
                        nb_steps = nb_steps_exploration[it0_subsys_nr] if 'nb_steps_exploration' in globals() else prevexploration_json['subsys_nr'][it_subsys_nr]['nb_steps']
                        nb_steps = nb_steps * 2
                        if nb_steps > 400/timestep:
                            nb_steps = int(400/timestep)
                        lammps_input = cf.replace_in_list(lammps_input,'_NUMBER_OF_STEPS_',str(nb_steps))
                    else:
                        lammps_input = cf.replace_in_list(lammps_input,'_DATA_FILE_',previous_data_file)
                        nb_steps = nb_steps_exploration[it0_subsys_nr] if 'nb_steps_exploration' in globals() else prevexploration_json['subsys_nr'][it_subsys_nr]['nb_steps']
                        if nb_steps > 400/timestep:
                            nb_steps = int(400/timestep)
                        lammps_input = cf.replace_in_list(lammps_input,'_NUMBER_OF_STEPS_',str(nb_steps))
                exploration_json['subsys_nr'][it_subsys_nr]['nb_steps'] = nb_steps

                #### Write DATA file
                cf.write_file(local_path+'/'+data_file_name,lammps_data)

                #### Get print freq
                print_freq_local = print_freq[it0_subsys_nr] if 'print_freq' in globals() else int(nb_steps*0.01)
                lammps_input = cf.replace_in_list(lammps_input,'_PRINT_FREQ_',str(print_freq_local))
                exploration_json['subsys_nr'][it_subsys_nr]['print_freq'] = print_freq_local

                if any('plumed' in f for f in lammps_input):
                    list_plumed_files=[x for x in Path( training_iterative_apath+'/inputs/').glob('*plumed*_'+it_subsys_nr+'.dat')]
                    if len(list_plumed_files) == 0 :
                        logging.critical('Plumed in LAMMPS input but no plumed files')
                        logging.critical('Aborting...')
                        sys.exit(1)
                    plumed_input={}
                    for it_list_plumed_files in list_plumed_files:
                        plumed_input[it_list_plumed_files.name] = cf.read_file(str(it_list_plumed_files))

                    lammps_input = cf.replace_in_list(lammps_input,'_PLUMED_IN_','plumed_'+str(it_subsys_nr)+'.dat')
                    lammps_input = cf.replace_in_list(lammps_input,'_PLUMED_OUT_','plumed_'+str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+'.log')
                    for it_plumed_input in plumed_input:
                        plumed_input[it_plumed_input] = cf.replace_in_list(plumed_input[it_plumed_input],'_PRINT_FREQ_',str(print_freq_local))
                        cf.write_file(local_path+'/'+it_plumed_input,plumed_input[it_plumed_input])

                cf.write_file(local_path+'/'+str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+'.in',lammps_input)
                if (it_number == 1) and (it_nnp == 1):
                    dim_string = ['xlo xhi', 'ylo yhi', 'zlo zhi']
                    cell=[]
                    for n,string in enumerate(dim_string):
                        temp = [zzz for zzz in lammps_data if string in zzz]
                        temp = cf.replace_in_list(temp,'\n','')
                        temp = [zzz for zzz in temp[0].split(' ') if zzz]
                        cell.append(float(temp[1]) - float(temp[0]))
                    temp = [zzz for zzz in lammps_data if 'atoms' in zzz]
                    temp = cf.replace_in_list(temp,'\n','')
                    temp = [zzz for zzz in temp[0].split(' ') if zzz]
                    nb_atm = int(temp[0])
                    del temp,dim_string

                if current_iteration == 1:
                    approx_time = 10
                else:
                    approx_time = ( prevexploration_json['subsys_nr'][it_subsys_nr]['avg_seconds_per_step'] * nb_steps ) / 3600
                    approx_time = approx_time * 1.25
                    approx_time = int(np.ceil(approx_time))

                slurm_file = slurm_file_master
                slurm_file = cf.replace_in_list(slurm_file,'_PROJECT_',project_name)
                slurm_file = cf.replace_in_list(slurm_file,'_WALLTIME_',str(approx_time)+':00:00')
                slurm_file = cf.replace_in_list(slurm_file,'SET_DEEPMD_MODEL_VERSION',str(exploration_json['deepmd_model_version']))
                if allocation_name == 'v100':
                    slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_',allocation_name)
                    if approx_time <= 20:
                        if arch_name == 'v100':
                            slurm_file = cf.replace_in_list(slurm_file,'_QOS_','qos_gpu-t3')
                            slurm_file = cf.replace_in_list(slurm_file,'_PARTITION_','gpu_p13')
                            slurm_file = cf.replace_in_list(slurm_file,'#SBATCH -C _SUBPARTITION_','##SBATCH -C _SUBPARTITION_')
                        elif arch_name == 'a100':
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
                    slurm_file = cf.replace_in_list(slurm_file,'_SUBPARTITION_',arch_name)
                else:
                    logging.critical('Unknown error. Please BUG REPORT')
                    logging.critical('Aborting...')
                    sys.exit(1)
                if slurm_email != '':
                    slurm_file = cf.replace_in_list(slurm_file,'##SBATCH --mail-type','#SBATCH --mail-type')
                    slurm_file = cf.replace_in_list(slurm_file,'##SBATCH --mail-user _EMAIL_','#SBATCH --mail-user '+slurm_email)

                slurm_file = cf.replace_in_list(slurm_file,"\"_INPUT_\"","\""+str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+"\"")
                slurm_file = cf.replace_in_list(slurm_file,"_DATA_FILE_",data_file_name)
                if any('plumed' in f for f in lammps_input):
                    for n,it_plumed_input in enumerate(plumed_input):
                        if n == 0:
                            slurm_file = cf.replace_in_list(slurm_file,"_PLUMED_FILES_LIST_",it_plumed_input)
                        else:
                            slurm_file = cf.replace_in_list(slurm_file,prev_plumed,prev_plumed+"\" \""+it_plumed_input)
                        prev_plumed = it_plumed_input

                else:
                    slurm_file = cf.replace_in_list(slurm_file,"\"_PLUMED_FILES_LIST_\"","")

                models_list_job = models_list.replace(' ','\" \"')
                slurm_file = cf.replace_in_list(slurm_file, '_MODELS_LIST_', models_list_job)

                cf.write_file(local_path+'/job_deepmd_'+exploration_type+'_'+arch_type+'_'+cluster+'.sh',slurm_file)

    config_json['subsys_nr'][it_subsys_nr]['cell'] = cell
    config_json['subsys_nr'][it_subsys_nr]['nb_atm'] = nb_atm
    config_json['subsys_nr'][it_subsys_nr]['exploration_type'] = exploration_type
    del cell,nb_atm
    cf.json_dump(config_json,config_json_fpath, True, 'configuration file')
    exploration_json['is_locked'] = True
    exploration_json['is_launched'] = False
    exploration_json['is_checked'] = False
    exploration_json['is_deviated'] = False
    exploration_json['is_extracted'] = False

    cf.json_dump(exploration_json,exploration_json_fpath, True, 'exploration file')

del it_list_plumed_files,it0_subsys_nr
del slurm_file_master, slurm_file, lammps_data, lammps_input
del prevtraining_json, prevtraining_json_fpath, previous_iteration, previous_iteration_zfill
del config_json, config_json_fpath

if 'temperature_K' in globals():
    del temperature_K
if 'timestep_ps' in globals():
    del timestep_ps
if 'print_freq' in globals():
    del print_freq

del exploration_json, exploration_json_fpath
del deepmd_iterative_apath, nb_traj, exploration_type

del project_name, allocation_name, arch_name, slurm_email, temperature, timestep, nb_steps_exploration_initial, arch_type, training_iterative_apath, current_iteration, current_iteration_zfill, deepmd_model_version, deepmd_model_type_descriptor
del cluster, it_subsys_nr, it_nnp, it_number, local_path, reorder_nnp_list, models_list, RAND, nb_steps, data_file_name, print_freq_local, n, string, approx_time, models_list_job, list_plumed_files, plumed_input, it_plumed_input, prev_plumed

del sys, Path, subprocess, np, random, logging, cf
import gc; gc.collect(); del gc
exit()