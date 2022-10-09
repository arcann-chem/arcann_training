####### Project name / AllocType/GPUType -- > v100/v100 or v100/a100 or a100/a100
project_name='nvs'
allocation_name='cpu'
arch_name='cpu'

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

exploration_json_fpath = training_iterative_apath+'/control/exploration_'+current_iteration_zfill+'.json'
exploration_json = cf.json_read(exploration_json_fpath, abort = True)
if exploration_json['is_extracted'] is False:
    logging.critical('Lock found. Run/Check first: exploration5_extract.py')
    logging.critical('Aborting...')
    sys.exit(1)
del exploration_json_fpath, exploration_json

labeling_json_fpath = training_iterative_apath+'/control/labeling_'+current_iteration_zfill+'.json'
labeling_json = cf.json_read(labeling_json_fpath, abort = False)

if arch_name == 'cpu':
    arch_type='cpu'

cluster = cf.check_cluster()
labeling_json['cluster'] = cluster
labeling_json['project_name'] = project_name if 'project_name' in globals() else 'nvs'
labeling_json['allocation_name'] = allocation_name if 'allocation_name' in globals() else 'cpu'
labeling_json['arch_name'] = arch_name if 'arch_name' in globals() else 'cpu'

slurm_email = '' if 'slurm_email' not in globals() else slurm_email

labeling_json['subsys_nr'] = {}

for it0_subsys_nr,it_subsys_nr in enumerate(config_json['subsys_nr']):
    cf.create_dir(it_subsys_nr)
    cf.change_dir('./'+str(it_subsys_nr))
    labeling_json['subsys_nr'][it_subsys_nr] = {}

    slurm_file_array_master = cf.read_file(deepmd_iterative_path+'/jobs/labeling/job_labeling_array_'+arch_type+'_'+cluster+'.sh')
    slurm_file_master = cf.read_file(deepmd_iterative_path+'/jobs/labeling/job_labeling_XXXXX_'+arch_type+'_'+cluster+'.sh')

    xyz_file=training_iterative_apath+'/'+current_iteration_zfill+'-exploration/'+it_subsys_nr+'/candidates_'+str(it_subsys_nr)+'_'+current_iteration_zfill+'.xyz'
    if Path(training_iterative_apath+'/'+current_iteration_zfill+'-exploration/'+it_subsys_nr+'/candidates_'+str(it_subsys_nr)+'_'+current_iteration_zfill+'_disturbed.xyz').is_file():
        xyz_file_disturbed = training_iterative_apath+'/'+current_iteration_zfill+'-exploration/'+it_subsys_nr+'/candidates_'+str(it_subsys_nr)+'_'+current_iteration_zfill+'_disturbed.xyz'

    cp2k_input_file_1 = cf.read_file(training_iterative_apath+'/inputs/1_'+str(it_subsys_nr)+'_labeling_XXXXX_'+cluster+'.inp')
    cp2k_input_file_2 = cf.read_file(training_iterative_apath+'/inputs/2_'+str(it_subsys_nr)+'_labeling_XXXXX_'+cluster+'.inp')

    cp2k_input_file_1 = cf.replace_in_list(cp2k_input_file_1,'_CELL_',' '.join([str(v) for v in config_json['subsys_nr'][it_subsys_nr]['cell']]))
    cp2k_input_file_2 = cf.replace_in_list(cp2k_input_file_2,'_CELL_',' '.join([str(v) for v in config_json['subsys_nr'][it_subsys_nr]['cell']]))

    cp2k_input_file_1 = cf.replace_in_list(cp2k_input_file_1,'_WALLTIME_','00:30:00')
    cp2k_input_file_2 = cf.replace_in_list(cp2k_input_file_2,'_WALLTIME_','01:00:00')

    n_atom, step_atoms, step_coordinates, blank = cf.import_xyz(xyz_file)

    for step_iter in range(1,step_atoms.shape[0]+1,1):
        step_iter_str = str(step_iter).zfill(5)
        cf.create_dir(step_iter_str)

        cp2k_input_file_t1 = cf.replace_in_list(cp2k_input_file_1,'XXXXX',step_iter_str)
        cp2k_input_file_t2 = cf.replace_in_list(cp2k_input_file_2,'XXXXX',step_iter_str)

        cf.write_file('./'+step_iter_str+'/1_labeling_'+step_iter_str+'.inp',cp2k_input_file_t1)
        cf.write_file('./'+step_iter_str+'/2_labeling_'+step_iter_str+'.inp',cp2k_input_file_t2)

        slurm_file = cf.replace_in_list(slurm_file_master,'XXXXX',step_iter_str)
        slurm_file = cf.replace_in_list(slurm_file,'_PROJECT_',project_name)
        slurm_file = cf.replace_in_list(slurm_file,'_ALLOC_',allocation_name)
        slurm_file = cf.replace_in_list(slurm_file,'_WALLTIME_','02:00:00')
        slurm_file = cf.replace_in_list(slurm_file,'_CP2K_JOBNAME_','CP2K_'+it_subsys_nr+'_'+current_iteration_zfill)
        cf.write_file('./'+step_iter_str+'/job_labeling_'+step_iter_str+'_'+arch_type+'_'+cluster+'.sh',slurm_file)

        cf.write_xyz_from_index(step_iter-1,'./'+step_iter_str+'/labeling_'+step_iter_str+'.xyz',n_atom,step_atoms,step_coordinates,blank)
        end_step = step_iter

    del n_atom, step_atoms, step_coordinates, blank, step_iter, step_iter_str
    labeling_json['subsys_nr'][it_subsys_nr]['standard'] = end_step

    if Path(training_iterative_apath+'/'+current_iteration_zfill+'-exploration/'+it_subsys_nr+'/candidates_'+str(it_subsys_nr)+'_'+current_iteration_zfill+'_disturbed.xyz').is_file():
        n_atom, step_atoms, step_coordinates, blank = cf.import_xyz(xyz_file_disturbed)
        for d_step_iter in range(end_step+1,step_atoms.shape[0]+end_step+1,1):
            d_step_iter_str = str(d_step_iter).zfill(5)

            cf.create_dir(d_step_iter_str)

            cp2k_input_file_t1 = cf.replace_in_list(cp2k_input_file_1,'XXXXX',d_step_iter_str)
            cp2k_input_file_t2 = cf.replace_in_list(cp2k_input_file_2,'XXXXX',d_step_iter_str)

            cf.write_file('./'+d_step_iter_str+'/1_labeling_'+d_step_iter_str+'.inp',cp2k_input_file_t1)
            cf.write_file('./'+d_step_iter_str+'/2_labeling_'+d_step_iter_str+'.inp',cp2k_input_file_t2)

            slurm_file=cf.replace_in_list(slurm_file_master,'XXXXX',d_step_iter_str)
            slurm_file=cf.replace_in_list(slurm_file,'_WALLTIME_','02:00:00')
            slurm_file=cf.replace_in_list(slurm_file,'_CP2K_JOBNAME_','CP2K_'+it_subsys_nr+'_'+current_iteration_zfill)
            cf.write_file('./'+d_step_iter_str+'/job_labeling_'+d_step_iter_str+'_'+arch_type+'_'+cluster+'.sh',slurm_file)

            cf.write_xyz_from_index(d_step_iter-end_step-1,'./'+d_step_iter_str+'/labeling_'+d_step_iter_str+'.xyz',n_atom,step_atoms,step_coordinates,blank)

            end_step_disturbed=d_step_iter

        labeling_json['subsys_nr'][it_subsys_nr]['disturbed'] = d_step_iter-end_step
        del n_atom, step_atoms, step_coordinates, blank, d_step_iter, d_step_iter_str, end_step, end_step_disturbed
    else:
        labeling_json['subsys_nr'][it_subsys_nr]['disturbed'] = 0

    slurm_file_array=cf.replace_in_list(slurm_file_array_master,'_WALLTIME_','02:00:00')
    slurm_file_array = cf.replace_in_list(slurm_file_array,'_PROJECT_',project_name)
    slurm_file_array = cf.replace_in_list(slurm_file_array,'_ALLOC_',allocation_name)
    slurm_file_array=cf.replace_in_list(slurm_file_array,'_CP2K_JOBNAME_','CP2K_'+it_subsys_nr+'_'+current_iteration_zfill)
    slurm_file_array=cf.replace_in_list(slurm_file_array,'_ARRAYCOUNT_',str(int(labeling_json['subsys_nr'][it_subsys_nr]['standard']+labeling_json['subsys_nr'][it_subsys_nr]['disturbed'] )))

    cf.write_file('./job_labeling_array_'+arch_type+'_'+cluster+'.sh',slurm_file_array)
    labeling_json['is_locked'] = True
    labeling_json['is_launched'] = False
    labeling_json['is_checked'] = False
    labeling_json['is_extracted'] = False

    cf.json_dump(labeling_json,labeling_json_fpath, True, 'labeling file')

    cf.change_dir('../')

del it0_subsys_nr, it_subsys_nr
del xyz_file, xyz_file_disturbed
del project_name, allocation_name, arch_name, training_iterative_apath, current_iteration, current_iteration_zfill, cluster, slurm_email
del config_json, config_json_fpath
del slurm_file_master, slurm_file_array_master, slurm_file, slurm_file_array
del cp2k_input_file_1, cp2k_input_file_2, cp2k_input_file_t1, cp2k_input_file_t2
del labeling_json, labeling_json_fpath, arch_type
del deepmd_iterative_path

del sys, Path, subprocess, logging, cf
import gc; gc.collect(); del gc
exit()