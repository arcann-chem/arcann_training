###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')

training_iterative_apath = str(Path('..').resolve())
### Check if the deepmd_iterative_apath is defined
if 'deepmd_iterative_path' in globals():
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

labeling_json_fpath = training_iterative_apath+'/control/labeling_'+current_iteration_zfill+'.json'
labeling_json = cf.json_read(labeling_json_fpath, abort=True)

### Checks
if labeling_json['is_launched'] is False:
    logging.critical('Lock found. Run/Check first: labeling2_launch.py')
    logging.critical('Aborting...')
    sys.exit(1)

### Check the normal termination of the labeling phase
total_steps = 0
step_1 = 0
step_2 = 0
for it_subsys_nr in labeling_json['subsys_nr']:
    total_steps = total_steps + labeling_json['subsys_nr'][it_subsys_nr]['standard'] + labeling_json['subsys_nr'][it_subsys_nr]['disturbed']
    average_per_step = 0
    timings_sum_1 = 0
    timings_1 = []
    timings_sum_2 = 0
    timings_2 = []
    for it_step in range(1, labeling_json['subsys_nr'][it_subsys_nr]['standard'] + 1):
        it_step_zfill = str(it_step).zfill(5)
        check_path='./'+str(it_subsys_nr)+'/'+it_step_zfill
        cp2k_output_file_1 = check_path+'/1_labeling_'+it_step_zfill+'.out'
        cf.check_file(cp2k_output_file_1,0,False,cp2k_output_file_1+' not present. Check manually.')
        if Path(cp2k_output_file_1).is_file():
            cp2k_output_1 = cf.read_file(cp2k_output_file_1)
            if any('SCF run converged in ' in f for f in cp2k_output_1):
                step_1 = step_1 + 1
                timings_1 = [zzz for zzz in cp2k_output_1 if 'CP2K                                 1  1.0' in zzz]
                timings_sum_1 = timings_sum_1 + float(timings_1[0].split(' ')[-1])
            else:
                logging.warning(cp2k_output_file_1+' not converged/failed. Check manually.')
        cp2k_output_file_2 = check_path+'/2_labeling_'+it_step_zfill+'.out'
        cf.check_file(cp2k_output_file_2,0,False,cp2k_output_file_2+' not present. Check manually.')
        if Path(cp2k_output_file_2).is_file():
            cp2k_output_2 = cf.read_file(cp2k_output_file_2)
            if any('SCF run converged in ' in f for f in cp2k_output_2):
                step_2 = step_2 + 1
                timings_2 = [zzz for zzz in cp2k_output_2 if 'CP2K                                 1  1.0' in zzz]
                timings_sum_2 = timings_sum_2 + float(timings_2[0].split(' ')[-1])
            else:
                logging.critical(cp2k_output_file_2+' not converged/failed. Check manually.')

    for it_step in range(labeling_json['subsys_nr'][it_subsys_nr]['standard'] + 1, labeling_json['subsys_nr'][it_subsys_nr]['standard'] + labeling_json['subsys_nr'][it_subsys_nr]['disturbed'] + 1):
        it_step_zfill = str(it_step).zfill(5)
        check_path='./'+str(it_subsys_nr)+'/'+it_step_zfill
        cp2k_output_file_1 = check_path+'/1_labeling_'+it_step_zfill+'.out'
        cf.check_file(cp2k_output_file_1,0,False,cp2k_output_file_1+' not present. Check manually.')
        if Path(cp2k_output_file_1).is_file():
            cp2k_output_1 = cf.read_file(cp2k_output_file_1)
            if any('SCF run converged in ' in f for f in cp2k_output_1):
                step_1 = step_1 + 1
                timings_1 = [zzz for zzz in cp2k_output_1 if 'CP2K                                 1  1.0' in zzz]
                timings_sum_1 = timings_sum_1 + float(timings_1[0].split(' ')[-1])
            else:
                logging.warning(cp2k_output_file_1+' not converged/failed. Check manually.')
        cp2k_output_file_2 = check_path+'/2_labeling_'+it_step_zfill+'.out'
        cf.check_file(cp2k_output_file_2,0,False,cp2k_output_file_2+' not present. Check manually.')
        if Path(cp2k_output_file_2).is_file():
            cp2k_output_2 = cf.read_file(cp2k_output_file_2)
            if any('SCF run converged in ' in f for f in cp2k_output_2):
                step_2 = step_2 + 1
                timings_2 = [zzz for zzz in cp2k_output_2 if 'CP2K                                 1  1.0' in zzz]
                timings_sum_2 = timings_sum_2 + float(timings_2[0].split(' ')[-1])
            else:
                logging.critical(cp2k_output_file_2+' not converged/failed. Check manually.')
    timings_1 = timings_sum_1/step_1
    timings_2 = timings_sum_2/step_2
    labeling_json['subsys_nr'][it_subsys_nr]['timing_s'] = [timings_1, timings_2]

if total_steps != step_1:
    logging.warning('Some jobs have failed/not converged/still running (first step). Check manually.')

if total_steps != step_2:
    logging.critical('Some jobs have failed/not converged/still running (second step). Check manually.')
    logging.critical('Aborting...')
    sys.exit(1)
else:
     labeling_json['is_checked'] = True
     cf.json_dump(labeling_json,labeling_json_fpath,True,'labeling file')
     for it_subsys_nr in labeling_json['subsys_nr']:
        cf.remove_file_glob('./'+it_subsys_nr+'/','CP2K.*')
        for it_step in range(1, labeling_json['subsys_nr'][it_subsys_nr]['standard'] + labeling_json['subsys_nr'][it_subsys_nr]['disturbed'] + 1):
            it_step_zfill = str(it_step).zfill(5)
            cf.remove_file_glob('./'+it_subsys_nr+'/'+it_step_zfill+'/','CP2K.*')

logging.info('The labeling job phase is a success!')

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del labeling_json, labeling_json_fpath
del deepmd_iterative_apath

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()