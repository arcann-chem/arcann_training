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
del deepmd_iterative_path

### Read what is needed
config_json_fpath = training_iterative_apath+'/control/config.json'
config_json = cf.json_read(config_json_fpath, abort=True)

config_json['current_iteration'] = current_iteration if 'current_iteration' in globals() else cf.check_if_in_dict(config_json,'current_iteration',False,0)
current_iteration = config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

labeling_json_fpath = training_iterative_apath+'/control/labeling_'+current_iteration_zfill+'.json'
labeling_json = cf.json_read(labeling_json_fpath, abort=True)
#del current_iteration_zfill, training_iterative_apath

#del current_iteration, training_iterative_apath, config_json, config_json_fpath

if labeling_json['is_launched'] is False:
    logging.critical('Lock found. Launch first (labeling2.py)')
    logging.critical('Aborting...')
    sys.exit(1)

count = 0
step_1 = 0
step_2 = 0
for it_subsys_nr in labeling_json['subsys_nr']:
    count = count + labeling_json['subsys_nr'][it_subsys_nr]['standard'] + labeling_json['subsys_nr'][it_subsys_nr]['disturbed']
    average_per_step = 0
    timings_sum_1 = 0
    timings_1 = []
    timings_sum_2 = 0
    timings_2 = []

    for it_step in range(1, labeling_json['subsys_nr'][it_subsys_nr]['standard'] + 1):
        it_step_zfill = str(it_step).zfill(5)
        check_path='./'+str(it_subsys_nr)+'/'+it_step_zfill
        cp2k_output_file_1 = check_path+'/1_labeling_'+it_step_zfill+'.out'
        cf.check_file(cp2k_output_file_1,0,1)
        if Path(cp2k_output_file_1).is_file():
            cp2k_output_1 = cf.read_file(cp2k_output_file_1)
            if any('SCF run converged in ' in f for f in cp2k_output_1):
                step_1 = step_1 + 1
                timings_1 = [zzz for zzz in cp2k_output_1 if 'CP2K                                 1  1.0' in zzz]
                timings_sum_1 = timings_sum_1 + float(timings_1[0].split(' ')[-1])

            else:
                logging.warning('1_'+str(it_subsys_nr)+'_'+it_step_zfill+' not converged/failed. Check manually')
        cp2k_output_file_2 = check_path+'/2_labeling_'+it_step_zfill+'.out'
        cf.check_file(cp2k_output_file_2,0,1)
        if Path(cp2k_output_file_2).is_file():
            cp2k_output_2 = cf.read_file(cp2k_output_file_2)
            if any('SCF run converged in ' in f for f in cp2k_output_2):
                step_2 = step_2 + 1
                timings_2 = [zzz for zzz in cp2k_output_2 if 'CP2K                                 1  1.0' in zzz]
                timings_sum_2 = timings_sum_2 + float(timings_2[0].split(' ')[-1])
            else:
                logging.critical('2_'+str(it_subsys_nr)+'_'+it_step_zfill+' not converged/failed. Check manually')

    for it_step in range(labeling_json['subsys_nr'][it_subsys_nr]['standard'] + 1, labeling_json['subsys_nr'][it_subsys_nr]['standard'] + labeling_json['subsys_nr'][it_subsys_nr]['disturbed'] + 1):
        it_step_zfill = str(it_step).zfill(5)
        check_path='./'+str(it_subsys_nr)+'/'+it_step_zfill
        cp2k_output_file_1 = check_path+'/1_labeling_'+it_step_zfill+'.out'
        cf.check_file(cp2k_output_file_1,0,1)
        if Path(cp2k_output_file_1).is_file():
            cp2k_output_1 = cf.read_file(cp2k_output_file_1)
            if any('SCF run converged in ' in f for f in cp2k_output_1):
                step_1 = step_1 + 1
                timings_1 = [zzz for zzz in cp2k_output_1 if 'CP2K                                 1  1.0' in zzz]
                timings_sum_1 = timings_sum_1 + float(timings_1[0].split(' ')[-1])
            else:
                logging.warning('1_'+str(it_subsys_nr)+'_'+it_step_zfill+' not converged/failed. Check manually')
        cp2k_output_file_2 = check_path+'/2_labeling_'+it_step_zfill+'.out'
        cf.check_file(cp2k_output_file_2,0,1)
        if Path(cp2k_output_file_2).is_file():
            cp2k_output_2 = cf.read_file(cp2k_output_file_2)
            if any('SCF run converged in ' in f for f in cp2k_output_2):
                step_2 = step_2 + 1
                timings_2 = [zzz for zzz in cp2k_output_2 if 'CP2K                                 1  1.0' in zzz]
                timings_sum_2 = timings_sum_2 + float(timings_2[0].split(' ')[-1])
            else:
                logging.critical('2_'+str(it_subsys_nr)+'_'+it_step_zfill+' not converged/failed. Check manually')

    timings_1 = timings_sum_1/step_1
    timings_2 = timings_sum_2/step_2
    labeling_json['subsys_nr'][it_subsys_nr]['timing'] = [ timings_1, timings_2 ]

if count != step_1:
    logging.warning('Some jobs have failed/not converged/still running (first step). Check manually.')

if count != step_2:
    logging.critical('Some jobs have failed/not converged/still running (second step). Check manually.')
    logging.critical('Aborting...')
    sys.exit(1)
else:
     labeling_json['is_checked'] = True
     cf.json_dump(labeling_json,labeling_json_fpath, True, 'labeling file')
     for it_subsys_nr in labeling_json['subsys_nr']:
        cf.remove_file_glob('./'+it_subsys_nr+'/','CP2K.*')
        for it_step in range(1, labeling_json['subsys_nr'][it_subsys_nr]['standard'] + labeling_json['subsys_nr'][it_subsys_nr]['disturbed'] + 1):
            it_step_zfill = str(it_step).zfill(5)
            cf.remove_file_glob('./'+it_subsys_nr+'/'+it_step_zfill+'/','CP2K.*')

logging.info('Labeling success')

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()