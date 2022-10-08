####### Override / Read from config.json (local subsys)
atomsk_fpath='/gpfswork/rech/nvs/commun/programs/apps/atomsk/0.11.2/bin/atomsk'
#vmd_fpath=''
disturb=[0.1,0.1]
disturb_devi=[0.1,0.1]
### WOLOLO

###################################### No change past here
import sys
import os
from pathlib import Path
import subprocess
import numpy as np
import logging

logging.basicConfig(level = logging.INFO,format='%(levelname)s: %(message)s')

if 'atomsk_fpath' not in globals():
    atomsk = subprocess.call(['command','-v','atomsk'])
    if atomsk == 1:
        logging.critical('atmsk not found.')
        logging.critical('Aborting...')
        sys.exit(1)
    else:
        atomsk_bin = 'atomsk'
else:
    atomsk = subprocess.call(['command','-v',atomsk_fpath])
    if atomsk == 1:
        logging.critical('Your path seems shifty: '+ atomsk_fpath)
        logging.critical('Aborting...')
        sys.exit(1)
    else:
        atomsk_bin = atomsk_fpath
del atomsk

if 'vmd_fpath' not in globals():
    vmd = subprocess.call(['command', '-v', 'vmd'])
    if vmd == 1:
        logging.critical('vmd not found.')
        logging.critical('Aborting...')
        sys.exit(1)
    else:
        vmd_bin = 'vmd'
else:
    vmd = subprocess.call(['command', '-v', vmd_fpath])
    if vmd == 1:
        logging.critical('Your path seems shifty: '+ vmd_fpath)
        logging.critical('Aborting...')
        sys.exit(1)
    else:
        vmd_bin = vmd_fpath
del vmd

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
config_json = cf.json_read(config_json_fpath, abort = True)

config_json['current_iteration'] = current_iteration if 'current_iteration' in globals() else cf.check_if_in_dict(config_json,'current_iteration',False,0)
current_iteration = config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

del current_iteration

exploration_json_fpath = training_iterative_apath+'/control/exploration_'+current_iteration_zfill+'.json'
exploration_json = cf.json_read(exploration_json_fpath, abort = True)

cf.check_file(deepmd_iterative_path+'/scripts/vmd_dcd_selection_index.tcl',0,0)
master_vmd_tcl = cf.read_file(deepmd_iterative_path+'/scripts/vmd_dcd_selection_index.tcl')

if exploration_json['is_deviated'] is False:
    logging.critical('Lock found. Check Normal termination (ex4.py)')
    logging.critical('Aborting...')
    sys.exit(1)

cf.create_dir(training_iterative_apath+'/starting_structures')

for it0_subsys_nr,it_subsys_nr in enumerate(config_json['subsys_nr']):
    cf.change_dir('./'+str(it_subsys_nr))
    print_freq = exploration_json['subsys_nr'][it_subsys_nr]['print_freq']
    if 'lammps' in exploration_json['subsys_nr'][it_subsys_nr]['exploration_type']:
        cf.check_file(training_iterative_apath+'/inputs/'+it_subsys_nr+'.lmp',0,0)
        subprocess.call([atomsk_bin,training_iterative_apath+'/inputs/'+it_subsys_nr+'.lmp','pdb',training_iterative_apath+'/inputs/'+it_subsys_nr,'-ow'])
        topo_file=training_iterative_apath+'/inputs/'+it_subsys_nr+'.pdb'

        for it_nnp in range(1,  exploration_json['nb_nnp'] + 1):
            cf.change_dir('./'+str(it_nnp))

            for it_each in range(1, exploration_json['nb_traj']+1):
                cf.change_dir('./'+str(it_each).zfill(5))

                devi_json = cf.json_read('./selection_candidates.json', abort = False)
                devi_json_index = cf.json_read('./selection_candidates_index.json', abort = False)

                ### Selection of the min for the next iteration starting point
                min_index = devi_json['min_index']
                min_index = int(min_index / print_freq)
                with open('min.vmd', 'w') as f:
                    f.write(str(min_index))
                f.close()
                del f
                vmd_tcl = master_vmd_tcl
                vmd_tcl = cf.replace_in_list(vmd_tcl,'_TOPO_FILE_',topo_file)
                if 'lammps' in exploration_json['subsys_nr'][it_subsys_nr]['exploration_type']:
                    traj_file = str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+'.dcd'

                vmd_tcl = cf.replace_in_list(vmd_tcl,'_DCD_FILE_',traj_file)
                vmd_tcl = cf.replace_in_list(vmd_tcl,'_SELECTION_FILE_','min.vmd')
                cf.write_file('vmd.tcl',vmd_tcl)
                subprocess.call([vmd_bin,'-e','vmd.tcl','-dispdev', 'text'])
                cf.remove_file('vmd.tcl')
                cf.remove_file('min.vmd')
                del vmd_tcl, traj_file, min_index

                if Path('vmd_00000.xyz').is_file():
                    min_file_name=current_iteration_zfill+'_'+str(it_subsys_nr)+'_'+str(it_nnp)+'_'+str(it_each).zfill(5)
                    Path('vmd_00000.xyz').rename(min_file_name+'.xyz')
                    if 'disturb' in globals() and disturb[it0_subsys_nr] != 0:
                        subprocess.call([atomsk_bin, '-ow', min_file_name+'.xyz',\
                            '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][0]), 'H1',\
                            '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][1]), 'H2',\
                            '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][2]), 'H3',\
                            '-disturb', str(disturb[it0_subsys_nr]),\
                            'xyz'])
                    
                        # subprocess.call([atomsk_bin, '-ow', min_file_name+'.xyz',\
                        #     '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][0]), 'H1',\
                        #     '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][1]), 'H2',\
                        #     '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][2]), 'H3',\
                        #     '-disturb', str(disturb[it0_subsys_nr]),\
                        #     'lmp'])
                    # else:
                    subprocess.call([atomsk_bin, '-ow', min_file_name+'.xyz',\
                        '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][0]), 'H1',\
                        '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][1]), 'H2',\
                        '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][2]), 'H3',\
                        'lmp'])
                    Path(min_file_name+'.xyz').rename(training_iterative_apath+'/starting_structures/'+min_file_name+'.xyz')
                    Path(min_file_name+'.lmp').rename(training_iterative_apath+'/starting_structures/'+min_file_name+'.lmp')
                    del min_file_name
                else:
                    logging.warning('Problem on preparing the min for: '+str(it_subsys_nr)+' / '+str(it_nnp)+' / '+str(it_each) )\

                ### Selection of labeling XYZ
                if len(devi_json_index['candidates_kept_ind']) != 0:
                    candidates_index = np.array(devi_json_index['candidates_kept_ind'])
                    candidates_index = candidates_index / print_freq
                    candidates_index = candidates_index.astype(int)
                    candidates_index = map(str, candidates_index.astype(int))
                    candidates_index = [ zzz + '\n' for zzz in candidates_index]
                    cf.write_file('label.vmd', candidates_index)
                    vmd_tcl = master_vmd_tcl
                    vmd_tcl = cf.replace_in_list(vmd_tcl,'_TOPO_FILE_',topo_file)
                    if 'lammps' in exploration_json['subsys_nr'][it_subsys_nr]['exploration_type']:
                        traj_file=str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+'.dcd'
                    vmd_tcl = cf.replace_in_list(vmd_tcl,'_DCD_FILE_',traj_file)
                    vmd_tcl = cf.replace_in_list(vmd_tcl,'_SELECTION_FILE_','label.vmd')
                    cf.write_file('vmd.tcl',vmd_tcl)
                    subprocess.call([vmd_bin,'-e','vmd.tcl','-dispdev', 'text'])
                    cf.remove_file('vmd.tcl')
                    cf.remove_file('label.vmd')
                    del candidates_index, vmd_tcl, traj_file

                   #TODO Replace with either subprocess call or read python
                   
                    cf.remove_file('./candidates_'+str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+'.xyz')
                    os.system('cat vmd_*.xyz >> temp_candidates_'+str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+'.xyz')

                    if 'disturb_devi' in globals() and disturb_devi[it0_subsys_nr] != 0:
                        vmd_xyz_files=[zzz for zzz in Path('.').glob('vmd_*')]
                        for it_vmd_xyz_files in vmd_xyz_files:
                            subprocess.call([atomsk_bin, '-ow', str(it_vmd_xyz_files),\
                                '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][0]), 'H1',\
                                '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][1]), 'H2',\
                                '-cell', 'set', str(config_json['subsys_nr'][it_subsys_nr]['cell'][2]), 'H3',\
                                'xyz', str(it_vmd_xyz_files)+'_disturbed'])
                        del it_vmd_xyz_files, vmd_xyz_files
                        os.system('cat vmd_*_disturbed.xyz >> temp_candidates_'+str(it_subsys_nr)+'_'+str(it_nnp)+'_'+current_iteration_zfill+'_disturbed.xyz')

                    cf.remove_file_glob('.','vmd_*.xyz')
                del devi_json, devi_json_index

                cf.change_dir('..')

            del it_each
            cf.change_dir('..')

        #TODO Replace with either subprocess call or read python

        if 'disturb_devi' in globals() and disturb_devi[it0_subsys_nr] != 0:
            cf.remove_file('candidates_'+str(it_subsys_nr)+'_'+current_iteration_zfill+'_disturbed.xyz')
            os.system('cat ./*/*/temp_candidates_*_disturbed.xyz >> candidates_'+str(it_subsys_nr)+'_'+current_iteration_zfill+'_disturbed.xyz')
            os.system('rm -rf ./*/*/temp_candidates_*_disturbed.xyz')
            exploration_json['subsys_nr'][it_subsys_nr]['disturbed'] = True
        else:
            exploration_json['subsys_nr'][it_subsys_nr]['disturbed'] = False

        cf.remove_file('candidates_'+str(it_subsys_nr)+'_'+current_iteration_zfill+'.xyz')
        os.system('cat ./*/*/temp_candidates_*.xyz >> candidates_'+str(it_subsys_nr)+'_'+current_iteration_zfill+'.xyz')
        os.system('rm -rf ./*/*/temp_candidates*.xyz')
        del it_nnp
        cf.change_dir('..')

exploration_json['is_extracted'] = True
cf.json_dump(exploration_json,exploration_json_fpath, True, 'exploration file')

del it0_subsys_nr, it_subsys_nr, topo_file, print_freq
del current_iteration_zfill, master_vmd_tcl, config_json_fpath, config_json, exploration_json, exploration_json_fpath
del atomsk_bin, vmd_bin, deepmd_iterative_path, training_iterative_apath

if 'disturb' in globals():
    del disturb
if 'atomsk_fpath' in globals():
    del atomsk_fpath
if 'vmd_fpath' in globals():
    del vmd_fpath

logging.info('Extraction success')

del sys, os, Path, np, subprocess, logging, cf
import gc; gc.collect(); del gc
exit()