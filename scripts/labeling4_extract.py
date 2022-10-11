###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')

import numpy as np

### Constants
Ha_to_eV=np.float64(27.211386245988)
Bohr_to_A=np.float64(0.529177210903)
au_to_eV_per_A=np.float64(Ha_to_eV/Bohr_to_A)
eV_per_A3_to_GPa=np.float64(160.21766208)

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
config_json = cf.json_read(config_json_fpath,True,True)

current_iteration = current_iteration if 'current_iteration' in globals() else config_json['current_iteration']
current_iteration_zfill = str(current_iteration).zfill(3)

labeling_json_fpath = training_iterative_apath+'/control/labeling_'+current_iteration_zfill+'.json'
labeling_json = cf.json_read(labeling_json_fpath,True,True)

### Checks
if labeling_json['is_checked'] is False:
    logging.critical('Lock found. Run/Check first: labeling3_check.py')
    logging.critical('Aborting...')
    sys.exit(1)

### Launch the extractions
cf.create_dir(training_iterative_apath+'/data/')

for it_subsys_nr in labeling_json['subsys_nr']:
    cf.create_dir(training_iterative_apath+'/data/'+it_subsys_nr+'_'+current_iteration_zfill)
    cf.create_dir(training_iterative_apath+'/data/'+it_subsys_nr+'_'+current_iteration_zfill+'/set.000')
    force_array_raw = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['candidates'], config_json['subsys_nr'][it_subsys_nr]['nb_atm'] * 3 ))
    energy_array_raw = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['candidates']))
    coord_array_raw = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['candidates'], config_json['subsys_nr'][it_subsys_nr]['nb_atm'] * 3 ))
    box_array_raw = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['candidates'], 9))
    virial_array_raw = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['candidates'], 9))

    box_array_raw[:,0] = config_json['subsys_nr'][it_subsys_nr]['cell'][0]
    box_array_raw[:,4] = config_json['subsys_nr'][it_subsys_nr]['cell'][1]
    box_array_raw[:,8] = config_json['subsys_nr'][it_subsys_nr]['cell'][2]

    volume = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['candidates']))
    volume = box_array_raw[:,0] * box_array_raw[:,4] * box_array_raw[:,8]

    for it_step in range(1, labeling_json['subsys_nr'][it_subsys_nr]['candidates'] + 1):
        it_step_zfill = str(it_step).zfill(5)
        check_path='./'+str(it_subsys_nr)+'/'+it_step_zfill
        if it_step == 1:
            cf.check_file(training_iterative_apath+'/inputs/'+it_subsys_nr+'.lmp',0,True,'Input data file (lmp file) not present.')
            lammps_data = cf.read_file(training_iterative_apath+'/inputs/'+it_subsys_nr+'.lmp')
            index = [idx for idx, s in enumerate(lammps_data) if 'Atoms' in s][0]
            del lammps_data[0:index+2]
            lammps_data = lammps_data[0:config_json['subsys_nr'][it_subsys_nr]['nb_atm']+1]
            lammps_data = [' '.join(f.replace('\n','').split()) for f in lammps_data]
            lammps_data = [g.split(' ')[1:2] for g in lammps_data]
            type_atom_array = np.asarray(lammps_data,dtype=np.int64).flatten()
            type_atom_array = type_atom_array - 1
            np.savetxt('./'+str(it_subsys_nr)+'/type.raw',type_atom_array,delimiter=' ',newline=' ',fmt='%d')
            np.savetxt(training_iterative_apath+'/data/'+it_subsys_nr+'_'+current_iteration_zfill+'/type.raw',type_atom_array,delimiter=' ',newline=' ',fmt='%d')
            cp2k_out = cf.read_file(check_path+'/2_labeling_'+it_step_zfill+'.out')
            cp2k_out = [zzz for zzz in cp2k_out if 'CP2K| version string:' in zzz]
            cp2k_out = [' '.join(f.replace('\n','').split()) for f in cp2k_out]
            cp2k_out = [g.split(' ')[-1] for g in cp2k_out]
            cp2k_version = float(cp2k_out[0])
            del lammps_data, cp2k_out, type_atom_array, index

        stress_xyz = cf.read_file(check_path+'/2_labeling_'+it_step_zfill+'-Stress_Tensor.st')
        if cp2k_version < 8.1:
            del stress_xyz[0:4]
            stress_xyz = stress_xyz[0:3]
            stress_xyz = [' '.join(f.replace('\n','').split()) for f in stress_xyz]
            stress_xyz = [g.split(' ')[1:4] for g in stress_xyz]
            stress_xyz_array = np.asarray(stress_xyz,dtype=np.float64).flatten()
            virial_array_raw[it_step-1,:] = stress_xyz_array * volume[it_step-1] / eV_per_A3_to_GPa

        force_cp2k = cf.read_file(check_path+'/2_labeling_'+it_step_zfill+'-Forces.for')
        del force_cp2k[0:4]
        del force_cp2k[-1]
        force_cp2k = [' '.join(f.replace('\n','').split()) for f in force_cp2k]
        force_cp2k = [g.split(' ')[3:] for g in force_cp2k]
        force_array = np.asarray(force_cp2k,dtype=np.float64).flatten()
        force_array_raw[it_step-1,:] = force_array*au_to_eV_per_A

        energy_cp2k = cf.read_file(check_path+'/2_labeling_'+it_step_zfill+'-Force_Eval.fe')
        del energy_cp2k[0]
        del energy_cp2k[-1]
        energy_cp2k = [' '.join(f.replace('\n','').split()) for f in energy_cp2k]
        energy_cp2k = [g.split(' ')[-1] for g in energy_cp2k]
        energy_array = np.asarray(energy_cp2k,dtype=np.float64).flatten()
        energy_array_raw[it_step-1] = energy_array*Ha_to_eV

        coord_xyz = cf.read_file(check_path+'/labeling_'+it_step_zfill+'.xyz')
        del coord_xyz[0:2]
        coord_xyz = [' '.join(f.replace('\n','').split()) for f in coord_xyz]
        coord_xyz = [g.split(' ')[1:] for g in coord_xyz]
        coord_array = np.asarray(coord_xyz,dtype=np.float64).flatten()
        coord_array_raw[it_step-1,:] = coord_array

        del coord_xyz, energy_cp2k, force_cp2k, stress_xyz, check_path, it_step_zfill
    del it_step

    np.savetxt('./'+str(it_subsys_nr)+'/box.raw',box_array_raw,delimiter=' ')
    np.save(training_iterative_apath+'/data/'+it_subsys_nr+'_'+current_iteration_zfill+'/set.000/box',box_array_raw)
    np.savetxt('./'+str(it_subsys_nr)+'/virial.raw',virial_array_raw,delimiter=' ')
    np.save(training_iterative_apath+'/data/'+it_subsys_nr+'_'+current_iteration_zfill+'/set.000/virial',virial_array_raw)
    np.savetxt('./'+str(it_subsys_nr)+'/force.raw',force_array_raw,delimiter=' ')
    np.save(training_iterative_apath+'/data/'+it_subsys_nr+'_'+current_iteration_zfill+'/set.000/force',force_array_raw)
    np.savetxt('./'+str(it_subsys_nr)+'/energy.raw',energy_array_raw,delimiter=' ')
    np.save(training_iterative_apath+'/data/'+it_subsys_nr+'_'+current_iteration_zfill+'/set.000/energy',energy_array_raw)
    np.savetxt('./'+str(it_subsys_nr)+'/coord.raw',coord_array_raw,delimiter=' ')
    np.save(training_iterative_apath+'/data/'+it_subsys_nr+'_'+current_iteration_zfill+'/set.000/coord',coord_array_raw)

    del box_array_raw, virial_array_raw, force_array_raw, energy_array_raw, coord_array_raw

    if labeling_json['subsys_nr'][it_subsys_nr]['disturbed_candidates'] != 0 :
        cf.create_dir(training_iterative_apath+'/data/'+it_subsys_nr+'-disturbed_'+current_iteration_zfill)
        cf.create_dir(training_iterative_apath+'/data/'+it_subsys_nr+'-disturbed_'+current_iteration_zfill+'/set.000')
        force_array_raw = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['disturbed_candidates'], config_json['subsys_nr'][it_subsys_nr]['nb_atm'] * 3 ))
        energy_array_raw = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['disturbed_candidates']))
        coord_array_raw = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['disturbed_candidates'], config_json['subsys_nr'][it_subsys_nr]['nb_atm'] * 3 ))
        box_array_raw = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['disturbed_candidates'], 9))
        virial_array_raw = np.zeros((labeling_json['subsys_nr'][it_subsys_nr]['disturbed_candidates'], 9))

        for count,it_step in enumerate(range(labeling_json['subsys_nr'][it_subsys_nr]['candidates'] + 1, labeling_json['subsys_nr'][it_subsys_nr]['candidates'] + labeling_json['subsys_nr'][it_subsys_nr]['disturbed_candidates'] + 1 )):
            it_step_zfill = str(it_step).zfill(5)
            check_path='./'+str(it_subsys_nr)+'/'+it_step_zfill
            if count == 0:
                cf.check_file(training_iterative_apath+'/inputs/'+it_subsys_nr+'.lmp',0,True,'Input data file (lmp file) not present.')
                lammps_data = cf.read_file(training_iterative_apath+'/inputs/'+it_subsys_nr+'.lmp')
                index = [idx for idx, s in enumerate(lammps_data) if 'Atoms' in s][0]
                del lammps_data[0:index+2]
                lammps_data = lammps_data[0:config_json['subsys_nr'][it_subsys_nr]['nb_atm']+1]
                lammps_data = [' '.join(f.replace('\n','').split()) for f in lammps_data]
                lammps_data = [g.split(' ')[1:2] for g in lammps_data]
                type_atom_array = np.asarray(lammps_data,dtype=np.int64).flatten()
                type_atom_array = type_atom_array - 1
                np.savetxt('./'+str(it_subsys_nr)+'/type-disturbed.raw',type_atom_array,delimiter=' ',newline=' ',fmt='%d')
                np.savetxt(training_iterative_apath+'/data/'+it_subsys_nr+'-disturbed_'+current_iteration_zfill+'/type.raw',type_atom_array,delimiter=' ',newline=' ',fmt='%d')
                cp2k_out = cf.read_file(check_path+'/2_labeling_'+it_step_zfill+'.out')
                cp2k_out = [zzz for zzz in cp2k_out if 'CP2K| version string:' in zzz]
                cp2k_out = [' '.join(f.replace('\n','').split()) for f in cp2k_out]
                cp2k_out = [g.split(' ')[-1] for g in cp2k_out]
                cp2k_version = float(cp2k_out[0])

            stress_xyz = cf.read_file(check_path+'/2_labeling_'+it_step_zfill+'-Stress_Tensor.st')

            if cp2k_version < 8.1:
                del stress_xyz[0:4]
                stress_xyz = stress_xyz[0:3]
                stress_xyz = [' '.join(f.replace('\n','').split()) for f in stress_xyz]
                stress_xyz = [g.split(' ')[1:4] for g in stress_xyz]
                stress_xyz_array = np.asarray(stress_xyz,dtype=np.float64).flatten()
                virial_array_raw[count,:] = stress_xyz_array * volume[count] / eV_per_A3_to_GPa
            else:
                True
                ### TODO WRITE

            force_cp2k = cf.read_file(check_path+'/2_labeling_'+it_step_zfill+'-Forces.for')
            del force_cp2k[0:4]
            del force_cp2k[-1]
            force_cp2k = [' '.join(f.replace('\n','').split()) for f in force_cp2k]
            force_cp2k = [g.split(' ')[3:] for g in force_cp2k]
            force_array = np.asarray(force_cp2k,dtype=np.float64).flatten()
            force_array_raw[count,:] = force_array*au_to_eV_per_A

            energy_cp2k = cf.read_file(check_path+'/2_labeling_'+it_step_zfill+'-Force_Eval.fe')
            del energy_cp2k[0]
            del energy_cp2k[-1]
            energy_cp2k = [' '.join(f.replace('\n','').split()) for f in energy_cp2k]
            energy_cp2k = [g.split(' ')[-1] for g in energy_cp2k]
            energy_array = np.asarray(energy_cp2k,dtype=np.float64).flatten()
            energy_array_raw[count] = energy_array*Ha_to_eV

            coord_xyz = cf.read_file(check_path+'/labeling_'+it_step_zfill+'.xyz')
            del coord_xyz[0:2]
            coord_xyz = [' '.join(f.replace('\n','').split()) for f in coord_xyz]
            coord_xyz = [g.split(' ')[1:] for g in coord_xyz]
            coord_array = np.asarray(coord_xyz,dtype=np.float64).flatten()
            coord_array_raw[count,:] = coord_array

            del coord_xyz, energy_cp2k, force_cp2k, stress_xyz, check_path, it_step_zfill
        del it_step

        box_array_raw[:,0] = config_json['subsys_nr'][it_subsys_nr]['cell'][0]
        box_array_raw[:,4] = config_json['subsys_nr'][it_subsys_nr]['cell'][1]
        box_array_raw[:,8] = config_json['subsys_nr'][it_subsys_nr]['cell'][2]

        np.savetxt('./'+str(it_subsys_nr)+'/box-disturbed.raw',box_array_raw,delimiter=' ')
        np.save(training_iterative_apath+'/data/'+it_subsys_nr+'-disturbed_'+current_iteration_zfill+'/set.000/box',box_array_raw)
        np.savetxt('./'+str(it_subsys_nr)+'/virial-disturbed.raw',virial_array_raw,delimiter=' ')
        np.save(training_iterative_apath+'/data/'+it_subsys_nr+'-disturbed_'+current_iteration_zfill+'/set.000/virial',virial_array_raw)
        np.savetxt('./'+str(it_subsys_nr)+'/force-disturbed.raw',force_array_raw,delimiter=' ')
        np.save(training_iterative_apath+'/data/'+it_subsys_nr+'-disturbed_'+current_iteration_zfill+'/set.000/force', force_array_raw)
        np.savetxt('./'+str(it_subsys_nr)+'/energy-disturbed.raw',energy_array_raw,delimiter=' ')
        np.save(training_iterative_apath+'/data/'+it_subsys_nr+'-disturbed_'+current_iteration_zfill+'/set.000/energy',energy_array_raw)
        np.savetxt('./'+str(it_subsys_nr)+'/coord-disturbed.raw',coord_array_raw,delimiter=' ')
        np.save(training_iterative_apath+'/data/'+it_subsys_nr+'-disturbed_'+current_iteration_zfill+'/set.000/coord',coord_array_raw)

        del box_array_raw, virial_array_raw, force_array_raw, energy_array_raw, coord_array_raw

labeling_json['is_extracted'] = True
cf.json_dump(labeling_json,labeling_json_fpath,True,'labeling.json')

logging.info('The labeling extraction phase is a success!')

### Cleaning
del config_json, config_json_fpath, training_iterative_apath
del current_iteration, current_iteration_zfill
del labeling_json, labeling_json_fpath
del deepmd_iterative_apath

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()