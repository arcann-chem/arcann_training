## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import numpy as np

### Constants
Ha_to_eV = np.float64(27.211386245988)
Bohr_to_A = np.float64(0.529177210903)
au_to_eV_per_A = np.float64(Ha_to_eV/Bohr_to_A)
eV_per_A3_to_GPa = np.float64(160.21766208)

training_iterative_apath = Path("..").resolve()
### Check if the deepmd_iterative_apath is defined
deepmd_iterative_apath_error = 1
if "deepmd_iterative_apath" in globals():
    if (Path(deepmd_iterative_apath)/"tools"/"common_functions.py").is_file():
        deepmd_iterative_apath = Path(deepmd_iterative_apath)
        deepmd_iterative_apath_error = 0
elif (Path().home()/"deepmd_iterative_py"/"tools"/"common_functions.py").is_file():
    deepmd_iterative_apath = Path().home()/"deepmd_iterative_py"
    deepmd_iterative_apath_error = 0
elif (training_iterative_apath/"control"/"path").is_file():
    deepmd_iterative_apath = Path((training_iterative_apath/"control"/"path").read_text())
    if (deepmd_iterative_apath/"tools"/"common_functions.py").is_file():
        deepmd_iterative_apath_error = 0
if deepmd_iterative_apath_error == 1:
    logging.critical("Can\'t find common_functions.py in usual places:")
    logging.critical("deepmd_iterative_apath variable or ~/deepmd_iterative_py or in the path file in control")
    logging.critical("Aborting...")
    sys.exit(1)
sys.path.insert(0, str(deepmd_iterative_apath/"tools"))
del deepmd_iterative_apath_error
import common_functions as cf

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
config_json = cf.json_read((control_apath/"config.json"),True,True)
labeling_json = cf.json_read((control_apath/("labeling_"+current_iteration_zfill+".json")),True,True)

### Checks
if labeling_json["is_checked"] is False:
    logging.critical("Lock found. Run/Check first: labeling3_check.py")
    logging.critical("Aborting...")
    sys.exit(1)

### Launch the extractions
(training_iterative_apath/"data").mkdir(exist_ok=True)

for it_subsys_nr in labeling_json["subsys_nr"]:
    subsys_path = Path(".").resolve()/it_subsys_nr

    data_apath = training_iterative_apath/"data"/(it_subsys_nr+"_"+current_iteration_zfill)
    data_apath.mkdir(exist_ok=True)
    (data_apath/"set.000").mkdir(exist_ok=True)

    force_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates"], config_json["subsys_nr"][it_subsys_nr]["nb_atm"] * 3 ))
    energy_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates"]))
    coord_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates"], config_json["subsys_nr"][it_subsys_nr]["nb_atm"] * 3 ))
    box_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates"], 9))
    virial_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates"], 9))
    wannier_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates"], config_json["subsys_nr"][it_subsys_nr]["nb_atm"] * 3 ))

    box_array_raw[:,0] = config_json["subsys_nr"][it_subsys_nr]["cell"][0]
    box_array_raw[:,4] = config_json["subsys_nr"][it_subsys_nr]["cell"][1]
    box_array_raw[:,8] = config_json["subsys_nr"][it_subsys_nr]["cell"][2]

    volume = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates"]))
    volume = box_array_raw[:,0] * box_array_raw[:,4] * box_array_raw[:,8]


    for it_step in range(1, labeling_json["subsys_nr"][it_subsys_nr]["candidates"] + 1):
        it_step_zfill = str(it_step).zfill(5)
        local_apath = Path(".").resolve()/it_subsys_nr/it_step_zfill

        if it_step == 1:
            cf.check_file(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp"),True,True,"Input data file (lmp file) not present.")
            lammps_data = cf.read_file(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp"))
            index = [idx for idx, s in enumerate(lammps_data) if "Atoms" in s][0]
            del lammps_data[0:index+2]
            lammps_data = lammps_data[0:config_json["subsys_nr"][it_subsys_nr]["nb_atm"]+1]
            lammps_data = [" ".join(f.replace("\n","").split()) for f in lammps_data]
            lammps_data = [g.split(" ")[1:2] for g in lammps_data]
            type_atom_array = np.asarray(lammps_data,dtype=np.int64).flatten()
            type_atom_array = type_atom_array - 1
            np.savetxt(str(subsys_path/"type.raw"),type_atom_array,delimiter=" ",newline=" ",fmt="%d")
            np.savetxt(str(data_apath/"type.raw"),type_atom_array,delimiter=" ",newline=" ",fmt="%d")
            cp2k_out = cf.read_file(local_apath/("2_labeling_"+it_step_zfill+".out"))
            cp2k_out = [zzz for zzz in cp2k_out if "CP2K| version string:" in zzz]
            cp2k_out = [" ".join(f.replace("\n","").split()) for f in cp2k_out]
            cp2k_out = [g.split(" ")[-1] for g in cp2k_out]
            cp2k_version = float(cp2k_out[0])
            del lammps_data, cp2k_out, type_atom_array, index

        stress_xyz = cf.read_file(local_apath/("2_labeling_"+it_step_zfill+"-Stress_Tensor.st"))
        if cp2k_version < 8.1:
            del stress_xyz[0:4]
            stress_xyz = stress_xyz[0:3]
            stress_xyz = [" ".join(f.replace("\n","").split()) for f in stress_xyz]
            stress_xyz = [g.split(" ")[1:4] for g in stress_xyz]
            stress_xyz_array = np.asarray(stress_xyz,dtype=np.float64).flatten()
            virial_array_raw[it_step-1,:] = stress_xyz_array * volume[it_step-1] / eV_per_A3_to_GPa
            del stress_xyz, stress_xyz_array
        else:
            True
            ### TODO

        force_cp2k = cf.read_file(local_apath/("2_labeling_"+it_step_zfill+"-Forces.for"))
        del force_cp2k[0:4]
        del force_cp2k[-1]
        force_cp2k = [" ".join(f.replace("\n","").split()) for f in force_cp2k]
        force_cp2k = [g.split(" ")[3:] for g in force_cp2k]
        force_array = np.asarray(force_cp2k,dtype=np.float64).flatten()
        force_array_raw[it_step-1,:] = force_array*au_to_eV_per_A
        del force_array, force_cp2k

        energy_cp2k = cf.read_file(local_apath/("2_labeling_"+it_step_zfill+"-Force_Eval.fe"))
        del energy_cp2k[0]
        del energy_cp2k[-1]
        energy_cp2k = [" ".join(f.replace("\n","").split()) for f in energy_cp2k]
        energy_cp2k = [g.split(" ")[-1] for g in energy_cp2k]
        energy_array = np.asarray(energy_cp2k,dtype=np.float64).flatten()
        energy_array_raw[it_step-1] = energy_array*Ha_to_eV
        del energy_array, energy_cp2k

        coord_xyz = cf.read_file(local_apath/("labeling_"+it_step_zfill+".xyz"))
        del coord_xyz[0:2]
        coord_xyz = [" ".join(f.replace("\n","").split()) for f in coord_xyz]
        coord_xyz = [g.split(" ")[1:] for g in coord_xyz]
        coord_array = np.asarray(coord_xyz,dtype=np.float64).flatten()
        coord_array_raw[it_step-1,:] = coord_array
        del coord_array, coord_xyz

        if (local_apath/("labeling_"+it_step_zfill+"-Wannier.xyz")).is_file():
            wannier_xyz = cf.read_file(local_apath/("labeling_"+it_step_zfill+"-Wannier.xyz"))
            del wannier_xyz[0:2+config_json["subsys_nr"][it_subsys_nr]["nb_atm"]]
            wannier_xyz = [" ".join(f.replace("\n","").split()) for f in wannier_xyz]
            wannier_xyz = [g.split(" ")[1:] for g in wannier_xyz]
            wannier_array = np.asarray(wannier_xyz,dtype=np.float64).flatten()
            wannier_array_raw[it_step-1,:] = wannier_array
            del wannier_array, wannier_xyz

        del it_step_zfill, local_apath
    del it_step

    np.savetxt(str(subsys_path/"box.raw"),box_array_raw,delimiter=" ")
    np.save(str(data_apath/"set.000"/"box"),box_array_raw)
    np.savetxt(str(subsys_path/"virial.raw"),virial_array_raw,delimiter=" ")
    np.save(str(data_apath/"set.000"/"virial"),virial_array_raw)
    np.savetxt(str(subsys_path/"force.raw"),virial_array_raw,delimiter=" ")
    np.save(str(data_apath/"set.000"/"force"),force_array_raw)
    np.savetxt(str(subsys_path/"energy.raw"),energy_array_raw,delimiter=" ")
    np.save(str(data_apath/"set.000"/"energy"),energy_array_raw)
    np.savetxt(str(subsys_path/"coord.raw"),coord_array_raw,delimiter=" ")
    np.save(str(data_apath/"set.000"/"coord"),coord_array_raw)
    np.savetxt(str(subsys_path/"wannier.raw"),wannier_array_raw,delimiter=" ")
    np.save(str(data_apath/"set.000"/"wannier"),wannier_array_raw)

    del box_array_raw, virial_array_raw, force_array_raw, energy_array_raw, coord_array_raw,wannier_array_raw

    if labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"] != 0 :
        data_apath = training_iterative_apath/"data"/(it_subsys_nr+"-disturbed_"+current_iteration_zfill)
        data_apath.mkdir(exist_ok=True)
        (data_apath/"set.000").mkdir(exist_ok=True)

        force_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"], config_json["subsys_nr"][it_subsys_nr]["nb_atm"] * 3 ))
        energy_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"]))
        coord_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"], config_json["subsys_nr"][it_subsys_nr]["nb_atm"] * 3 ))
        box_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"], 9))
        virial_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"], 9))
        wannier_array_raw = np.zeros((labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"], config_json["subsys_nr"][it_subsys_nr]["nb_atm"] * 3 ))

        for count,it_step in enumerate(range(labeling_json["subsys_nr"][it_subsys_nr]["candidates"] + 1, labeling_json["subsys_nr"][it_subsys_nr]["candidates"] + labeling_json["subsys_nr"][it_subsys_nr]["candidates_disturbed"] + 1 )):
            it_step_zfill = str(it_step).zfill(5)
            local_apath = Path(".").resolve()/it_subsys_nr/it_step_zfill
            if count == 0:
                cf.check_file(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp"),True,True,"Input data file (lmp file) not present.")
                lammps_data = cf.read_file(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp"))
                index = [idx for idx, s in enumerate(lammps_data) if "Atoms" in s][0]
                del lammps_data[0:index+2]
                lammps_data = lammps_data[0:config_json["subsys_nr"][it_subsys_nr]["nb_atm"]+1]
                lammps_data = [" ".join(f.replace("\n","").split()) for f in lammps_data]
                lammps_data = [g.split(" ")[1:2] for g in lammps_data]
                type_atom_array = np.asarray(lammps_data,dtype=np.int64).flatten()
                type_atom_array = type_atom_array - 1
                np.savetxt(str(subsys_path/"type-disturbed.raw"),type_atom_array,delimiter=" ",newline=" ",fmt="%d")
                np.savetxt(str(data_apath/"type.raw"),type_atom_array,delimiter=" ",newline=" ",fmt="%d")
                cp2k_out = cf.read_file(local_apath/("2_labeling_"+it_step_zfill+".out"))
                cp2k_out = [zzz for zzz in cp2k_out if "CP2K| version string:" in zzz]
                cp2k_out = [" ".join(f.replace("\n","").split()) for f in cp2k_out]
                cp2k_out = [g.split(" ")[-1] for g in cp2k_out]
                cp2k_version = float(cp2k_out[0])
                del lammps_data, cp2k_out, type_atom_array, index

            stress_xyz = cf.read_file(local_apath/("2_labeling_"+it_step_zfill+"-Stress_Tensor.st"))
            if cp2k_version < 8.1:
                del stress_xyz[0:4]
                stress_xyz = stress_xyz[0:3]
                stress_xyz = [" ".join(f.replace("\n","").split()) for f in stress_xyz]
                stress_xyz = [g.split(" ")[1:4] for g in stress_xyz]
                stress_xyz_array = np.asarray(stress_xyz,dtype=np.float64).flatten()
                virial_array_raw[count,:] = stress_xyz_array * volume[count] / eV_per_A3_to_GPa
                del stress_xyz, stress_xyz_array
            else:
                True
                ### TODO

            force_cp2k = cf.read_file(local_apath/("2_labeling_"+it_step_zfill+"-Forces.for"))
            del force_cp2k[0:4]
            del force_cp2k[-1]
            force_cp2k = [" ".join(f.replace("\n","").split()) for f in force_cp2k]
            force_cp2k = [g.split(" ")[3:] for g in force_cp2k]
            force_array = np.asarray(force_cp2k,dtype=np.float64).flatten()
            force_array_raw[count,:] = force_array*au_to_eV_per_A
            del force_array, force_cp2k

            energy_cp2k = cf.read_file(local_apath/("2_labeling_"+it_step_zfill+"-Force_Eval.fe"))
            del energy_cp2k[0]
            del energy_cp2k[-1]
            energy_cp2k = [" ".join(f.replace("\n","").split()) for f in energy_cp2k]
            energy_cp2k = [g.split(" ")[-1] for g in energy_cp2k]
            energy_array = np.asarray(energy_cp2k,dtype=np.float64).flatten()
            energy_array_raw[count] = energy_array*Ha_to_eV
            del energy_array, energy_cp2k

            coord_xyz = cf.read_file(local_apath/("labeling_"+it_step_zfill+".xyz"))
            del coord_xyz[0:2]
            coord_xyz = [" ".join(f.replace("\n","").split()) for f in coord_xyz]
            coord_xyz = [g.split(" ")[1:] for g in coord_xyz]
            coord_array = np.asarray(coord_xyz,dtype=np.float64).flatten()
            coord_array_raw[count,:] = coord_array
            del coord_array, coord_xyz

            if (local_apath/("labeling_"+it_step_zfill+"-Wannier.xyz")).is_file():
                wannier_xyz = cf.read_file(local_apath/("labeling_"+it_step_zfill+"-Wannier.xyz"))
                del wannier_xyz[0:2+config_json["subsys_nr"][it_subsys_nr]["nb_atm"]]
                wannier_xyz = [" ".join(f.replace("\n","").split()) for f in wannier_xyz]
                wannier_xyz = [g.split(" ")[1:] for g in wannier_xyz]
                wannier_array = np.asarray(wannier_xyz,dtype=np.float64).flatten()
                wannier_array_raw[it_step-1,:] = wannier_array
                del wannier_array, wannier_xyz

            del it_step_zfill, local_apath
        del it_step

        box_array_raw[:,0] = config_json["subsys_nr"][it_subsys_nr]["cell"][0]
        box_array_raw[:,4] = config_json["subsys_nr"][it_subsys_nr]["cell"][1]
        box_array_raw[:,8] = config_json["subsys_nr"][it_subsys_nr]["cell"][2]

        np.savetxt(str(subsys_path/"box-disturbed.raw"),box_array_raw,delimiter=" ")
        np.save(str(data_apath/"set.000"/"box"),box_array_raw)
        np.savetxt(str(subsys_path/"virial-disturbed.raw"),virial_array_raw,delimiter=" ")
        np.save(str(data_apath/"set.000"/"virial"),virial_array_raw)
        np.savetxt(str(subsys_path/"force-disturbed.raw"),virial_array_raw,delimiter=" ")
        np.save(str(data_apath/"set.000"/"force"),force_array_raw)
        np.savetxt(str(subsys_path/"energy-disturbed.raw"),energy_array_raw,delimiter=" ")
        np.save(str(data_apath/"set.000"/"energy"),energy_array_raw)
        np.savetxt(str(subsys_path/"coord-disturbed.raw"),coord_array_raw,delimiter=" ")
        np.save(str(data_apath/"set.000"/"coord"),coord_array_raw)
        np.savetxt(str(subsys_path/"wannier-disturbed.raw"),wannier_array_raw,delimiter=" ")
        np.save(str(data_apath/"set.000"/"wannier"),wannier_array_raw)

        del box_array_raw, virial_array_raw, force_array_raw, energy_array_raw, coord_array_raw, wannier_array_raw
del volume, cp2k_version, count, subsys_path, data_apath, it_subsys_nr
del Ha_to_eV, Bohr_to_A, au_to_eV_per_A,eV_per_A3_to_GPa

labeling_json["is_extracted"] = True
cf.json_dump(labeling_json,(control_apath/("labeling_"+current_iteration_zfill+".json")),True)

logging.info("Labeling: Extraction phase is a success!")

### Cleaning
del config_json, training_iterative_apath, control_apath
del current_iteration, current_iteration_zfill
del labeling_json
del deepmd_iterative_apath

del sys, Path, logging, cf
del np
import gc; gc.collect(); del gc
exit()