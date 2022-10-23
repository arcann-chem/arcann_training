## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## These are the default
atomsk_fpath: str ="/gpfswork/rech/nvs/commun/programs/apps/atomsk/0.11.2/bin/atomsk"
# vmd_fpath: str=""
# disturbed_min_value: list = [0.0, 0.0]
# disturbed_candidates_value: list = [0.0, 0.0]

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO,format="%(levelname)s: %(message)s")

import os
import subprocess
import numpy as np

if "atomsk_fpath" not in globals():
    atomsk = subprocess.call(["command","-v","atomsk"])
    if atomsk == 1:
        logging.critical("atmsk not found.")
        logging.critical("Aborting...")
        sys.exit(1)
    else:
        atomsk_bin = "atomsk"
else:
    atomsk = subprocess.call(["command","-v",atomsk_fpath])
    if atomsk == 1:
        logging.critical("Your path seems shifty: "+ atomsk_fpath)
        logging.critical("Aborting...")
        sys.exit(1)
    else:
        atomsk_bin = atomsk_fpath
del atomsk

if "vmd_fpath" not in globals():
    vmd = subprocess.call(["command", "-v", "vmd"])
    if vmd == 1:
        logging.critical("vmd not found.")
        logging.critical("Aborting...")
        sys.exit(1)
    else:
        vmd_bin = "vmd"
else:
    vmd = subprocess.call(["command", "-v", vmd_fpath])
    if vmd == 1:
        logging.critical("Your path seems shifty: "+ vmd_fpath)
        logging.critical("Aborting...")
        sys.exit(1)
    else:
        vmd_bin = vmd_fpath
del vmd

training_iterative_apath = Path("..").resolve()
### Check if the deepmd_iterative_apath is defined
deepmd_iterative_apath_error = 1
if "deepmd_iterative_apath" in globals():
    if (Path(deepmd_iterative_apath)/"scripts"/"common_functions.py").is_file():
        deepmd_iterative_apath = Path(deepmd_iterative_apath)
        deepmd_iterative_apath_error = 0
elif (Path().home()/"deepmd_iterative_py"/"scripts"/"common_functions.py").is_file():
    deepmd_iterative_apath = Path().home()/"deepmd_iterative_py"
    deepmd_iterative_apath_error = 0
elif (training_iterative_apath/"control"/"path").is_file():
    deepmd_iterative_apath = Path((training_iterative_apath/"control"/"path").read_text())
    if (deepmd_iterative_apath/"scripts"/"common_functions.py").is_file():
        deepmd_iterative_apath_error = 0
if deepmd_iterative_apath_error == 1:
    logging.critical("Can\'t find common_functions.py in usual places:")
    logging.critical("deepmd_iterative_apath variable or ~/deepmd_iterative_py or in the path file in control")
    logging.critical("Aborting...")
    sys.exit(1)
sys.path.insert(0, str(deepmd_iterative_apath/"scripts"))
del deepmd_iterative_apath_error
import common_functions as cf

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
exploration_json = cf.json_read((control_apath/("exploration_"+current_iteration_zfill+".json")),True,True)
if int(current_iteration_zfill) > 1:
    previous_iteration_zfill = str(int(current_iteration_zfill) - 1).zfill(3)
    prevexploration_json = cf.json_read((control_apath/("exploration_"+previous_iteration_zfill+".json")),True,True)
scripts_apath = deepmd_iterative_apath/"scripts"

### Checks
if exploration_json["is_deviated"] is False:
    logging.critical("Lock found. Run/Check first: exploration4_devi.py")
    logging.critical("Aborting...")
    sys.exit(1)

cf.check_file(scripts_apath/"vmd_dcd_selection_index.tcl",True,True,"The vmd_dcd_selection_index.tcl file is missing")
master_vmd_tcl = cf.read_file(scripts_apath/"vmd_dcd_selection_index.tcl")

starting_structures_apath = training_iterative_apath/"starting_structures"
starting_structures_apath.mkdir(exist_ok=True)
cf.check_dir(starting_structures_apath,True)

###TODO Use of atomsk --> work avoid loop and cat

### Extract for labeling
for it0_subsys_nr,it_subsys_nr in enumerate(config_json["subsys_nr"]):

    print_freq = exploration_json["subsys_nr"][it_subsys_nr]["print_freq"]
    if exploration_json["exploration_type"] == "lammps":
        cf.check_file(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp"),True,True)
        subprocess.call([atomsk_bin,str(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp")),"pdb",str(training_iterative_apath/"inputs"/it_subsys_nr),"-ow"],\
            stdout=subprocess.DEVNULL,\
            stderr=subprocess.STDOUT)
        topo_file=training_iterative_apath/"inputs"/(it_subsys_nr+".pdb")
    elif exploration_json["exploration_type"] == "i-PI":
        cf.check_file(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp"),True,True)
        cf.check_file(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp"),True,True)
        subprocess.call([atomsk_bin,str(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp")),"pdb",str(training_iterative_apath/"inputs"/it_subsys_nr),"-ow"],\
            stdout=subprocess.DEVNULL,\
            stderr=subprocess.STDOUT)
        topo_file=training_iterative_apath/"inputs"/(it_subsys_nr+".pdb")

    for it_nnp in range(1,  exploration_json["nb_nnp"] + 1):
        for it_each in range(1, exploration_json["nb_traj"]+1):

            local_apath = Path(".").resolve()/str(it_subsys_nr)/str(it_nnp)/(str(it_each).zfill(5))
            devi_info_json = cf.json_read(local_apath/"devi_info.json",True,False)
            devi_index_json = cf.json_read(local_apath/"devi_index.json",True,False)

            ### Selection of the min for the next iteration starting point
            if devi_info_json["min_index"] != -1:

                min_index = int(devi_info_json["min_index"] / print_freq)
                (local_apath/"min.vmd").write_text(str(min_index))
                del min_index

                if exploration_json["exploration_type"] == "lammps":
                    traj_file = local_apath/(str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".dcd")
                elif exploration_json["exploration_type"] == "i-PI":
                    traj_file = local_apath/(str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+"_random_beads.dcd")

                min_file_name = current_iteration_zfill+"_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+str(it_each).zfill(5)

                vmd_tcl = master_vmd_tcl.copy()
                vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_PDB_FILE_",str(topo_file))
                vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_DCD_FILE_",str(traj_file))
                vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_SELECTION_FILE_",str((local_apath/"min.vmd")))
                vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_XYZ_OUT_",str(starting_structures_apath/(min_file_name+".xyz")))
                cf.write_file((local_apath/"vmd.tcl"),vmd_tcl)
                del vmd_tcl, traj_file

                ### VMD DCD -> XYZ
                cf.remove_file(starting_structures_apath/(min_file_name+".xyz"))
                subprocess.call([vmd_bin,"-e",str((local_apath/"vmd.tcl")),"-dispdev", "text"],\
                    stdout=subprocess.DEVNULL,\
                    stderr=subprocess.STDOUT)
                cf.remove_file((local_apath/"vmd.tcl"))
                cf.remove_file((local_apath/"min.vmd"))

                ### Atomsk XYZ -> LMP
                cf.remove_file(starting_structures_apath/(min_file_name+".lmp"))
                subprocess.call([atomsk_bin, "-ow", str(starting_structures_apath/(min_file_name+".xyz")),\
                        "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][0]), "H1",\
                        "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][1]), "H2",\
                        "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][2]), "H3",\
                        str(starting_structures_apath/(min_file_name+".lmp"))],\
                        stdout=subprocess.DEVNULL,\
                        stderr=subprocess.STDOUT)

                if ("disturbed_min_value" in globals() and disturbed_min_value[it0_subsys_nr] != 0) \
                        or (int(current_iteration_zfill) > 1 and prevexploration_json["subsys_nr"][it_subsys_nr]["disturbed_min"]):

                    disturbed_min_value_subsys = disturbed_min_value[it0_subsys_nr] if "disturbed_min_value" in globals() else prevexploration_json["subsys_nr"][it_subsys_nr]["disturbed_min"]
                    cf.remove_file((starting_structures_apath/(min_file_name+"_disturbed.xyz")))
                    (starting_structures_apath/(min_file_name+"_disturbed.xyz")).write_text((starting_structures_apath/(min_file_name+".xyz")).read_text())

                    ### Atomsk XYZ ==> XYZ_disturbed
                    subprocess.call([atomsk_bin, "-ow", str(starting_structures_apath/(min_file_name+"_disturbed.xyz")),\
                        "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][0]), "H1",\
                        "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][1]), "H2",\
                        "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][2]), "H3",\
                        "-disturb", str(disturbed_min_value_subsys),\
                        "xyz"],\
                        stdout=subprocess.DEVNULL,\
                        stderr=subprocess.STDOUT)
                    ### Atomsk XYZ -> LMP
                    cf.remove_file((starting_structures_apath/(min_file_name+"_disturbed.lmp")))
                    subprocess.call([atomsk_bin, "-ow", str(starting_structures_apath/(min_file_name+"_disturbed.xyz")),\
                        "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][0]), "H1",\
                        "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][1]), "H2",\
                        "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][2]), "H3",\
                        str(starting_structures_apath/(min_file_name+"_disturbed.lmp"))],\
                        stdout=subprocess.DEVNULL,\
                        stderr=subprocess.STDOUT)

                    del disturbed_min_value_subsys
                del min_file_name

            ### Selection of labeling XYZ
            if len(devi_index_json["candidates_kept_ind"]) != 0:
                candidates_index = np.array(devi_index_json["candidates_kept_ind"])
                candidates_index = candidates_index / print_freq
                candidates_index = candidates_index.astype(int)
                candidates_index = map(str, candidates_index.astype(int))
                candidates_index = [ zzz + "\n" for zzz in candidates_index]
                cf.write_file(local_apath/"label.vmd", candidates_index)
                del candidates_index

                if exploration_json["exploration_type"] == "lammps":
                    traj_file = local_apath/(str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".dcd")
                elif exploration_json["exploration_type"] == "i-PI":
                    traj_file = local_apath/(str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+"_random_beads.dcd")

                vmd_tcl = master_vmd_tcl.copy()
                vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_PDB_FILE_",str(topo_file))
                vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_DCD_FILE_",str(traj_file))
                vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_SELECTION_FILE_",str((local_apath/"label.vmd")))
                vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_XYZ_OUT_",str(local_apath/("vmd_${j}.xyz")))
                cf.write_file((local_apath/"vmd.tcl"),vmd_tcl)
                del vmd_tcl, traj_file

                ### VMD DCD -> A lot of XYZ
                subprocess.call([vmd_bin,"-e",str((local_apath/"vmd.tcl")),"-dispdev", "text"],\
                    stdout=subprocess.DEVNULL,\
                    stderr=subprocess.STDOUT)
                cf.remove_file((local_apath/"vmd.tcl"))
                cf.remove_file((local_apath/"min.vmd"))

                cf.remove_file(local_apath/("temp_candidates_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".xyz"))
                ####TODO Not Path friendly / Replace with either subprocess call or read python
                os.system("cat "+str(local_apath)+"/vmd_*.xyz >> "+str(local_apath)+"/temp_candidates_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".xyz")

                if "disturbed_candidates_value" in globals() and disturbed_candidates_value[it0_subsys_nr] != 0:
                    vmd_xyz_files = [zzz for zzz in local_apath.glob("vmd_*")]
                    for it_vmd_xyz_files in vmd_xyz_files:
                        ### Atomsk XYZ ==> XYZ_disturbed
                        subprocess.call([atomsk_bin, "-ow", str(it_vmd_xyz_files),\
                            "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][0]), "H1",\
                            "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][1]), "H2",\
                            "-cell", "set", str(config_json["subsys_nr"][it_subsys_nr]["cell"][2]), "H3",\
                            "-disturb", str(disturbed_candidates_value[it0_subsys_nr]),\
                            "xyz", str(it_vmd_xyz_files)+"_disturbed"],\
                            stdout=subprocess.DEVNULL,\
                            stderr=subprocess.STDOUT)
                    del it_vmd_xyz_files, vmd_xyz_files

                    cf.remove_file(local_apath/("temp_candidates_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+"_disturbed.xyz"))
                    ####TODO Not Path friendly / Replace with either subprocess call or read python
                    os.system("cat "+str(local_apath)+"/vmd_*_disturbed.xyz >> "+str(local_apath)+"/temp_candidates_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+"_disturbed.xyz")

                ### Remove all vmd_*.xyz
                cf.remove_file_glob(local_apath,"vmd_*.xyz")

            if devi_info_json["min_index"] == -1:
                logging.warning(str(it_subsys_nr)+" / "+str(it_nnp)+" / "+str(it_each)+" has been processed but no candidates or minimum")
            else:
                logging.info(str(it_subsys_nr)+" / "+str(it_nnp)+" / "+str(it_each)+" has been processed")

            del devi_info_json, devi_index_json
        del it_each

    subsys_apath = Path(".").resolve()/str(it_subsys_nr)

    if "disturbed_candidates_value" in globals() and disturbed_candidates_value[it0_subsys_nr] != 0:

        cf.remove_file(subsys_apath/("candidates_"+str(it_subsys_nr)+"_"+current_iteration_zfill+"_disturbed.xyz"))
        ####TODO Not Path friendly / Replace with either subprocess call or read python
        os.system("cat "+str(subsys_apath)+"/*/*/temp_candidates_*_disturbed.xyz >> "+str(subsys_apath)+"/candidates_"+str(it_subsys_nr)+"_"+current_iteration_zfill+"_disturbed.xyz")
        cf.remove_file_glob(subsys_apath,"**/temp_candidates_*_disturbed.xyz")

        exploration_json["subsys_nr"][it_subsys_nr]["disturbed_candidates"] = True
        exploration_json["subsys_nr"][it_subsys_nr]["disturbed_candidates_value"] = disturbed_candidates_value[it0_subsys_nr]
    else:
        exploration_json["subsys_nr"][it_subsys_nr]["disturbed_candidates"] = False
        exploration_json["subsys_nr"][it_subsys_nr]["disturbed_candidates_value"] = 0

    if "disturbed_min_value_subsys" in globals():
        exploration_json["subsys_nr"][it_subsys_nr]["disturbed_min"] = True
        exploration_json["subsys_nr"][it_subsys_nr]["disturbed_min_value"] = disturbed_min_value_subsys
    else:
        exploration_json["subsys_nr"][it_subsys_nr]["disturbed_min"] = False
        exploration_json["subsys_nr"][it_subsys_nr]["disturbed_min_value"] = 0

    cf.remove_file(subsys_apath/("candidates_"+str(it_subsys_nr)+"_"+current_iteration_zfill+".xyz"))
    ####TODO Not Path friendly / Replace with either subprocess call or read python
    os.system("cat "+str(subsys_apath)+"/*/*/temp_candidates_*.xyz >> "+str(subsys_apath)+"/candidates_"+str(it_subsys_nr)+"_"+current_iteration_zfill+".xyz")
    cf.remove_file_glob(subsys_apath,"**/temp_candidates_*.xyz")

    del it_nnp, subsys_apath, local_apath
del it0_subsys_nr, it_subsys_nr, topo_file, print_freq
del master_vmd_tcl, atomsk_bin, vmd_bin

exploration_json["is_extracted"] = True
cf.json_dump(exploration_json,(control_apath/("exploration_"+current_iteration_zfill+".json")),True)

logging.info("Exploration: Extraction phase is a success!")

### Cleaning
if int(current_iteration_zfill) > 1:
    del previous_iteration_zfill, prevexploration_json
del config_json, training_iterative_apath, scripts_apath, control_apath, starting_structures_apath
del current_iteration_zfill
del exploration_json
del deepmd_iterative_apath

del sys, Path, logging, cf
del os, np, subprocess
import gc; gc.collect(); del gc
print(globals())
exit()