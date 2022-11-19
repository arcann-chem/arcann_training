## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""
## These are the default
atomsk_fpath: str ="/gpfswork/rech/nvs/commun/programs/apps/atomsk/0.11.2/bin/atomsk"
# vmd_fpath: str=""

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import os
import subprocess
import numpy as np
import random

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
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
exploration_json = cf.json_read((control_apath/("exploration_"+current_iteration_zfill+".json")),True,True)
scripts_apath = deepmd_iterative_apath/"tools"

### Checks
if "i-PI" not in exploration_json["exploration_type"]:
    logging.critical("This is not an i-PI exploration")
    logging.critical("Aborting...")
    sys.exit(1)

if not exploration_json["is_checked"]:
    logging.critical("Lock found. Run/Check first: exploration3_check.py")
    logging.critical("Aborting...")
    sys.exit(1)

master_vmd_tcl = cf.read_file(scripts_apath/"vmd_xyz_selection_index.tcl")

nb_beads = 8

for it0_subsys_nr,it_subsys_nr in enumerate(config_json["subsys_nr"]):
    cf.check_file(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp"),True,True)
    subprocess.call([atomsk_bin,str(training_iterative_apath/"inputs"/(it_subsys_nr+".lmp")),"pdb",str(training_iterative_apath/"inputs"/it_subsys_nr),"-ow"],\
        stdout=subprocess.DEVNULL,\
        stderr=subprocess.STDOUT)
    topo_file=training_iterative_apath/"inputs"/(it_subsys_nr+".pdb")
    
    nb_pos = int(exploration_json["subsys_nr"][it_subsys_nr]["nb_steps"] / exploration_json["subsys_nr"][it_subsys_nr]["print_every_x_steps"])
    
    for it_nnp in range(1, exploration_json["nb_nnp"] + 1):
        for it_each in range(1, exploration_json["nb_traj"] + 1):

            local_apath = Path(".").resolve()/str(it_subsys_nr)/str(it_nnp)/(str(it_each).zfill(5))

            if not (local_apath/"skip").is_file():

                beads_index_json = cf.json_read(local_apath/"devi_index.json",False,False)

                beads_selection_index=[]
                random.seed(a=(it0_subsys_nr+it_nnp+it_each+random.randint(0,1000)))
                possible_beads = [zzz for zzz in range(0, nb_beads)]
                for f in range(0,int(nb_pos)):
                    if len(possible_beads) == 0:
                        possible_beads = [zzz for zzz in range(0,nb_beads)]
                    bead_idx = random.randrange(0,len(possible_beads))
                    beads_selection_index.append(possible_beads[bead_idx])
                    del possible_beads[bead_idx]
                del bead_idx, possible_beads

                for f in range(0, nb_beads):
                    beads_index_json[f] = [i for i,val in enumerate(beads_selection_index) if val==f]
                    beads_idx = np.array(beads_index_json[f])
                    beads_idx = map(str,  beads_idx.astype(int))
                    beads_idx = [ zzz + "\n" for zzz in beads_idx]
                    cf.write_file(local_apath / ("beads_"+str(f)+".vmd"), beads_idx)

                del beads_idx, beads_selection_index
                
                for f in range(0, nb_beads):
                    traj_file = local_apath/(str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".beads_"+str(f)+".xyz")
                    vmd_tcl = master_vmd_tcl.copy()
                    vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_PDB_FILE_",str(topo_file))
                    vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_XYZ_FILE_",str(traj_file))
                    vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_SELECTION_FILE_",str(local_apath/ ("beads_"+str(f)+".vmd")))
                    vmd_tcl = cf.replace_in_list(vmd_tcl,"_R_XYZ_OUT_",str(local_apath/("vmd_${j}.xyz")))
                    cf.write_file((local_apath/"vmd.tcl"),vmd_tcl)
                    del vmd_tcl, traj_file
                    subprocess.call([vmd_bin,"-e",str((local_apath/"vmd.tcl")),"-dispdev", "text"],\
                        stdout=subprocess.DEVNULL,\
                        stderr=subprocess.STDOUT)
                    cf.remove_file((local_apath/"vmd.tcl"))
                    cf.remove_file((local_apath/("beads_"+str(f)+".vmd")))

                cf.remove_file(local_apath/("beads_rerun_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".xyz"))
                ### #TODO: Not Path friendly / Replace with either subprocess call or read python
                os.system("cat "+str(local_apath)+"/vmd_*.xyz >> "+str(local_apath)+"/beads_rerun_"+str(it_subsys_nr)+"_"+str(it_nnp)+"_"+current_iteration_zfill+".xyz")
                cf.remove_file_glob(local_apath,"vmd_*.xyz")
                cf.json_dump(beads_index_json,local_apath/"beads_index_json")
