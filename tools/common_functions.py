import os
import sys
import json
import numpy as np
from pathlib import Path
import socket
import logging
import xml.etree.ElementTree as ET


def json_read(file_path: Path,abort: bool=True,is_logged: bool=False) -> dict:
    """Read a JSON file to a JSON dict

    Args:
        file_path (Path): Path object to the file
        abort (bool, optional): True to abort, False create a new dict. Defaults to True.
        is_logged (bool, optional): Logging. Defaults to False.

    Returns:
        dict: JSON dictionary
    """
    if file_path.is_file():
        if is_logged:
            logging.info("Loading: "+file_path.name+" from "+str(file_path.parent))
        return json.load(file_path.open())
    else:
        if abort:
            logging.critical("File not found: "+file_path.name+" not in "+str(file_path.parent))
            logging.critical("Aborting...")
            sys.exit(1)
        else:
            if is_logged:
                logging.info("File not found: "+file_path.name)
                logging.info("Creating a new one...")
            return {}


def json_dump(json_dict: dict,file_path: Path,is_logged: bool=False):
    """Write a JSON dict to a JSON file

    Args:
        json_dict (dict): JSON dictionary
        file_path (Path): Path object to the file
        is_logged (bool, optional): Logging. Defaults to False.
    """
    with file_path.open("w", encoding="UTF-8") as f:
        json.dump(json_dict, f)
        if is_logged:
            logging.info("Writing "+file_path.name+" in "+str(file_path.parent))


def check_file(file_path: Path,exists: bool,abort: bool,error_msg: str="default"):
    """Check if a file exists or not, abort or not
        exists/abort:
        True/True: if the file does't exist, abort
        False/True: if the file does exist, abort
        True/False: if the file does't exist, log only
        False/False: if the file does exist, log only

    Args:
        file_path (Path): Path object to the file
        exists (bool):  True to check if it should exists, False to check if it shouldn't
        abort (bool): True to abort, False to log only
        error_msg (str, optional): To override default error message. Defaults to "default".
    """
    if not exists and not file_path.is_file():
        if abort:
            logging.critical("File not found: "+file_path.name+" not in "+str(file_path.parent)) if error_msg == "default" else logging.critical(error_msg)
            logging.critical("Aborting...")
            sys.exit(1)
        else:
            logging.warning("File not found: "+file_path.name+" not in "+str(file_path.parent)) if error_msg == "default" else logging.warning(error_msg)
    elif not exists and file_path.is_file():
        if abort:
            logging.critical("File found: "+file_path.name+" in "+str(file_path.parent)) if error_msg == "default" else logging.critical(error_msg)
            logging.critical("Aborting...")
            sys.exit(1)
        else:
            logging.warning("File found: "+file_path.name+" in "+str(file_path.parent))if error_msg == "default" else logging.warning(error_msg)


def check_dir(directory_path: Path, abort: bool,error_msg: str="default"):
    """Check if directory exists

    Args:
        directory_path (Path): Path object to the directory
        abort (bool): True to abort, False to log only
        error_msg (str, optional): To override default error message. Defaults to "default".
    """
    if not directory_path.is_dir():
        if abort:
            if error_msg == "data":
                logging.critical("No data folder to search for initial sets: "+str(directory_path))
                logging.critical("Aborting...")
                sys.exit(1)
            else:
                logging.critical("Directory not found: "+str(directory_path)) if error_msg == "default" else logging.critical(error_msg)
                logging.critical("Aborting...")
                sys.exit(1)
        else:
            logging.warning("Directory not found: "+str(directory_path)) if error_msg == "default" else logging.critical(error_msg)


def check_initial_datasets(training_iterative_apath: Path) -> dict:
    """Check the initial datasets

    Args:
        training_iterative_apath (Path): Path object to the root training folder

    Returns:
        dict: JSON dictionary
    """
    if (training_iterative_apath/"control"/"initial_datasets.json").is_file():
        logging.info("Loading: "+str((training_iterative_apath/"control"/"initial_datasets.json")))
        initial_datasets_json = json.load((training_iterative_apath/"control"/"initial_datasets.json").open())
        for f in initial_datasets_json:
            if not (training_iterative_apath/"data"/f).is_dir():
                logging.critical("Initial set not found in data: "+f)
                logging.critical("Aborting...")
                sys.exit(1)
            else:
                if np.load(str(training_iterative_apath/"data"/f/"set.000"/"box.npy")).shape[0] != initial_datasets_json[f]:
                    logging.critical("Missmatch in count for the set: "+f)
                    logging.critical("Aborting...")
                    sys.exit(1)
        return initial_datasets_json
    else:
        logging.critical("datasets_initial.json not present in: "+str(training_iterative_apath/"control"))
        logging.critical("Aborting...")
        sys.exit(1)


def read_file(file_path: Path) -> list:
    """Read a file as a list of strings (one line, one string)

    Args:
        file_path (Path): Path object to the file

    Returns:
        list: list of strings
    """
    if not file_path.is_file():
        logging.critical("File not found: "+file_path.name+" not in "+str(file_path.parent))
        logging.critical("Aborting...")
        sys.exit(1)
    else:
        with file_path.open() as f:
            return f.readlines()

### XML maninulations
def read_xml(file_path: Path) -> ET.ElementTree:
    """_summary_

    Args:
        file_path (Path): _description_

    Returns:
        ET.ElementTree: _description_
    """
    with open(file_path) as zzz:
        xml_tree = ET.parse(zzz)
        return xml_tree


def convert_xml_to_listofstrings(xml_tree: ET.ElementTree) -> list:
    """_summary_

    Args:
        xml_tree (ET.ElementTree): _description_

    Returns:
        list: _description_
    """
    xml_string = ET.tostring(xml_tree.getroot(), encoding='unicode', method='xml')
    list_string = []
    current_string=""
    for zzz in xml_string:
        if "\n" not in zzz:
            current_string = current_string + zzz
        else:
            current_string = current_string + zzz
            list_string.append(current_string)
            current_string=""
    list_string.append(current_string)
    return list_string


def convert_listofstrings_to_xml(list_string: list) -> ET.ElementTree:
    lst = []
    for zzz in list_string:
        for yyy in zzz:
            lst.append(yyy)
    string="".join(lst)
    return ET.ElementTree(ET.fromstring(string))


def write_xml(xml_tree: ET.ElementTree, file_path: Path):
    """_summary_

    Args:
        xml_tree (ET.ElementTree): _description_
        file_path (Path): _description_
    """
    xml_tree.write(file_path)

def get_temp_from_xml_tree(xml_tree):
    temp = -1
    for state in xml_tree.iter():
        try:
            temp = state.find('temperature').text
        except:
            pass
    return temp


### Cluster
def check_cluster() -> str:
    """Get the cluster name

    Returns:
        str: The short string name for the cluster: jz, oc, pc, ir
    """
    if socket.gethostname().find(".")>=0:
        cluster_name = socket.gethostname()
    else:
        cluster_name = socket.gethostbyaddr(socket.gethostname())[0]
    if ("jean-zay" or "idris.fr") in cluster_name:
        logging.info("Cluster found: Jean Zay")
        return "jz"
    elif ("occigen" ) in cluster_name:
        logging.info("Cluster found: Occigen")
        return "oc"
    elif ("debye.net") in cluster_name:
        logging.info("Cluster found: Paracelsus")
        return "pc"
    elif ("irene") in cluster_name:
        logging.info("Cluster found: Irene Rome")
        return "ir"
    else:
        logging.warning("Not on a known cluster, some features will not work.")
        return "0"


def clusterize(deepmd_iterative_apath: Path,training_iterative_apath: Path,step: str, cluster: str=None, user_keyword=None):
    """_summary_

    Args:
        deepmd_iterative_apath (Path): _description_
        training_iterative_apath (Path): _description_
        step (str): _description_
        cluster (str, optional): _description_. Defaults to None.
        user_keyword (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if cluster is None:
        cluster = check_cluster()
    if (training_iterative_apath / "inputs" / "machine_file.json").is_file():
        machine_file = json_read(training_iterative_apath / "inputs" / "machine_file.json")
    else:
        machine_file = json_read(deepmd_iterative_apath / "tools" / "machine_file.json")
    if user_keyword is None:
        for zzz in machine_file[cluster].keys():
            if "default" in machine_file[cluster][zzz].keys():
                for yyy in machine_file[cluster][zzz]["default"]:
                    if step in yyy:
                        return cluster, machine_file[cluster][zzz], 0
                    else:
                        True
        return "", [], 1
    elif type(user_keyword) == list and len(user_keyword) == 3:
        for zzz in machine_file[cluster].keys():
            if (
                machine_file[cluster][zzz]["project_name"] == user_keyword[0]
            ) and (
                machine_file[cluster][zzz]["allocation_name"] == user_keyword[1]
            ) and (
                machine_file[cluster][zzz]["arch_name"] == user_keyword[2]
            ) and (
                step in machine_file[cluster][zzz]["valid_for"]
            ):
                return cluster, machine_file[cluster][zzz], 0
        return "", [], 5
    elif type(user_keyword) == str:
        if (
            user_keyword in machine_file[cluster].keys()
        ) and (
            step in machine_file[cluster][user_keyword]["valid_for"]
        ):
            return cluster, machine_file[cluster][user_keyword], 0
        else:
            return "", [], 4
    else:
        return "", [], 3


def check_same_cluster(cluster: str, _json: dict):
    """_summary_

    Args:
        cluster (str): _description_
        _json (dict): _description_
    """
    if _json["cluster"] != cluster:
        logging.critical("Different cluster ("+str(cluster)+") than the one for prep ("+str(_json["cluster"])+")")
        logging.critical("Aborting...")
        sys.exit(1)


### LMP Files
def get_cell_nbatoms_from_lmp(subsys_lammps_data: list):
    """_summary_

    Args:
        subsys_lammps_data (list): _description_

    Returns:
        _type_: _description_
    """
    dim_string = ["xlo xhi", "ylo yhi", "zlo zhi"]
    subsys_cell=[]
    for n,string in enumerate(dim_string):
        temp = [zzz for zzz in subsys_lammps_data if string in zzz]
        temp = replace_in_list(temp,"\n","")
        temp = [zzz for zzz in temp[0].split(" ") if zzz]
        subsys_cell.append(float(temp[1]) - float(temp[0]))
    del n, string
    temp = [zzz for zzz in subsys_lammps_data if "atoms" in zzz]
    temp = replace_in_list(temp,"\n","")
    temp = [zzz for zzz in temp[0].split(" ") if zzz]
    subsys_nb_atm = int(temp[0])
    return subsys_cell, subsys_nb_atm


### Generic files (as list of string)
def write_file(file_path: Path, list_of_string: list):
    """_summary_

    Args:
        file_path (Path): _description_
        list_of_string (list): _description_
    """
    file_path.write_text("".join(list_of_string))


def replace_in_list(input_list: list,substring_in: str,substring_out: str) -> list:
    """_summary_

    Args:
        input_list (list): input list of strings
        substring_in (str): string to replace
        substring_out (str): new string

    Returns:
        list: output list of strings
    """
    output_list = [f.replace(substring_in,substring_out) for f in input_list]
    return output_list


def delete_in_list(input_list: list, substring_in: str) -> list:
    """_summary_

    Args:
        input_list (list): input list of strings
        substring_in (str): substring to look for and delete the whole string

    Returns:
        list: output list of strings
    """
    output_list = [zzz for zzz in input_list if not substring_in in zzz]
    return output_list

def remove_file(file_path: Path):
    """_summary_

    Args:
        file_path (Path): _description_
    """
    if file_path.is_file():
        file_path.unlink()


def remove_file_glob(directory_path: Path,file_glob: str):
    """_summary_

    Args:
        directory_path (Path): _description_
        file_glob (str): _description_
    """
    for f in directory_path.glob(file_glob):
        f.unlink()


def remove_tree(directory_path: Path):
    """_summary_

    Args:
        pth (Path): _description_
    """
    for child in directory_path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            remove_tree(child)
    directory_path.rmdir()



def change_dir(directory_path: Path):
    """_summary_

    Args:
        directory_path (Path): Path to the new directory
    """
    if not directory_path.is_dir():
        logging.critical("Directory not found: "+str(directory_path))
        logging.critical("Aborting...")
        sys.exit(1)
    else:
        try:
            os.chdir(directory_path)
        except:
            logging.critical("Error in changing dir: "+str(directory_path))
            logging.critical("Aborting...")
            sys.exit(1)


def seconds_to_walltime(seconds: float) -> str:
    """_summary_

    Args:
        seconds (float): Float in seconds

    Returns:
        str: A string in 00000:00:00 format
    """
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%d:%02d:%02d" % (hour, min, sec)


#### Training only
def get_decay_steps(total_trained: int,min: int=5000) -> int:
    """Calculate decay steps if not defined:
        Floor the number of total trained structures to nearest 10000
        If < 20 000, decay_steps = 5 000
        [20 000 and 100000[, decay_steps = total / 4
        If >= 100 000, decay_steps = 20 0000 + 5 000 per 50 000 increments (100 000 = 20 000 + 5 000, 200 000 = 20 000 + 5 000 + 10 000)

    Args:
        total_trained (int): Number of otal structures to train
        min (int, optional): Minimum decay steps. Defaults to 5000.

    Returns:
        decay_steps (int): decay steps (tau)
    """
    power_val = np.power(10,np.int64(np.log10(total_trained)))
    total_trained_floored = int(np.floor(total_trained/power_val)*power_val)
    if total_trained_floored < 20000:
        decay_steps = min
    elif total_trained_floored < 100000:
        decay_steps = total_trained_floored // 4
    else:
         decay_steps = 20000 + (( total_trained_floored - 50000 )/100000)*10000
    return int(decay_steps)


def get_decay_rate(stop_batch :int,start_lr: float,stop_lr: float,decay_steps: int) -> float:
    """Get the decay rate (lambda)

    Args:
        stop_batch (int): final training step (tfinal)
        start_lr (float): starting learning rate (alpha0)
        stop_lr (float): ending learning rate (alpha(tfinal))
        decay_steps (int): decay steps (tau)

    Returns:
        (float): decay_rate (lambda)
    """
    return np.exp(np.log(stop_lr / start_lr) / (stop_batch /  decay_steps))


def get_learning_rate(training_step: int,start_lr: float,decay_rate: float,decay_steps: int) -> float:
    """Get the learning rate at step t

    Args:
        training_step (int): training step (t)
        start_lr (float): starting learning rate (alpha0)
        decay_rate (float): decay rate (lambda)
        decay_steps (int): decay steps (tau)

    Returns:
        (float): learning rate (alpha(t)) at step t
    """
    return (start_lr*decay_rate**(training_step/decay_steps))




### Trash ?
def check_if_in_dict(params_f,key_f,default_f,error_f):
    """
    """
    if (key_f in params_f.keys()):
        return params_f[key_f]
    else:
        if error_f == 1:
            return default_f
        else:
            sys.exit("Error: "+key_f+" not found.\n Aborting...")


def import_xyz(file_path: Path):
    if file_path.is_file() is False:
        sys.exit("File not found: "+file_path+"\nAborting...")
    xyz = file_path.open("r")
    atoms = []
    coordinates = []
    n_atom = []
    step_atoms = []
    step_coordinates = []
    blank = []
    c = 0
    s = 0
    line = xyz.readline()
    while line != "":
        n_atom_iter = int(line)
        n_atom.append(n_atom_iter)
        blank.append(xyz.readline())
        for i in range(n_atom_iter):
            line =  xyz.readline()
            atom, x, y, z = line.split()
            atoms.append(atom)
            coordinates.append([float(x), float(y), float(z)])
            c = c + 1
            if c >= n_atom[s]:
                step_atoms.append(atoms)
                step_coordinates.append(coordinates)
                atoms=[]
                coordinates=[]
                c = 0
                s = s + 1
        line = xyz.readline()
    n_atom = np.array(n_atom)
    step_atoms = np.array(step_atoms)
    step_coordinates = np.array(step_coordinates)
    xyz.close()
    return n_atom,step_atoms,step_coordinates,blank


def write_xyz_from_index(file_path: Path,select_nb: int,n_atom :np.ndarray,step_atoms: np.ndarray,step_coordinates: np.ndarray,blank: list):
    xyz = file_path.open("w")
    xyz.write(str(n_atom[select_nb])+"\n")
    xyz.write(str(blank[select_nb]))
    x = 0
    while x < n_atom[select_nb]:
        xyz.write(str(step_atoms[select_nb,x])+" "+str(step_coordinates[select_nb,x,0])+" "+str(step_coordinates[select_nb,x,1])+" "+str(step_coordinates[select_nb,x,2])+"\n")
        x =  x + 1
    xyz.close()

