import os
import sys
import json
import numpy as np
from pathlib import Path
import socket
import logging

#### TODO Better descriptions

def check_file(file_path:str,status:int,abort:bool,error_msg='None'):
    """Check if a file exists or not, abort or not

    Args:
        file_path (str): full path of the file
        status (int): 0 to check if it is present, 1 to check if it isn't
        abort (bool): True to abort, False to log only
        error_msg (str, optional): To override default error message.
    """
    if status == 0 and not Path(file_path).is_file():
        if abort:
            logging.critical(file_path+' does not exist.') if error_msg == 'None' else logging.critical(error_msg)
            logging.critical('Aborting...')
            sys.exit(1)
        else:
            logging.warning(file_path+' does not exist.') if error_msg == 'None' else logging.warning(error_msg)
    elif status == 1 and Path(file_path).is_file():
        if abort:
            logging.critical(file_path+' exists.') if error_msg == 'None' else logging.critical(error_msg)
            logging.critical('Aborting...')
            sys.exit(1)
        else:
            logging.warning(file_path+' exists.') if error_msg == 'None' else logging.warning(error_msg)

def check_dir(directory_path:str,abort:bool,error_msg='None'):
    """Check if directory exists

    Args:
        directory_path (str):  full path of the directory to check
        abort (bool): True to abort, False to log only
        error_msg (str, optional): override default error message
    """
    if Path(directory_path).is_dir() is False:
        if abort:
            if error_msg == 'data':
                logging.critical('No data folder to search for initial sets: '+directory_path)
                logging.critical('Aborting...')
                sys.exit(1)
            elif error_msg == 'None':
                logging.critical('No existing folder: '+directory_path)
                logging.critical('Aborting...')
                sys.exit(1)
            else:
                logging.critical(error_msg)
                logging.critical('Aborting...')
                sys.exit(1)
        else:
            if error_msg == 'None':
                logging.warning('No existing folder: '+directory_path)
            else:
                logging.warning(error_msg)

def create_dir(directory_path:str):
    """Create directory

    Args:
        directory_path (str): full path of the directory to create
    """
    if Path(directory_path).is_dir() is False:
        try:
            Path(directory_path).mkdir(parents=True)
        except(FileExistsError):
            pass
        except:
            logging.critical('Could not create '+directory_path)
            logging.critical('Aborting...')
            sys.exit(1)

def change_dir(directory_path:str):
    if Path(directory_path).is_dir() is False:
        logging.critical('No existing folder: '+directory_path)
        logging.critical('Aborting...')
        sys.exit(1)
    else:
        try:
            os.chdir(directory_path)
        except:
            logging.critical('Error changing dir: '+directory_path)
            logging.critical('Aborting...')
            sys.exit(1)

def check_cluster():
    """_summary_

    Returns:
        str: The short string name for the cluster: jz, oc, pc, ir
    """
    if socket.gethostname().find('.')>=0:
        cluster_name = socket.gethostname()
    else:
        cluster_name = socket.gethostbyaddr(socket.gethostname())[0]
    if ('jean-zay' or 'idris.fr') in cluster_name:
        logging.info('Cluster found: Jean Zay')
        return 'jz'
    elif ('occigen' ) in cluster_name:
        logging.info('Cluster found: Occigen')
        return 'oc'
    elif ('debye.net') in cluster_name:
        logging.info('Cluster found: Paracelsus')
        return 'pc'
    elif ('irene') in cluster_name:
        logging.info('Cluster found: Irene Rome')
        return 'ir'
    else:
        logging.warning('Not on a known cluster, some features will not work.')
        return '0'

def get_decay_steps(total_trained:int,min=5000):
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
        decay_steps = decay_steps / 4
    else:
         decay_steps = 20000 + (( total_trained_floored - 50000 )/100000)*10000
    return int(decay_steps)

def get_decay_rate(stop_batch:int,start_lr:float,stop_lr:float,decay_steps:int):
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

def get_learning_rate(training_step:int,start_lr:float,decay_rate:float,decay_steps:int):
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

def json_dump(json_dict:dict,output_file_path:str,print_log=False,name=None):
    """Dump json file

    Args:
        json_dict (dict): JSON dictionary
        output_file_path (str): full path of the output json file
        print_log (bool): log the action
        name (str): JSON fancy name for logging
    """
    if name == 'None':
        name = json_dict
    with open(output_file_path, 'w') as f:
        json.dump(json_dict, f)
        if print_log:
            logging.info('Writing '+name+' in: '+output_file_path)

def json_read(json_file_path,should_abort=True,should_print=True):
    """Read json file

    Args:
        input_file (str): full path of json file
        abort (bool): if True then abort if the file is not found; else create empty dict

    Returns:
        (dict): json as a dict
    """
    if Path(json_file_path).is_file():
        if should_print:
            logging.info('Loading: '+json_file_path)
        return json.load(open(json_file_path,))
    else:
        if should_abort:
            logging.critical('Config not found: '+json_file_path)
            logging.critical('Aborting...')
            sys.exit(1)
        else:
            if should_print:
                logging.info('No '+json_file_path+' file found, creating a new one...')
            return {}

def write_file(output_file_path:str,list_of_string:list):

    """Write a file as a list of string (each string must end with \n for new lines)

    Args:
        output_file_path (str): full path of output file
        list_of_string (list): list of strings to write (one string per line)
    """
    with open(output_file_path,'w') as f:
         f.writelines(list_of_string)

def read_file(input_file_path:str):
    """_summary_

    Args:
        input_file (str): full path of input file

    Returns:
        list: the file as a list of strings (one string per line)
    """
    if Path(input_file_path).is_file() is False:
        logging.critical('File not found: '+input_file_path)
        logging.critical('Aborting...')
        sys.exit(1)
    with open(input_file_path) as f:
        OUT_output_list_strings = f.readlines()
    return OUT_output_list_strings

def replace_in_list(input_list,substring_in,substring_out):
    """_summary_

    Args:
        input_list (list): input list of string
        substring_in (str): string to replace
        substring_out (str): desired string

    Returns:
        list: output list of string
    """
    output_list = [f.replace(substring_in,substring_out) for f in input_list]
    return output_list

def check_if_in_dict(params_f,key_f,default_f,error_f):
    """
    """
    if (key_f in params_f.keys()):
        return params_f[key_f]
    else:
        if error_f == 1:
            return default_f
        else:
            sys.exit('Error: '+key_f+' not found.\n Aborting...')

def check_datasets_initial(current_path):
    """_summary_

    Args:
        current_path (str): current path

    Returns:
        dict: Initial set dict
    """
    if Path(current_path+'/control/datasets_initial.json').is_file():
        logging.info('Loading: '+current_path+'/control/datasets_initial.json')
        initial_sets_json_f = json.load(open(current_path+'/control/datasets_initial.json',))
        for f in initial_sets_json_f:
            if Path(current_path+'/data/'+f).is_dir() is False:
                logging.critical('Initial set not found in data: '+f)
                logging.critical('Aborting...')
                sys.exit(1)
            else:
                if np.load(current_path+'/data/'+f+'/set.000/box.npy').shape[0] != initial_sets_json_f[f]:
                    logging.critical('Missmatch in count for : '+f)
                    logging.critical('Aborting...')
                    sys.exit(1)
        return initial_sets_json_f
    else:
        logging.critical('datasets_initial.json not present in: '+current_path+'/control/')
        logging.critical('Aborting...')
        sys.exit(1)

def remove_file(file_path:str):
    """_summary_

    Args:
        file_path (_type_): _description_
    """
    if Path(file_path).is_file():
        Path(file_path).unlink()

def remove_file_glob(file_path:str,file_glob:str):
    """_summary_

    Args:
        file_path (str): _description_
        file_glob (str): _description_
    """
    for f in Path(file_path).glob(file_glob):
        f.unlink()

def remove_tree(pth:Path):
    """_summary_

    Args:
        pth (Path): _description_
    """
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            remove_tree(child)
    pth.rmdir()

def seconds_to_walltime(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return '%d:%02d:%02d' % (hour, min, sec)

def import_xyz(file_path:str):
    """_summary_

    Args:
        file_path (str): _description_

    Returns:
        _type_: _description_
    """
    if Path(file_path).is_file() is False:
        sys.exit('File not found: '+file_path+'\nAborting...')
    xyz = open(file_path, 'r')
    atoms = []
    coordinates = []
    n_atom = []
    step_atoms = []
    step_coordinates = []
    blank = []
    c = 0
    s = 0
    line = xyz.readline()
    while line != '':
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

def write_xyz_from_index(select_nb,xyz_out,n_atom,step_atoms,step_coordinates,blank):
    """
    """
    xyz2 = open(xyz_out, 'w')
    xyz2.write(str(n_atom[select_nb])+'\n')
    xyz2.write(str(blank[select_nb]))
    x = 0
    while x < n_atom[select_nb]:
        xyz2.write(str(step_atoms[select_nb,x])+' '+str(step_coordinates[select_nb,x,0])+' '+str(step_coordinates[select_nb,x,1])+' '+str(step_coordinates[select_nb,x,2])+'\n')
        x =  x+ 1