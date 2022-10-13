 ## deepmd_iterative_apath
# deepmd_iterative_apath = ''

###################################### No change past here
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')

import subprocess

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

cf.remove_file_glob('.','DeepMD_*')
cf.remove_file_glob('.','*.npz')
cf.remove_file_glob('.','*.pb')
cf.remove_file_glob('.','*.sh')
cf.remove_file_glob('.','_*.py')
for it_data_folders in Path('.').iterdir():
    if it_data_folders.is_dir():
        if 'out' in str(it_data_folders.name) or 'log' in str(it_data_folders.name):
            cf.remove_file_glob(str(it_data_folders.name)+'/','*')
            it_data_folders.rmdir()

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()