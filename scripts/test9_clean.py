#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Very fancy DeepMD-based semi-automatic highly-customizable iterative training procedure. #
#   Copyright 2022-2023 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#

## deepmd_iterative_apath
# deepmd_iterative_apath: str = ""

###################################### No change past here
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

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

current_apath = Path(".").resolve()
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]

logging.info("Deleting SLURM out/error files...")
cf.remove_file_glob(current_apath,"DeepMD_*")
logging.info("Deleting NPZ files...")
cf.remove_file_glob(current_apath,"*.npz")
logging.info("Deleting NNP PB files...")
cf.remove_file_glob(current_apath,"*.pb")
logging.info("Deleting SLURM launch files...")
cf.remove_file_glob(current_apath,"*.sh")
logging.info("Deleting Python helper files...")
cf.remove_file_glob(current_apath,"_*.py")
logging.info("Deleting DP-Test out/log files...")
for it_data_folders in current_apath.iterdir():
    if it_data_folders.is_dir():
        if "out" in str(it_data_folders.name) or "log" in str(it_data_folders.name):
            cf.remove_file_glob(it_data_folders,"*")
            it_data_folders.rmdir()
del it_data_folders
logging.info("Cleaning done!")
logging.info("If you are done with any testing you can safely delete the "+current_iteration_zfill+"-test folder")

del deepmd_iterative_apath, training_iterative_apath, current_apath, current_iteration_zfill

del sys, Path, logging, cf
import gc; gc.collect(); del gc
exit()