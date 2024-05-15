"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15

This module contains functions for checking the availability of certain commands on the system, as well as a function for validating the current working directory during the execution of a specific step.

Functions
---------
check_atomsk(atomsk_path: str = None) -> str
    Check if the Atomsk command is available on the system.
check_vmd(vmd_path: str = None) -> str
    Check if the VMD command is available on the system.
validate_step_folder(step_name: str) -> None
    Check if the current directory matches the expected directory for the given step.
"""

# TODO: Homogenize the docstrings for this module

# Standard library modules
import logging
import os
import re
import subprocess
import shutil
from pathlib import Path

# Local imports
from arcann_training.common.utils import catch_errors_decorator


# Unittested
@catch_errors_decorator
def check_atomsk(atomsk_path: str = None) -> str:
    """
    Check if the Atomsk command is available on the system.

    This function first checks if 'atomsk_path' is provided and is a valid path.
    If it is, it returns the path to 'atomsk'. If 'atomsk_path' is not valid, it logs a warning and continues to the next step.

    The next step checks if the 'ATOMSK_PATH' environment variable is defined and is a valid path.
    If it is, it returns the path to 'atomsk'.

    If neither 'atomsk_path' nor 'ATOMSK_PATH' is valid, the function tries to find the 'atomsk' command in the system path.
    If it is found, it returns the full path to 'atomsk'. If 'atomsk' is not found, the function logs a critical error and exits the program.

    Parameters
    ----------
    atomsk_path : str
        The path to the 'atomsk' command, if it is not in the system path.

    Returns
    -------
    str
        The full path to the 'atomsk' command.

    Raises
    ------
    FileNotFoundError
        If the 'atomsk' command is not found in the system path.
    """
    logger = logging.getLogger("ArcaNN")

    # Check if atomsk_path is provided and is valid
    if atomsk_path is not None:
        if Path(atomsk_path).is_file():
            return str(Path(atomsk_path).resolve())
        else:
            logger.warning(f"Atomsk path '{atomsk_path}' is invalid. Deleting the atomsk_path variable. Checking environment variable and system path...")
            del atomsk_path

    # Check if ATOMSK_PATH is defined and is valid
    atomsk_path = os.environ.get("ATOMSK_PATH")
    if atomsk_path is not None and atomsk_path != "" and Path(atomsk_path).is_file():
        logger.info(f"Atomsk found in $ATOMSK_PATH.")
        return str(Path(atomsk_path).resolve())

    # Check if atomsk is available in system path
    atomsk = shutil.which("atomsk")
    if atomsk is not None:
        logger.info(f"Atomsk found in $PATH.")
        atomsk_path = Path(atomsk).resolve()
        return f"{atomsk_path}"
    else:
        error_msg = f"Atomsk not found. Please add it to your system path (or use ATOMSK_PATH environment variable to point towards the atomsk binary)."
        raise FileNotFoundError(error_msg)


# Unittested
@catch_errors_decorator
def check_vmd(vmd_path: str = None) -> str:
    """
    Check if the VMD command is available on the system.

    This function first checks if 'vmd_path' is provided and is a valid path.
    If it is, it returns the path to 'vmd'. If 'vmd_path' is not valid, it logs a warning and continues to the next step.

    The next step checks if the 'VMD_PATH' environment variable is defined and is a valid path.
    If it is, it returns the path to 'vmd'.

    If neither 'vmd_path' nor 'VMD_PATH' is valid, the function tries to find the 'vmd' command in the system path.
    If it is found, it returns the full path to 'vmd'. If 'vmd' is not found, the function logs a critical error and exits the program.

    Parameters
    ----------
    vmd_path : str
        The path to the 'vmd' command, if it is not in the system path.

    Returns
    -------
    str
        The full path to the 'vmd' command.

    Raises
    ------
    FileNotFoundError
        If the 'vmd' command is not found in the system path.
    """
    logger = logging.getLogger("ArcaNN")

    # Check if vmd_path is provided and is valid
    if vmd_path is not None and vmd_path != "" and Path(vmd_path).is_file():
        return str(Path(vmd_path).resolve())
    else:
        logger.warning(f"VMD path '{vmd_path}' is invalid. Deleting the vmd_path variable. Checking environment variable and system path...")
        del vmd_path

    # Check if VMD_PATH is defined and is valid
    vmd_path = os.environ.get("VMD_PATH")
    if vmd_path is not None and vmd_path != "" and Path(vmd_path).is_file():
        logger.info(f"Atomsk found in $VMD_PATH.")
        return str(Path(vmd_path).resolve())

    # Check if vmd is available in system path
    vmd = shutil.which("vmd")
    if vmd is not None:
        logger.info(f"VMD found in $PATH.")
        vmd_path = Path(vmd).resolve()
        return f"{vmd_path}"
    else:
        error_msg = f"VMD not found in $PATH. Please add it to your system path (or use VMD_PATH environment variable to point towards the vmd binary)."
        raise FileNotFoundError(error_msg)


# Unittested
@catch_errors_decorator
def validate_step_folder(step_name: str) -> None:
    """
    Check if the current directory matches the expected directory for the step.

    Parameters
    ----------
    step_name :str
        The name of the step being executed.

    Returns
    -------
    None

    Raises
    -------
    ValueError
        If the current directory name does not contain the step name.
    """
    # Get the path of the current directory
    current_directory = Path(".").resolve()

    # Check if the current directory name contains the step name
    if step_name not in current_directory.name:
        error_msg = f"The current directory '{current_directory}' does not match the expected directory for the '{step_name}' step."
        raise ValueError(error_msg)


# TODO: Add tests for this function
@catch_errors_decorator
def check_dcd_is_valid(dcd_path: Path, vmd_bin: Path) -> bool:
    """
    Check if the dcd file is valid.

    Returns
    -------
    bool
        True if the dcd file is valid, False otherwise.
    """
    vmd_script = f"""
    # Load your trajectory file
    mol addfile {dcd_path} type dcd waitfor all

    # Get the number of frames
    set numFrames [molinfo top get numframes]

    # Output the number of frames
    puts "Number of frames: $numFrames"

    # Quit VMD
    quit
    """
    # Run VMD script from command line
    result = subprocess.run([vmd_bin, "-dispdev", "text", "-e", "/dev/stdin"], input=vmd_script, text=True, capture_output=True)

    # Check if the output contains "Unable to load file "
    if "Unable to load file " in result.stdout:
        return False
    else:
        match = re.search(r"Number of frames: (\d+)", result.stdout)
        if match:
            return True
        else:
            return False


# TODO: Add tests for this function
@catch_errors_decorator
def check_nc_is_valid(nc_path: Path, vmd_bin: Path) -> bool:
    """
    Check if the nc file is valid.

    Returns
    -------
    bool
        True if the nc file is valid, False otherwise.
    """
    vmd_script = f"""
    # Load your trajectory file
    mol addfile {nc_path} type netcdf waitfor all

    # Get the number of frames
    set numFrames [molinfo top get numframes]

    # Output the number of frames
    puts "Number of frames: $numFrames"

    # Quit VMD
    quit
    """
    # Run VMD script from command line
    result = subprocess.run([vmd_bin, "-dispdev", "text", "-e", "/dev/stdin"], input=vmd_script, text=True, capture_output=True)

    # Check if the output contains "Unable to load file "
    if "Unable to load file " in result.stdout:
        return False
    else:
        match = re.search(r"Number of frames: (\d+)", result.stdout)
        if match:
            return True
        else:
            return False
