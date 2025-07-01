"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15

The utils module provides functions for the training step.

Functions
---------
generate_training_json(user_input_json: Dict, previous_json: Dict, default_input_json: Dict, merged_input_json: Dict) -> Tuple[Dict, Dict]:
    A function to generate the training JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

calculate_decay_steps(num_structures: int, min_decay_steps: int = 5000) -> int
    A function to calculate the number of decay steps for a given number of structures to train.

calculate_decay_rate(stop_batch: int, start_lr: float, stop_lr: float, decay_steps: int) -> float
    A function to calculate the decay rate based on the given training parameters.

calculate_learning_rate(current_step: int, start_lr: float, decay_rate: float, decay_steps: int) -> float
    A function to calculate the learning rate at a given training step, based on the given parameters.

check_initial_datasets(training_dir: Path) -> Dict[str, int]
    A function to check if the initial datasets exist and are properly formatted.

validate_deepmd_config(training_config) -> None
    A function to validate the provided training configuration for a DeePMD model.

"""

# TODO: Homogenize the docstrings for this module

# Standard library modules
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple
import json

# Third-party modules
import numpy as np

# Local imports
from arcann_training.common.utils import catch_errors_decorator


# Unittested
@catch_errors_decorator
def generate_training_json(
    current_input_json: Dict,
    previous_json: Dict,
    default_input_json: Dict,
) -> Tuple[Dict, Dict]:
    """
    Generate the training JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.
    If the user input JSON is found to be invalid, an error will be raised and the script execution will be terminated.
    This function also updates the merged input JSON.

    Parameters
    ----------
    current_input_json : dict
        The input JSON provided by the user, containing user-defined parameters.
    previous_json : dict
        The JSON from the previous iteration.
    default_input_json : dict
        The default input JSON containing default parameters.

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple containing:
        - The generated training JSON.
        - The updated merged input JSON.

    Raises
    ------
    ValueError
        If a key is not found in any of the dictionaries.
    TypeError
        If a value has a different type compared to the corresponding default value.
    """

    # We copy the default to get the order of the keys constant.
    merged_input_json = deepcopy(default_input_json)
    training_json = deepcopy(default_input_json)

    # Update the training JSON with values from the current JSON, the previous training JSON, and the default JSON.
    for key in default_input_json:
        # Check if the key is present in any of the dictionaries, and set the value accordingly
        if key in current_input_json:
            if current_input_json[key] != "default":
                training_json[key] = current_input_json[key]
                merged_input_json[key] = current_input_json[key]
        elif key in previous_json:
            training_json[key] = previous_json[key]
            merged_input_json[key] = previous_json[key]

        if "user_machine_keyword" in key:
            continue

        # Check if the value has the correct type
        if not isinstance(training_json[key], type(default_input_json[key])):
            error_msg = f"Type mismatch: '{key}' has type '{type(training_json[key])}', but should have type '{type(default_input_json[key])}'."
            raise TypeError(error_msg)

    return training_json, merged_input_json


# Unittested
@catch_errors_decorator
def calculate_decay_steps(num_structures: int, min_decay_steps: int = 5000) -> int:
    """
    Calculate the number of decay steps for a given number of structures to train.

    Parameters
    ----------
    num_structures : int
        The total number of structures to train.
    min_decay_steps : int, optional
        The minimum number of decay steps. Default is 5000.

    Returns
    -------
    int
        The calculated number of decay steps.

    Raises
    ------
    ValueError
        If num_structures is not a positive integer, or if min_decay_steps is not a positive integer.
    """
    # Check if num_structures is a positive integer
    if not isinstance(num_structures, int) or num_structures <= 0:
        error_msg = f"The argument 'num_structures' must be a positive integer."
        raise ValueError(error_msg)

    # Check if min_decay_steps is a positive integer
    if not isinstance(min_decay_steps, int) or min_decay_steps <= 0:
        error_msg = f"The argument 'min_decay_steps' must be a positive integer."
        raise ValueError(error_msg)

    # Round down to the nearest multiple of 10000
    floored_num_structures = num_structures // 10000 * 10000

    # Calculate the decay steps based on the number of structures
    if floored_num_structures < 20000:
        decay_steps = min_decay_steps
    elif floored_num_structures < 100000:
        decay_steps = floored_num_structures // 4
    else:
        extra_decay_steps = (floored_num_structures - 50000) // 50000 * min_decay_steps
        decay_steps = 20000 + extra_decay_steps

    return decay_steps


# Unittested
@catch_errors_decorator
def calculate_decay_rate(
    stop_batch: int, start_lr: float, stop_lr: float, decay_steps: int
) -> float:
    """
    Calculate the decay rate based on the given training parameters.

    Parameters
    ----------
    stop_batch : int
        The final training step.
    start_lr : float
        The starting learning rate.
    stop_lr : float
        The ending learning rate.
    decay_steps : int
        The number of decay steps.

    Returns
    -------
    float
        The calculated decay rate.

    Raises
    ------
    ValueError
        If the start learning rate or the number of decay steps is not a positive number.
    """
    # Check that the start learning rate is a positive number.
    if start_lr <= 0 or not isinstance(start_lr, (int, float)):
        error_msg = f"The argument 'start_lr' must be a positive number."
        raise ValueError(error_msg)

    if decay_steps <= 0 or not isinstance(decay_steps, int):
        error_msg = f"The argument 'decay_steps' must be a positive integer."
        raise ValueError(error_msg)

    # Calculate the decay rate using the given training parameters
    decay_rate = np.exp(np.log(stop_lr / start_lr) / (stop_batch / decay_steps))

    return decay_rate


# Unittested
@catch_errors_decorator
def calculate_learning_rate(
    current_step: int, start_lr: float, decay_rate: float, decay_steps: int
) -> float:
    """
    Calculate the learning rate at a given training step, based on the given parameters.

    Parameters
    ----------
    current_step : int
        The current training step.
    start_lr : float
        The starting learning rate.
    decay_rate : float
        The decay rate.
    decay_steps : int
        The number of decay steps.

    Returns
    -------
    float
        The learning rate at the current training step.

    Raises
    ------
    ValueError
        If any of the arguments are not positive or if decay_steps is not an integer.
    """
    # Check that all arguments are positive
    if not all(
        arg > 0 for arg in (current_step, start_lr, decay_rate, decay_steps)
    ) or not all(
        isinstance(arg, (int, float))
        for arg in (current_step, start_lr, decay_rate, decay_steps)
    ):
        error_msg = f"All arguments must be positive."
        raise ValueError(error_msg)
    if not isinstance(decay_steps, int):
        error_msg = f"The argument 'decay_steps' must be a positive integer."
        raise ValueError(error_msg)

    # Calculate the learning rate based on the current training step
    step_ratio = current_step / decay_steps
    decay_factor = decay_rate**step_ratio
    learning_rate = start_lr * decay_factor

    return learning_rate


# Unittested
@catch_errors_decorator
def check_initial_datasets(training_dir: Path) -> Dict[str, int]:
    """
    Check if the initial datasets exist and are properly formatted.

    Parameters
    ----------
    training_dir : Path
        Path to the root training folder.

    Returns
    -------
    Dict[str, int]
        A dictionary containing the name of each initial dataset and the expected number of samples in each.

    Raises
    ------
    FileNotFoundError
        If the 'initial_datasets.json' file is not found in the 'control' subfolder, or if any of the initial datasets is missing from the 'data' subfolder.
    ValueError
        If the number of samples in any of the initial datasets does not match the expected count.
    """
    initial_datasets_info_file = training_dir / "control" / "initial_datasets.json"

    # Check if the 'initial_datasets.json' file exists
    if not initial_datasets_info_file.is_file():
        error_msg = f"The initial_datasets.json file is missing from '{initial_datasets_info_file.parent}'"
        raise FileNotFoundError(error_msg)

    # Load the 'initial_datasets.json' file
    with initial_datasets_info_file.open() as file:
        initial_datasets_info = json.load(file)

    # Check each initial dataset
    for dataset_name, expected_num_samples in initial_datasets_info.items():
        dataset_path = training_dir / "data" / dataset_name

        # Check if the dataset exists in the 'data' subfolder
        if not dataset_path.is_dir():
            error_msg = (
                f"Initial dataset '{dataset_name}' is missing from the data subfolder."
            )
            raise FileNotFoundError(error_msg)

        # Check if the number of samples in the dataset matches the expected count
        box_path = dataset_path / "set.000" / "box.npy"
        if not box_path.is_file():
            error_msg = f"No 'box.npy' found in the dataset '{dataset_name}'."
            raise FileNotFoundError(error_msg)

        num_samples = len(np.load(str(box_path)))
        if num_samples != expected_num_samples:
            error_msg = f"Unexpected number of samples ('{num_samples}') found in initial dataset '{dataset_name}'. Expected:'{expected_num_samples}'."
            raise ValueError(error_msg)

    return initial_datasets_info


# Unittested
@catch_errors_decorator
def validate_deepmd_config(training_config) -> None:
    """
    Validate the provided training configuration for a DeePMD model.

    Parameters
    ----------
    training_config : dict
        A dictionary containing the training configuration.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If deepmd_model_version is not 2.x or 3.0, or if deepmd_model_type_descriptor is not "se_e2_a".
        If the configuration is not valid with respect to machine/arch_name/arch and DeePMD.
    """
    # Check DeePMD version
    if (
        float(training_config["deepmd_model_version"]) < 2.0
        or float(training_config["deepmd_model_version"]) > 3.0
    ):
        error_msg = f"Only 2.x and 3.0 versions of deepmd are suppported: '{training_config['deepmd_model_version']}'."
        raise ValueError(error_msg)
