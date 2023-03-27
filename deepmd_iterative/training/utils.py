"""
Created: 2023/01/01
Last modified: 2023/03/27

The utils module provides functions for the training step.

Functions
---------
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
# Standard library modules
import json
from pathlib import Path
from typing import Dict

# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


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
        error_msg = "nb_structures must be a positive integer"
        raise ValueError(error_msg)

    # Check if min_decay_steps is a positive integer
    if not isinstance(min_decay_steps, int) or min_decay_steps <= 0:
        error_msg = "min_decay_steps must be a positive integer"
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
        error_msg = "start_lr must be a positive number."
        raise ValueError(error_msg)

    if decay_steps <= 0 or not isinstance(decay_steps, int):
        error_msg = "decay_steps must be a positive integer."
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
        error_msg = "All arguments must be positive."
        raise ValueError(error_msg)
    if not isinstance(decay_steps, int):
        error_msg = "decay_steps must be a positive integer"
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
        error_msg = f"The 'initial_datasets.json' file is missing from '{initial_datasets_info_file.parent}'."
        raise FileNotFoundError(error_msg)

    # Load the 'initial_datasets.json' file
    with initial_datasets_info_file.open() as file:
        initial_datasets_info = json.load(file)

    # Check each initial dataset
    for dataset_name, expected_num_samples in initial_datasets_info.items():
        dataset_path = training_dir / "data" / dataset_name

        # Check if the dataset exists in the 'data' subfolder
        if not dataset_path.is_dir():
            error_msg = f"Initial dataset '{dataset_name}' is missing from the 'data' subfolder."
            raise FileNotFoundError(error_msg)

        # Check if the number of samples in the dataset matches the expected count
        box_path = dataset_path / "set.000" / "box.npy"
        if not box_path.is_file():
            error_msg = f"No box.npy found in the dataset '{dataset_name}'."
            raise FileNotFoundError(error_msg)

        num_samples = len(np.load(str(box_path)))
        if num_samples != expected_num_samples:
            error_msg = f"Unexpected number of samples ({num_samples}) found in initial dataset '{dataset_name}'. Expected {expected_num_samples}."
            raise ValueError(error_msg)

    return initial_datasets_info


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
        If deepmd_model_version is not 2.0 or 2.1, or if deepmd_model_type_descriptor is not "se_e2_a".
        If the configuration is not valid with respect to machine/arch_name/arch and DeePMD.
    """
    # Check DeePMD version
    if training_config["deepmd_model_version"] not in [2.0, 2.1]:
        error_msg = f"Invalid deepmd model version (2.0 or 2.1): {training_config['deepmd_model_version']}."
        raise ValueError(error_msg)

    # Check DeePMD descriptor type
    if training_config["deepmd_model_type_descriptor"] not in ["se_e2_a"]:
        error_msg = f"Invalid deepmd type descriptor (se_e2_a): {training_config['deepmd_model_type_descriptor']}."
        raise ValueError(error_msg)

    # Check mismatch between machine/arch_name/arch and DeePMD
    if training_config["deepmd_model_version"] < 2.0:
        error_msg = "Only version >= 2.0 on Jean Zay!"
        raise ValueError(error_msg)
    if (
        training_config["deepmd_model_version"] < 2.1
        and training_config["arch_name"] == "a100"
    ):
        error_msg = "Only version >= 2.1 on Jean Zay A100!"
        raise ValueError(error_msg)