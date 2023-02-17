from pathlib import Path
import logging
import sys
import json
from typing import Dict

# Non-standard imports
import numpy as np


def calculate_decay_steps(num_structures: int, min_decay_steps: int = 5000) -> int:
    """
    Calculate the number of decay steps (tau) for a given number of structures to train (N).

    Args:
        num_structures (int): The total number of structures to train (N).
        min_decay_steps (int, optional): The minimum number of decay steps (tau_min). Default is 5000.

    Returns:
        int: The calculated number of decay steps (tau).

    Raises:
        ValueError: If num_structures is not a positive integer, or if min_decay_steps is not a positive integer.
    """

    # Check if num_structures is a positive integer
    if not isinstance(num_structures, int) or num_structures <= 0:
        error_msg = "nb_structures must be a positive integer"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)
        # raise ValueError(error_msg)

    # Check if min_decay_steps is a positive integer
    if not isinstance(min_decay_steps, int) or min_decay_steps <= 0:
        error_msg = "min_decay_steps must be a positive integer"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)
        # raise ValueError(error_msg)

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


def calculate_decay_rate(
    stop_batch: int, start_lr: float, stop_lr: float, decay_steps: int
) -> float:
    """
    Calculate the decay rate (lambda) based on the given training parameters.

    Args:
        stop_batch (int): The final training step (tfinal).
        start_lr (float): The starting learning rate (alpha0).
        stop_lr (float): The ending learning rate (alpha(tfinal)).
        decay_steps (int): The number of decay steps (tau).

    Returns:
        float: The calculated decay rate (lambda).

    Raises:
        ValueError: The start learning rate must be a positive number.

    """
    # Check that the start learning rate is a positive number.
    if start_lr <= 0:
        error_msg = "The start learning rate must be a positive number."
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)
        # raise ValueError(error_msg)

    # Calculate the decay rate using the given training parameters
    decay_rate = np.exp(np.log(stop_lr / start_lr) / (stop_batch / decay_steps))

    return decay_rate


def calculate_learning_rate(
    current_step: int, start_lr: float, decay_rate: float, decay_steps: int
) -> float:
    """
    Calculate the learning rate (alpha) at a given training step (t), based on the given parameters.

    Args:
        training_step (int): The current training step (t).
        start_lr (float): The starting learning rate (alpha0).
        decay_rate (float): The decay rate (labmda).
        decay_steps (int): The number of decay steps (tau).

    Returns:
        float: The learning rate at the current training step.

    Raises:
        ValueError: If any of the arguments are not positive
    """

    # Check that all arguments are positive
    if not all(arg > 0 for arg in (current_step, start_lr, decay_rate, decay_steps)):
        error_msg = "All arguments must be positive."
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)
        # raise ValueError(error_msg)

    # Calculate the learning rate based on the current training step
    step_ratio = current_step / decay_steps
    decay_factor = decay_rate**step_ratio
    learning_rate = start_lr * decay_factor

    return learning_rate


def check_initial_datasets(training_dir: Path) -> Dict[str, int]:
    """
    Check if the initial datasets exist and are properly formatted.

    Args:
        training_dir  (Path): Path to the root training folder.

    Returns:
        Dict[str, int]: A dictionary containing the name of each initial dataset and the expected number of samples in each.

    Raises:
        FileNotFoundError: If the 'initial_datasets.json' file is not found in the 'control' subfolder, or if any of the initial datasets is missing from the 'data' subfolder.
        ValueError: If the number of samples in any of the initial datasets does not match the expected count.

    """

    initial_datasets_json_file = training_dir / "control" / "initial_datasets.json"

    # Check if the 'initial_datasets.json' file exists
    if not initial_datasets_json_file.is_file():
        error_msg = f"The 'initial_datasets.json' file is missing from '{initial_datasets_json_file.parent}'."
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)
        # raise FileNotFoundError(error_msg)

    # Load the 'initial_datasets.json' file
    with initial_datasets_json_file.open() as file:
        initial_datasets = json.load(file)

    # Check each initial dataset
    for dataset_name, expected_num_samples in initial_datasets.items():
        dataset_path = training_dir / "data" / dataset_name

        # Check if the dataset exists in the 'data' subfolder
        if not dataset_path.is_dir():
            error_msg = f"Initial dataset '{dataset_name}' is missing from the 'data' subfolder."
            logging.error(f"{error_msg}\nAborting...")
            sys.exit(1)
            # raise ValueError(error_msg)

        # Check if the number of samples in the dataset matches the expected count
        box_path = dataset_path / "set.000" / "box.npy"
        if not box_path.is_file():
            error_msg = f"No box.npy found in the dataset '{dataset_name}'."
            logging.error(f"{error_msg}\nAborting...")
            sys.exit(1)
            # raise ValueError(error_msg)

        num_samples = len(np.load(str(box_path)))
        if num_samples != expected_num_samples:
            error_msg = f"Unexpected number of samples ({num_samples}) found in initial dataset '{dataset_name}'. Expected {expected_num_samples}."
            logging.error(f"{error_msg}\nAborting...")
            sys.exit(1)
            # raise ValueError(error_msg)

    return initial_datasets
