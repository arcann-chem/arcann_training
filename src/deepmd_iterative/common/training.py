from pathlib import Path
import logging
import sys
import json

# Non-standard imports
import numpy as np


def get_decay_steps(nb_structures: int, decay_steps_min: int = 5000) -> int:
    """
    Calculate the number of decay steps (tau) for a given number of structures to train (N).

    Args:
        nb_structures (int): The total number of structures to train (N).
        decay_steps_min (int, optional): The minimum number of decay steps (tau_min). Default is 5000.

    Returns:
        int: The calculated number of decay steps (tau).

    Raises:
        ValueError: If nb_structures is not a positive integer, or if decay_steps_min is not a positive integer.
    """

    # Check if nb_structures is a positive integer
    if not isinstance(nb_structures, int) or nb_structures <= 0:
        logging.critical("nb_structures must be a positive integer")
        logging.critical("Aborting...")
        sys.exit(1)
        #raise ValueError("nb_structures must be a positive integer")

    # Check if decay_steps_min is a positive integer
    if not isinstance(decay_steps_min, int) or decay_steps_min <= 0:
        logging.critical("decay_steps_min must be a positive integer")
        logging.critical("Aborting...")
        sys.exit(1)
        #raise ValueError("decay_steps_min must be a positive integer")

    # Round down to the nearest multiple of 10000
    floored_nb_structures = nb_structures // 10000 * 10000

    # Calculate the decay steps based on the number of structures
    if floored_nb_structures < 20000:
        decay_steps = decay_steps_min
    elif floored_nb_structures < 100000:
        decay_steps = floored_nb_structures // 4
    else:
        extra_decay_steps = (floored_nb_structures - 50000) // 50000 * decay_steps_min
        decay_steps = 20000 + extra_decay_steps

    return decay_steps


def get_decay_rate(stop_batch: int, start_lr: float, stop_lr: float, decay_steps: int) -> float:
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
        logging.critical("The start learning rate must be a positive number.")
        logging.critical("Aborting...")
        sys.exit(1)
        #raise ValueError("The start learning rate must be a positive number.")

    # Calculate the decay rate using the given training parameters
    decay_rate = np.exp(np.log(stop_lr / start_lr) / (stop_batch / decay_steps))

    return decay_rate


def get_learning_rate(training_step: int, start_lr: float, decay_rate: float, decay_steps: int) -> float:
    """
    Calculate the learning rate (alpha) at a given training step (t), based on the given parameters.

    Args:
        training_step (int): The current training step (t).
        start_lr (float): The starting learning rate (alpha0).
        decay_rate (float): The decay rate (labmda).
        decay_steps (int): The number of decay steps (tau).

    Returns:
        The learning rate at the current training step.

    Raises:
        ValueError: If any of the arguments are not positive
    """

    # Check that all arguments are positive
    if not all(arg > 0 for arg in (training_step, start_lr, decay_rate, decay_steps)):
        logging.critical("All arguments must be positive.")
        logging.critical("Aborting...")
        sys.exit(1)
        #raise ValueError("All arguments must be positive.")

    # Calculate the learning rate based on the current training step
    step_ratio = training_step / decay_steps
    decay_factor = decay_rate ** step_ratio
    learning_rate = start_lr * decay_factor

    return learning_rate


def check_initial_datasets(training_iterative_apath: Path) -> dict:
    """Check the initial datasets

    Args:
        training_iterative_apath (Path): Path object to the root training folder

    Returns:
        dict: JSON dictionary
    """
    if (training_iterative_apath / "control" / "initial_datasets.json").is_file():
        logging.info(
            f"Loading initial_datasets.json from {training_iterative_apath / 'control'}"
        )
        # logging.info("Loading: "+str((training_iterative_apath/"control"/"initial_datasets.json")))
        initial_datasets_json = json.load(
            (training_iterative_apath / "control" / "initial_datasets.json").open()
        )
        for f in initial_datasets_json:
            if not (training_iterative_apath / "data" / f).is_dir():
                logging.critical(f"Initial set not found in data: {f}")
                logging.critical(f"Aborting...")
                sys.exit(1)
            else:
                if (
                        np.load(
                            str(
                                training_iterative_apath
                                / "data"
                                / f
                                / "set.000"
                                / "box.npy"
                            )
                        ).shape[0]
                        != initial_datasets_json[f]
                ):
                    logging.critical(f"Missmatch in count for the set: {f}")
                    logging.critical(f"Aborting...")
                    sys.exit(1)
        return initial_datasets_json
    else:
        logging.critical(
            f"datasets_initial.json not present in: {training_iterative_apath / 'control'}"
        )
        logging.critical(f"Aborting...")
        sys.exit(1)
