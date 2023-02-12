from pathlib import Path
import logging
import sys
import json

# ### Non-standard imports
import numpy as np


def get_decay_steps(nb_structures: int, min_decay: int) -> int:
    """Calculate decay steps in function of nb_structures:
        Floor the number of total trained structures to nearest 10000: floored_nb_structures
        If floored_nb_structures < 20 000, decay_steps = min
        if floored_nb_structures is => 20 000 and < 100000, decay_steps = floored_nb_structures / 4
        If floored_nb_structures >= 100 000, decay_steps = 20 000 + min for each 50 000 increments over 50 000
            100 000 >= floored_nb_structures < 150 000, decay_steps = 20 000 + 2 * min
            200 000 >= floored_nb_structures < 250 000 decay_steps = 20 000 + 3 * min

    Args:
        nb_structures (int): Number of total structures to train
        min_decay (int, optional): Minimum decay steps.

    Returns:
        decay_steps (int): decay steps (tau)
    """
    decay_steps = min_decay
    power_val = np.power(10, np.int64(np.log10(nb_structures)))
    floored_nb_structures = int(np.floor(nb_structures / power_val) * power_val)
    if floored_nb_structures < 20000:
        decay_steps = min_decay
    elif floored_nb_structures < 100000:
        decay_steps = floored_nb_structures / 4
    else:
        decay_steps = 20000 + ((floored_nb_structures - 50000) / 100000) * 10000
    return int(decay_steps)


def get_decay_rate(
        stop_batch: int, start_lr: float, stop_lr: float, decay_steps: int
) -> float:
    """Get the decay rate (lambda)

    Args:
        stop_batch (int): final training step (tfinal)
        start_lr (float): starting learning rate (alpha0)
        stop_lr (float): ending learning rate (alpha(tfinal))
        decay_steps (int): decay steps (tau)

    Returns:
        (float): decay_rate (lambda)
    """
    return np.exp(np.log(stop_lr / start_lr) / (stop_batch / decay_steps))


def get_learning_rate(
        training_step: int, start_lr: float, decay_rate: float, decay_steps: int
) -> float:
    """Get the learning rate at step t

    Args:
        training_step (int): training step (t)
        start_lr (float): starting learning rate (alpha0)
        decay_rate (float): decay rate (lambda)
        decay_steps (int): decay steps (tau)

    Returns:
        (float): learning rate (alpha(t)) at step t
    """
    return start_lr * decay_rate ** (training_step / decay_steps)


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
