from pathlib import Path
import logging
import sys


def step_path_match(step_name: str):
    """Check if the requested step matches the folder where it is launched

    Args:
        step_name (str): The name of the step
    """
    current_apath = Path(".").resolve()
    if step_name not in current_apath.name:
        logging.error(f"The folder is not an {step_name} one")
        logging.error(f"Current folder: {current_apath}")
        logging.error(f"Aborting...")
        sys.exit(1)
    else:
        pass
