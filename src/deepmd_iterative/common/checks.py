from pathlib import Path
import logging
import sys


def validate_step_folder(step_name: str) -> None:
    """
    Check if the current directory matches the expected directory for the step.

    Args:
        step_name (str): The name of the step being executed.

    Raises:
        ValueError: If the current directory name does not contain the step name.
    """
    # Get the path of the current directory
    current_directory = Path(".").resolve()

    # Check if the current directory name contains the step name
    if step_name not in current_directory.name:
        logging.error(
            f"The current directory ({current_directory}) does not match the expected directory for the {step_name} step."
        )
        logging.error("Aborting...")
        sys.exit(1)
        # raise ValueError(f"The current directory ({current_directory}) does not match the expected directory for the {step_name} step.")
