from pathlib import Path
import logging
import sys
import os


def check_file(file_path: Path, exists: bool, abort: bool, error_msg: str = "default"):
    """Check if a file exists or not, abort or not
        exists/abort:
        True/True: if the file does't exist, abort
        False/True: if the file does exist, abort
        True/False: if the file does't exist, log only
        False/False: if the file does exist, log only

    Args:
        file_path (Path): Path object to the file
        exists (bool):  True to check if it should exists, False to check if it shouldn't
        abort (bool): True to abort, False to log only
        error_msg (str, optional): To override default error message. Defaults to "default".
    """
    if not exists and not file_path.is_file():
        if abort:
            logging.critical(
                f"File not found: {file_path.name} not in {file_path.parent}"
            ) if error_msg == "default" else logging.critical(error_msg)
            logging.critical("Aborting...")
            sys.exit(1)
        else:
            logging.warning(
                f"File not found: {file_path.name} not in {file_path.parent}"
            ) if error_msg == "default" else logging.warning(error_msg)
    elif not exists and file_path.is_file():
        if abort:
            logging.critical(
                f"File found: {file_path.name} not in {file_path.parent}"
            ) if error_msg == "default" else logging.critical(error_msg)
            logging.critical("Aborting...")
            sys.exit(1)
        else:
            logging.warning(
                f"File found: {file_path.name} not in {file_path.parent}"
            ) if error_msg == "default" else logging.warning(error_msg)


def check_dir(directory_path: Path, abort: bool, error_msg: str = "default"):
    """Check if directory exists

    Args:
        directory_path (Path): Path object to the directory
        abort (bool): True to abort, False to log only
        error_msg (str, optional): To override default error message. Defaults to "default".
    """
    if not directory_path.is_dir():
        if abort:
            if error_msg == "data":
                logging.critical(
                    f"No data folder to search for initial sets: {directory_path}"
                )
                logging.critical(f"Aborting...")
                sys.exit(1)
            else:
                logging.critical(
                    f"Directory not found: {directory_path}"
                ) if error_msg == "default" else logging.critical(error_msg)
                logging.critical(f"Aborting...")
                sys.exit(1)
        else:
            logging.warning(
                f"Directory not found: {directory_path}"
            ) if error_msg == "default" else logging.critical(error_msg)


def file_to_strings(file_path: Path) -> list:
    """Read a file as a list of strings (one line, one string)

    Args:
        file_path (Path): Path object to the file

    Returns:
        list: list of strings
    """
    if not file_path.is_file():
        logging.critical(f"File not found {file_path.name} not in {file_path.parent}")
        logging.critical(f"Aborting...")
        sys.exit(1)
    else:
        with file_path.open() as f:
            return f.readlines()


def remove_file(file_path: Path):
    """_summary_

    Args:
        file_path (Path): _description_
    """
    if file_path.is_file():
        file_path.unlink()


def remove_file_glob(directory_path: Path, file_glob: str):
    """_summary_

    Args:
        directory_path (Path): _description_
        file_glob (str): _description_
    """
    for f in directory_path.glob(file_glob):
        f.unlink()


def remove_tree(directory_path: Path):
    """_summary_

    Args:
        pth (Path): _description_
    """
    for child in directory_path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            remove_tree(child)
    directory_path.rmdir()


def write_file(file_path: Path, list_of_string: list):
    """Write a list of string to a file

    Args:
        file_path (Path): _description_
        list_of_string (list): _description_
    """
    file_path.write_text("".join(list_of_string))


def change_dir(directory_path: Path):
    """_summary_

    Args:
        directory_path (Path): Path to the new directory
    """
    if not directory_path.is_dir():
        logging.error(f"Directory not found: {directory_path}")
        logging.error(f"Aborting...")
        sys.exit(1)
    else:
        try:
            os.chdir(directory_path)
        except:
            logging.error(f"Error in changing dir: {directory_path}")
            logging.error(f"Aborting...")
            sys.exit(1)
