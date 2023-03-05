from pathlib import Path
import logging
import sys
from typing import List
import os


def check_file_existence(
    file_path: Path,
    expected_existence: bool = True,
    abort_on_error: bool = True,
    error_msg: str = "default",
) -> None:
    """
    Checks if a file exists or not, logs a message or aborts the program depending on the parameters.

    Args:
        file_path (Path): The path to the file to check.
        expected_existence (bool): True if the file is expected to exist, False if it is expected not to exist.
        abort_on_error (bool): True if the program should abort if the file does not exist (or exists, depending on expected_existence), False otherwise.
        error_msg (str, optional): A custom error message to display in case of error. Defaults to "default".

    Returns:
        None

    Raises:
        SystemExit(1): If the directory does not exist and `abort_on_error` is True.
    """

    exists = file_path.is_file()
    if exists != expected_existence:
        if expected_existence:
            logging_func = logging.critical
            message = f"File not found: {file_path.name} not in {file_path.parent}"
        else:
            logging_func = logging.warning
            message = f"File found: {file_path.name} not in {file_path.parent}"
        if abort_on_error:
            if error_msg == "default":
                logging_func(f"{message}\nAborting...")
            else:
                logging_func(f"{error_msg}\nAborting...")
            sys.exit(1)
        else:
            if error_msg == "default":
                logging_func(message)
            else:
                logging_func(error_msg)


def check_directory(
    directory_path: Path, abort_on_error: bool = True, error_msg: str = "default"
) -> None:
    """
    Check if the given directory exists and logs an error or aborts execution if it does not.

    Args:
        directory_path (Path): The path to the directory to check.
        abort (bool): If True, aborts execution with a critical error message. If False, logs a warning message.
        error_msg (str, optional): An optional error message to use in place of the default message.

    Returns:
        None

    Raises:
        SystemExit(1): If the directory does not exist and `abort_on_error` is True.
    """

    # Check if the directory exists
    if not directory_path.is_dir():
        # Create error message
        if error_msg == "default":
            error_msg = f"Directory not found: {directory_path}"

        # Log or raise error and abort if needed
        if abort_on_error:
            logging.error(f"{error_msg}\nAborting...")
            sys.exit(1)
        else:
            logging.warning(error_msg)


def file_to_list_of_strings(file_path: Path) -> List[str]:
    """
    Reads a file and returns its contents as a list of strings.

    Args:
        file_path (Path): A `Path` object pointing to the file to be read.

    Returns:
        List[str]: A list of strings, where each string is a line from the file.

    Raises:
        SystemExit(1): If the specified file does not exist.
    """

    if not file_path.is_file():
        # If the file does not exist, log an error message and abort
        error_msg = f"File not found {file_path.name} not in {file_path.parent}"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)
    else:
        # If the file exists, open it and read its lines into a list
        with file_path.open() as f:
            return [line.strip() for line in f.readlines()]


def write_list_of_strings_to_file(file_path: Path, list_of_strings: List[str]) -> None:
    """
    Write a list of strings to a file.

    Args:
        file_path (Path): The path to the file to be written.
        list_of_strings (List): The list of strings to write to the file.

    Returns:
        None

    Raises:
        SystemExit(1): If the specified file does not exist.
    """
    try:
        # Write the strings to the file
        with file_path.open(mode="w") as file:
            file.write("\n".join(list_of_strings) + "\n")
    except (OSError, IOError) as e:
        # Handle any errors that occur during file writing
        error_msg = f"Error writing to file {file_path}: {e}"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)


def remove_file(file_path: Path) -> None:
    """
    Deletes a file at the specified path if it exists.

    Args:
        file_path (Path): The path to the file to delete.

    Returns:
        None
    """

    # Delete the file
    if file_path.is_file():
        file_path.unlink()


def remove_files_matching_glob(directory_path: Path, file_glob: str) -> None:
    """
    Remove all files in a directory that match a specified file glob pattern.

    Args:
        directory_path (Path): The directory where the files are located.
        file_glob (str): The file glob pattern to match.

    Returns:
        None

    Raises:
        SystemExit(1): If the directory does not exist or is not a directory.
    """

    # Remove the matching files
    for file_path in directory_path.glob(file_glob):
        try:
            file_path.unlink()
            # logging.info(f"Removed file: {file_path}")
        except Exception as e:
            error_msg = f"Failed to remove file {file_path}: {e}"
            logging.error(f"{error_msg}\nAborting...")
            sys.exit(1)


def remove_tree(directory_path: Path) -> None:
    """
    Recursively remove a directory tree and its contents.

    Args:
        directory_path (Path): The path to the directory to remove.

    Returns:
        None
    """

    # Iterate over each child of the directory
    for child in directory_path.iterdir():
        if child.is_file():
            # If the child is a file, delete it
            child.unlink()
        else:
            # If the child is a directory, recursively remove its contents
            remove_tree(child)

    # Remove the now-empty directory
    directory_path.rmdir()


def change_directory(directory_path: Path) -> None:
    """
    Change the current working directory to the given path.

    Args:
        directory_path (Path): The path to the directory to change to.

    Returns:
        None

    Raises:
        SysExit(1): If the directory does not exist or there is an error in changing the directory.
    """

    # Check if the directory exists
    if not directory_path.is_dir():
        error_msg = f"Directory not found: {directory_path}"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)

    # Try to change the directory
    try:
        os.chdir(directory_path)
    except Exception as e:
        error_msg = f"Error in changing directory to {directory_path}: {e}"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)
