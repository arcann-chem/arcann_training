"""
Created: 2023/01/01
Last modified: 2023/04/17

The filesystem module provides functions to handle file and directory management 

Functions
---------
change_directory(directory_path: Path) -> None
    A function to change the current working directory to the given path.
    
check_directory(directory_path: Path, abort_on_error: bool = True, error_msg: str = "default") -> None
    A function to check if the given directory exists and logs a warning or raises an error if it does not.
    
check_file_existence(file_path: Path, expected_existence: bool = True, abort_on_error: bool = True, error_msg: str = "default") -> None
    A function to check if a file exists or not and logs a message or raises an error depending on the parameters.
    
remove_file(file_path: Path) -> None
    A function to delete a file at the specified path if it exists.
    
remove_files_matching_glob(directory_path: Path, file_glob: str) -> None
    A function to remove all files in a directory that match a specified file glob pattern.
    
remove_tree(directory_path: Path) -> None
    A function to recursively remove a directory tree and its contents.
"""
# Standard library modules
import logging
import os
from pathlib import Path

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator

# Unittested
@catch_errors_decorator
def change_directory(directory_path: Path) -> None:
    """
    Change the current working directory to the given path.

    Parameters
    ----------
    directory_path : Path
        The path to the directory to change to.

    Returns
    -------
    None

    Raises
    -------
    FileNotFoundError
        If the directory does not exist.
    OSError
        If there is an error in changing the directory.
    """

    # Check if the directory exists and is a ddir
    if not directory_path.is_dir():
        error_msg = f"Directory not found: {directory_path}"
        raise FileNotFoundError(error_msg)

    # Try to change the directory
    try:
        os.chdir(directory_path)
    except OSError as e:
        error_msg = f"Error in changing directory to {directory_path}: {e}"
        raise OSError(error_msg)


# Unittested
@catch_errors_decorator
def check_directory(
    directory_path: Path, abort_on_error: bool = True, error_msg: str = "default"
) -> None:
    """
    Check if the given directory exists and logs a warning or raises an error if it does not.

    Parameters
    ----------
    directory_path : Path
        The path to the directory to check.
    abort_on_error : bool, optional
        If True, raise an error and abort execution. If False, log a warning message instead. Default is True.
    error_msg : str, optional
        An optional error message to use in place of the default message. Default is "default".

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the directory does not exist and `abort_on_error` is True.
    """
    # Check if the directory exists
    if not directory_path.is_dir():
        # Create error message
        if error_msg == "default":
            error_msg = f"Directory not found: {directory_path}"

        # Log or raise error and abort if needed
        if abort_on_error:
            raise FileNotFoundError(error_msg)
        else:
            logging.warning(error_msg)


# Unittested
@catch_errors_decorator
def check_file_existence(
    file_path: Path,
    expected_existence: bool = True,
    abort_on_error: bool = True,
    error_msg: str = "default",
) -> None:
    """
    Check if a file exists or not and logs a message or raises an error depending on the parameters.

    Parameters
    ----------
    file_path : Path
        The path to the file to check.
    expected_existence : bool, optional
        True if the file is expected to exist, False if it is expected not to exist. Default is True.
    abort_on_error : bool, optional
        True if the program should abort if the file does not exist (or exists, depending on `expected_existence`). False otherwise. Default is True.
    error_msg : str, optional
        A custom error message to display in case of error. Default is "default".

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the file is expected to exist but does not, and `abort_on_error` is True.
    FileExistsError
        If the file is expected not to exist but does, and `abort_on_error` is True.
    """

    exists = file_path.is_file()

    if exists != expected_existence:
        if expected_existence:
            message = f"File not found: {file_path.name} not in {file_path.parent}"
            if abort_on_error:
                raise FileNotFoundError(
                    message if error_msg == "default" else error_msg
                )
            else:
                logging.warning(message if error_msg == "default" else error_msg)
        else:
            message = f"File found: {file_path.name}  in {file_path.parent}"
            if abort_on_error:
                raise FileExistsError(message if error_msg == "default" else error_msg)
            else:
                logging.warning(message if error_msg == "default" else error_msg)


# Unittested
@catch_errors_decorator
def remove_file(file_path: Path) -> None:
    """
    Delete a file at the specified path if it exists.

    Parameters
    ----------
    file_path : Path
        The path to the file to delete.

    Returns
    -------
    None

    Raises
    ------
    None
    """

    # Delete the file
    if file_path.is_file():
        file_path.unlink()


# Unittested
@catch_errors_decorator
def remove_files_matching_glob(directory_path: Path, file_glob: str) -> None:
    """
    Remove all files in a directory that match a specified file glob pattern.

    Parameters
    ----------
    directory_path : Path
        The directory where the files are located.
    file_glob : str
        The file glob pattern to match.

    Returns
    -------
    None

    Raises
    ------
    NotADirectoryError
        If the `directory_path` argument does not point to a directory.
    Exception
        If there is an error in removing the files.
    """

    # Remove the matching files
    if not directory_path.is_dir():
        error_msg = f"Not a directory {directory_path}."
        raise NotADirectoryError(error_msg)

    for file_path in directory_path.glob(file_glob):
        try:
            file_path.unlink()
            logging.debug(f"Removed file: {file_path}")
        except Exception as e:
            error_msg = f"Failed to remove file {file_path}: {e}"
            raise Exception(error_msg)


# Unittested
@catch_errors_decorator
def remove_tree(directory_path: Path) -> None:
    """
    Recursively remove a directory tree and its contents.

    Parameters
    ----------
    directory_path : Path
        The path to the directory to remove.

    Returns
    -------
    None

    Raises
    ------
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
