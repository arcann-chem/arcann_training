"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/07/14

The utils module provides functions to manipulate lists.

Functions
---------
exclude_substring_from_string_list(input_list: List[str], substring: str) -> List[str]
    A function to remove all strings containing a given substring from a list of strings.

replace_substring_in_string_list(input_list: List[str], substring_in: str, substring_out: str) -> List[str]
    A function to replace a specified substring with a new substring in each string of a list.

string_list_to_textfile(file_path: Path, string_list: List[str]) -> None
    A function to write a list strings to a text file.

textfile_to_string_list(file_path: Path) -> List[str]
    A function to read the contents of a text file and return a list of strings.
"""

# TODO: Homogenize the docstrings for this module

# Standard library modules
from pathlib import Path
from typing import List

# Local imports
from arcann_training.common.utils import catch_errors_decorator


# Unittested
@catch_errors_decorator
def exclude_substring_from_string_list(
    input_list: List[str], substring: str
) -> List[str]:
    """
    Remove all strings containing a given substring from a list of strings.

    Parameters
    ----------
    input_list : List[str]
        The list of strings to remove the strings containing the substring.
    substring : str
        The substring to look for in the input strings.

    Returns
    -------
    List[str]
        A new list of strings with all strings containing the given substring removed.

    Raises
    ------
    TypeError
        If input_list is not a list of strings, or substring is not a string.
    ValueError
        If input_list is empty.
    """
    if not isinstance(input_list, list) or not isinstance(substring, str):
        error_msg = f"Invalid input type. '{input_list}' must be a '{type([])}' of '{type('')}' and substring must be a '{type('')}'."
        raise TypeError(error_msg)

    # if len(input_list) == 0:
    #     error_msg = f"'{input_list}' must not be empty."
    #     raise ValueError(error_msg)

    output_list = [string.strip() for string in input_list if substring not in string]
    return output_list


# Unittested
@catch_errors_decorator
def replace_substring_in_string_list(
    input_list: List[str], substring_in: str, substring_out: str
) -> List[str]:
    """
    Replace a specified substring with a new substring in each string of a list.

    Parameters
    ----------
    input_list : List[str]
        A list of input strings.
    substring_in : str
        The substring to replace in the input strings.
    substring_out : str
        The new substring to replace with.

    Returns
    -------
    List[str]
        A list of output strings with the specified substring replaced by the new substring.

    Raises
    ------
    TypeError
        If input_list is not a list of strings.
    ValueError
        If substring_in is an empty string.
    """
    if not isinstance(input_list, list):
        error_msg = f"Invalid input type. '{input_list}' must be a '{type([])}' of '{type('')}'."
        raise TypeError(error_msg)

    if not substring_in:
        error_msg = f"Invalid input. '{substring_in}' must be a non-empty '{type('')}'."
        raise ValueError(error_msg)

    # if not substring_out:
    #    raise ValueError("Invalid input. substring_out must be a non-empty string.")

    output_list = [
        string.replace(substring_in, substring_out).strip() for string in input_list
    ]
    return output_list


# Unittested
@catch_errors_decorator
def string_list_to_textfile(
    file_path: Path, string_list: List[str], read_only: bool = False
) -> None:
    """
    Write a list of strings to a text file.

    Parameters
    ----------
    file_path : Path
        A 'Path' object representing the path to the file.
    string_list : list[str]
        A list of strings to be written to the text file.
    read_only : bool, optional
        If True, sets the file's permissions to read-only after writing. If False (the default), the file's
        permissions are not modified.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the 'file_path' argument is not a 'Path' object or if the 'string_list'
        argument is not a list of strings.
    ValueError
        If the 'string_list' argument is empty.
    OSError
        If there is an error writing the file.

    Examples
    --------
    >>> file_path = Path('path/to/file.txt')
    >>> string_list = ['This is the first line.', 'This is the second line.', 'This is the third line.']
    >>> string_list_to_textfile(file_path, string_list)
    """

    if not isinstance(file_path, Path):
        error_msg = f"'{file_path}' must be a '{type(Path(''))}'."
        raise TypeError(error_msg)

    if not isinstance(string_list, list) or not all(
        isinstance(s, str) for s in string_list
    ):
        error_msg = f"'{string_list}' must be a '{type([])}' of '{type('')}.'"
        raise TypeError(error_msg)

    if not string_list:
        error_msg = f"'{string_list}' must not be empty."
        raise ValueError(error_msg)

    if file_path.exists():
        file_path.chmod(0o644)

    try:
        with file_path.open("w") as text_file:
            text_file.write("\n".join(string_list))
            text_file.write("\n")
    except OSError as e:
        error_msg = f"error writing to file '{file_path}': '{e}'"
        raise OSError(error_msg)

    if read_only:
        current_permissions = file_path.stat().st_mode
        # Remove the write permission (0222) while keeping others intact
        new_permissions = current_permissions & ~0o222
        # Update the file permissions
        file_path.chmod(new_permissions)


# Unittested
@catch_errors_decorator
def textfile_to_string_list(file_path: Path) -> List[str]:
    """
    Read the contents of a text file and return a list of strings,
    where each string represents a line of text from the file. The function also
    removes newline characters from the end of each line.

    Parameters
    ----------
    file_path : Path
        A 'Path' object representing the path to the file.

    Returns
    -------
    list
        A list of strings, where each string represents a line of text
        from the file. Returns an empty list if the file is empty.

    Raises
    ------
    TypeError
        If the 'file_path' argument is not a 'Path' object.
    FileNotFoundError
        If the file does not exist or is not a file.
    OSError
        If there is an error reading the file.

    Examples
    --------
    >>> file_path = Path('path/to/file.txt')
    >>> textfile_to_string_list(file_path)
    ['This is the first line.', 'This is the second line.', 'This is the third line.']
    """

    if not isinstance(file_path, Path):
        error_msg = f"'{file_path}' must be a .{type(Path(''))}'."
        raise TypeError(error_msg)

    if not file_path.exists() or not file_path.is_file():
        error_msg = f"File '{file_path}' does not exist."
        raise FileNotFoundError(error_msg)

    try:
        with file_path.open("r") as text_file:
            file_content = text_file.readlines()
    except OSError as e:
        error_msg = f"error reading the file '{file_path}': '{e}'."
        raise OSError(error_msg)

    with file_path.open("r") as text_file:
        file_content = text_file.readlines()

    file_content = [line.strip() for line in file_content]
    return file_content if file_content else []
