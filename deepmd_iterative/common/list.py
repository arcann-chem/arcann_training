"""
Author: Rolf David
Created: 2023/01/01
Last modified: 2023/03/26
"""
# Standard library modules
from typing import List

# Local imports
from deepmd_iterative.common.errors import catch_errors_decorator

# Unittested
@catch_errors_decorator
def remove_strings_containing_substring_in_list_of_strings(
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
        raise TypeError(
            "Invalid input type. input_list must be a list of strings and substring must be a string."
        )

    if len(input_list) == 0:
        raise ValueError("input_list must not be empty.")

    output_list = [string.strip() for string in input_list if substring not in string]
    return output_list


# Unittested
@catch_errors_decorator
def replace_substring_in_list_of_strings(
    input_list: List[str], substring_in: str, substring_out: str
) -> List[str]:
    """
    Replaces a specified substring with a new substring in each string of a list.

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
        If substring_in or substring_out is an empty string.
    """
    if not isinstance(input_list, list):
        raise TypeError("Invalid input type. input_list must be a list of strings.")

    if not substring_in:
        raise ValueError("Invalid input. substring_in must be a non-empty string.")

    if not substring_out:
        raise ValueError("Invalid input. substring_out must be a non-empty string.")

    output_list = [
        string.replace(substring_in, substring_out).strip() for string in input_list
    ]
    return output_list
