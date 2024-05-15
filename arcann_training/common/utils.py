"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15

Utility module providing helper functions.

Functions
---------
catch_errors_decorator(func: Callable[..., Any]) -> Callable[..., Any]
    Decorator to wrap a function and catch exceptions during execution.
convert_seconds_to_hh_mm_ss(seconds: float) -> str
    Converts a time duration in seconds to the format HH:MM:SS.
natural_sort_key(s: str) -> List[Union[int, str]]
    Provides a natural sorting key for alphanumeric strings.
"""

# Standard library modules
import logging
import re
from typing import Any, Callable, List, Union


# Unittested
def catch_errors_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to wrap a function and catch exceptions during execution.

    Parameters
    ----------
    func : function
        The function to be decorated.

    Returns
    -------
    function
        The wrapped function.

    Raises
    ------
    Exception
        If an error occurs during execution of the decorated function.
    """
    logger = logging.getLogger("ArcaNN")

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{e}")
            logger.debug(f"Error in '{func.__name__}': {e}", exc_info=True)
            logger.error("Aborting the program due to an error.")
            raise

    return wrapper


# Unittested
@catch_errors_decorator
def convert_seconds_to_hh_mm_ss(seconds: float) -> str:
    """
    Converts a time duration in seconds to the format HH:MM:SS.

    Parameters
    ----------
    seconds : float
        The time duration in seconds.

    Returns
    -------
    str
        The equivalent time duration in HH:MM:SS format.

    Raises
    ------
    None
    """
    # Convert the duration to hours, minutes, and seconds
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    # Return the time duration as a string in the format of HH:MM:SS
    return "%d:%02d:%02d" % (hours, minutes, seconds)


# TODO: Add tests for this function
@catch_errors_decorator
def natural_sort_key(s: str) -> List[Union[int, str]]:
    """
    Provides a natural sorting key for alphanumeric strings.

    Parameters
    ----------
    s : str
        The string to generate the sorting key for.

    Returns
    -------
    List[Union[int, str]]
        The sorting key.

    Raises
    ------
    TypeError
        If the input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    if s == "":
        return []
    _nsre = re.compile("([0-9]+)")
    return [
        format(int(text), "020d") if text.isdigit() else text.lower()
        for text in re.split(_nsre, s)
        if text
    ]
