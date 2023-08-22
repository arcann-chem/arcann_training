"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/22

The utils module provides helper functions.

Functions
---------
catch_errors_decorator(func: Callable[..., Any]) -> Callable[..., Any]
    A decorator to wrap a function and catche exceptions raised during its execution.

convert_seconds_to_hh_mm_ss(seconds: float) -> str
    A function to convert a time duration in seconds to the format of HH:MM:SS.
"""
# Standard library modules
import logging
from typing import Any, Callable


def catch_errors_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap a function and catches exceptions raised during its execution.

    Parameters
    ----------
    func : function
        The function to be decorated.

    Returns
    -------
    function
        The wrapped function.

    Raises
    -------
    Exception
        If an error occurs during the execution of the decorated function.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.debug(
                f"An error occurred while executing `{func.__name__}`: `{e.__class__.__name__}`"
            )
            logging.error(f"{e}")
            logging.error(f"Aborting...")
            raise

    return wrapper


# Unittested
@catch_errors_decorator
def convert_seconds_to_hh_mm_ss(seconds: float) -> str:
    """
    Convert a time duration in seconds to the format of HH:MM:SS.

    Parameters
    ----------
    seconds : float
        The time duration in seconds.

    Returns
    -------
    str
        The equivalent time duration in hours, minutes, and seconds in the format of HH:MM:SS.

    Raises
    ------
    None
    """
    # Convert the duration to hours, minutes, and seconds
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    # Return the time duration as a string in the format of HH:MM:SS
    return "%d:%02d:%02d" % (hours, minutes, seconds)
