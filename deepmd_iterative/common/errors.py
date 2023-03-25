# Standard library modules
import logging
from typing import Any, Callable


def catch_errors_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that wraps a function and catches exceptions raised during its execution.

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
                f"An error occurred while executing {func.__name__}: {e.__class__.__name__}"
            )
            logging.error(f"{e}")
            logging.error(f"Aborting...")
            raise

    return wrapper
