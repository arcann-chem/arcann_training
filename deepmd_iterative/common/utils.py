"""
Author: Rolf David
Created: 2023/01/01
Last modified: 2023/03/26
"""
# Local imports
from deepmd_iterative.common.errors import catch_errors_decorator

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
    """
    # Convert the duration to hours, minutes, and seconds
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    # Return the time duration as a string in the format of HH:MM:SS
    return "%d:%02d:%02d" % (hours, minutes, seconds)
