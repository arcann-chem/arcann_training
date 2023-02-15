def convert_seconds_to_hh_mm_ss(seconds: float) -> str:
    """
    Convert a time in seconds to a string in the format of HH:MM:SS.

    Args:
        seconds (float): The time duration in seconds.

    Returns:
        str: The equivalent time duration in hours, minutes, and seconds in the format of HH:MM:SS.
    """
    # Convert the duration to hours, minutes, and seconds
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    # Return the time duration as a string in the format of HH:MM:SS
    return "%d:%02d:%02d" % (hours, minutes, seconds)
