"""
Created: 2023/01/01
Last modified: 2023/03/27

The plumed module provides functions to manipulate PLUMED data (as list of strings).

Functions
---------
analyze_plumed_file_for_movres(plumed_lines: List[str]) -> Tuple[bool, Union[int, bool]]
    A function to analyze a Plumed file to extract information about the MOVINGRESTRAINT keyword and the last STEP value used.
"""
# Standard library modules
import re
from typing import List, Tuple, Union

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


@catch_errors_decorator
def analyze_plumed_file_for_movres(
    plumed_lines: List[str],
) -> Tuple[bool, Union[int, bool]]:
    """
    Analyze a Plumed file to extract information about the MOVINGRESTRAINT keyword and the last STEP value used.

    Parameters
    ----------
    plumed_lines : List[str]
        A list of strings representing the contents of a Plumed file.

    Returns
    -------
    Tuple[bool, Union[int, bool]]
        A tuple containing two items:
        - A boolean indicating whether the MOVINGRESTRAINT keyword is present in the Plumed file.
        - An integer indicating the value of the last STEP keyword in the MOVINGRESTRAINT section of the Plumed file,
          or False if the STEP keyword is not found.

    Raises
    ------
    ValueError
        If the STEP keyword is not found for MOVINGRESTRAINT.
    """
    # Find if MOVINGRESTRAINT is present
    movres_found = False
    for line in plumed_lines:
        if "MOVINGRESTRAINT" in line:
            movres_found = True
            break

    if movres_found:
        # Find the last value of the STEP keyword
        step_matches = re.findall(r"STEP\d*\s*=\s*(\d+)", "".join(plumed_lines))
        if len(step_matches) > 0:
            last_step = int(step_matches[-1])
            return True, last_step
        else:
            error_msg = "STEP not found for MOVINGRESTRAINT."
            raise ValueError(error_msg)
    else:
        return False, False
