import logging
import sys
from typing import List, Tuple, Union

# Others
import re


def analyze_plumed_file_for_movres(
    plumed_lines: List[str],
) -> Tuple[bool, Union[int, bool]]:

    """
    Analyzes a Plumed file to extract information about the MOVINGRESTRAINT keyword and the last STEP value used.

    Args:
        plumed_lines (List[str]): A list of strings representing the contents of a Plumed file.

    Returns:
        A tuple containing two items:
        - A boolean indicating whether the MOVINGRESTRAINT keyword is present in the Plumed file.
        - An integer indicating the value of the last STEP keyword in the MOVINGRESTRAINT section of the Plumed file,
          or False if the STEP keyword is not found.

    Raises:
        SystemExit: If the STEP keyword is not found for MOVINGRESTRAINT.

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
            logging.error(f"{error_msg}\nAborting...")
            sys.exit(1)
    else:
        return False, False
