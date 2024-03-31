"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/31

The ipi module provides functions to manipulate i-PI data (as XML tree).

Functions
---------
get_temperature_from_ipi_xml(input_file: str) -> float
    A function to extract the temperature value from an XML file and returns it as a float.
"""

# TODO: Homogenize the docstrings for this module

# Standard library modules
import xml.etree.ElementTree as ET

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


# Unittested
@catch_errors_decorator
def get_temperature_from_ipi_xml(input_file: str) -> float:
    """
    Extract the temperature value from an XML file and returns it as a float.

    Parameters
    ----------
    input_file : str
        A string representing the file path of the input XML file.

    Returns
    -------
    float
        The temperature value in the XML file.

    Raises
    ------
    Exception
        If the input_file could not be read.
    ValueError
        If the temperature value could not be extracted from the input_file.

    """
    try:
        tree = ET.parse(input_file)
    except (OSError, ET.ParseError) as e:
        error_msg = f"Error reading input file '{input_file}':'{e}'."
        raise Exception(error_msg)
    root = tree.getroot()

    temperature = None
    for child in root.iter():
        if "temperature" in child.tag:
            try:
                temperature = float(child.text)
            except ValueError as e:
                error_msg = f"Error parsing temperature value in '{input_file}': '{e}'."
                raise ValueError(error_msg)

    if temperature is None:
        error_msg = f"Temperature value not found in '{input_file}'."
        raise ValueError(error_msg)

    return temperature
