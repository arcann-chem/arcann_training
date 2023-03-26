"""
Author: Rolf David
Created: 2023/01/01
Last modified: 2023/03/26
"""
# Standard library modules
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from xml.dom import minidom

# Local imports
from deepmd_iterative.common.errors import catch_errors_decorator

# Unittested
@catch_errors_decorator
def convert_list_of_strings_to_xml(list_string: List[str]) -> ET.ElementTree:
    """
    Convert a list of strings to an XML tree.

    Parameters
    ----------
    list_string : List[str]
        A list of strings, where each string represents a single line of the XML tree.

    Returns
    -------
    ET.ElementTree
        An XML tree.
    """
    # Join the lines of the list into a single string.
    xml_string = "".join(list_string)
    # Parse the string into an XML tree and return it.
    return ET.ElementTree(ET.fromstring(xml_string))


# Unittested
@catch_errors_decorator
def convert_xml_to_list_of_strings(xml_tree: ET.ElementTree) -> List[str]:
    """
    Convert an XML tree to a list of strings.

    Parameters
    ----------
    xml_tree : xml.etree.ElementTree.ElementTree
        The XML tree to convert.

    Returns
    -------
    List[str]
        A list of strings, where each string represents a single line of the XML tree.
    """
    # Convert the XML tree to a string.
    xml_string = ET.tostring(xml_tree.getroot(), encoding="unicode", method="xml")
    # Split the string into a list of lines.
    lines = [line.strip() for line in xml_string.splitlines()]
    # Return the list of lines.
    return lines


# Unittested
@catch_errors_decorator
def parse_xml_file(xml_file_path: Path) -> ET.ElementTree:
    """
    Parses an XML file and returns its corresponding ElementTree object.

    Parameters
    ----------
    xml_file_path : Path
        The path to the XML file.

    Returns
    -------
    ElementTree
        The parsed ElementTree object representing the XML file.

    Raises
    ------
    FileNotFoundError
        If the specified file cannot be found.
    ET.ParseError
        If the XML file is not well-formed and cannot be parsed.
    """
    # Check if the file exists
    if not xml_file_path.is_file():
        error_msg = f"File not found {xml_file_path.name} not in {xml_file_path.parent}"
        raise FileNotFoundError(error_msg)
    else:
        try:
            with open(xml_file_path, "r") as xml_file:
                xml_tree = ET.parse(xml_file)
                return xml_tree
        except ET.ParseError:
            error_msg = f"Failed to parse XML file: {xml_file_path.name}"
            raise ET.ParseError(error_msg)


# Unittested
@catch_errors_decorator
def write_xml(xml_tree: ET.ElementTree, file_path: Path):
    """
    Writes an XML tree to a file at the specified path.

    Parameters
    ----------
    xml_tree : ElementTree
        The ElementTree object to write to a file.
    file_path : Path
        The path to the file to write.
    """
    # Convert the ElementTree object to an XML string.
    xml_string = ET.tostring(xml_tree.getroot(), encoding="unicode", method="xml")
    # Use minidom to pretty-print the XML string with indentation.
    xml_pretty_string = minidom.parseString(xml_string).toprettyxml(indent=" ")
    # Write the pretty-printed XML string to the specified file path.
    with file_path.open("w") as f:
        f.write(xml_pretty_string)
