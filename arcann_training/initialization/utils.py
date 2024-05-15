"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15

Functions
---------
generate_main_json(user_input_json: Dict, default_input_json: Dict) -> Tuple[Dict, Dict, str]
    A function to generate the main JSON by combining values from the user input JSON and the default JSON.
check_properties_file(file_path: Path) -> dict
    A function to validate and extract the properties from a file containing types and masses.
check_lmp_properties(lmp_file: Path, properties: Dict) -> bool
    A function to validate that the properties in a LAMMPS data file match those specified in a properties dictionary.
check_dptrain_properties(user_files_path: Path, properties_dict: Dict)
    A function to check the properties in dptrain files.
check_typeraw_properties(type_raw_path, properties_dict)
    A function to check the properties in type.raw files.
"""

# TODO: Homogenize the docstrings for this module

# Standard library modules
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple

# Third-party modules
import numpy as np

# Local imports
from arcann_training.common.utils import catch_errors_decorator
from arcann_training.common.lammps import read_lammps_data
from arcann_training.common.json import load_json_file


# Unittested
@catch_errors_decorator
def generate_main_json(user_input_json: Dict, default_input_json: Dict) -> Tuple[Dict, Dict, str]:
    """
    Generate the main JSON by combining values from the user input JSON and the default JSON.
    If the user input JSON is invalid, an error is raised, and the script is terminated.
    Additionally, generate the merged input JSON.

    Parameters
    ----------
    user_input_json : dict
        The JSON containing user-defined parameters.
    default_input_json : dict
        The JSON containing default parameters.

    Returns
    -------
    Tuple[Dict, Dict, str]
        A tuple containing:
        - The main JSON.
        - The merged input JSON.
        - A message describing the validation result.

    Raises
    ------
    TypeError
        If the type of a user-defined parameter differs from the type of the default parameter.
    """
    # Merge the default and user input JSONs, with the user input JSON taking precedence.
    merged_input_json = {**default_input_json, **user_input_json}

    for key, default_value in default_input_json.items():
        user_value = user_input_json.get(key, default_value)

        if not isinstance(user_value, type(default_value)):
            error_msg = f"Type mismatch for '{key}': expected {type(default_value).__name__}, got {type(user_value).__name__}"
            raise TypeError(error_msg)

        if isinstance(user_value, list):
            if not all(isinstance(x, type(default_value[0])) for x in user_value):
                error_msg = f"Element type mismatch in list for key '{key}'."
                raise TypeError(error_msg)

    # Custom initialization for JSON structures
    main_json = deepcopy(merged_input_json)
    main_json["current_iteration"] = 0

    main_json["systems_auto"] = {key: {"index": idx} for idx, key in enumerate(main_json["systems_auto"])}

    return main_json, merged_input_json, str(main_json["current_iteration"]).zfill(3)


@catch_errors_decorator
def check_properties_file(file_path: Path) -> dict:
    """
    Validates and extracts the properties from a file containing types and masses.

    The file should have a section named 'type' followed by 'masses'. Each section lists elements with their types
    or masses. This function validates the structure and content of these sections and returns a dictionary where keys
    are the types as integers, and values are dictionaries containing the 'symbol' and 'mass' of each element.

    Parameters
    ----------
    file_path : Path
        The path to the properties file to be checked.

    Returns
    -------
    dict
        A dictionary where keys are types as integers, and values are dictionaries with 'symbol' as a string and 'mass' as a float.

    Raises
    ------
    ValueError
        If the file structure is incorrect, or line formats in the 'type' or 'masses' sections are invalid.
    """

    if not file_path.exists():
        error_msg = f"File not found: {file_path}"
        raise FileNotFoundError(error_msg)

    content = file_path.read_text()
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    if "type" not in lines or "masses" not in lines:
        error_msg = f"Both 'type' and 'masses' sections are required in the properties file {file_path}."
        raise ValueError(error_msg)

    type_index = lines.index("type")
    masses_index = lines.index("masses")

    if type_index >= masses_index:
        error_msg = f"'type' section should come before 'masses'. Check your properties file."
        raise ValueError(error_msg)

    types = {}
    masses = {}

    # Process 'type' section
    for line in lines[type_index + 1 : masses_index]:
        parts = line.split()
        if len(parts) != 2:
            error_msg = f"Line '{line}' does not have two parts in your 'type' section. Check your properties file."
            raise ValueError(error_msg)
        else:
            try:
                types[parts[0]] = int(parts[1])
            except ValueError:
                error_msg = f"Second part of line '{line}' in your 'type' section is not an integer. Check your properties file."
                raise ValueError(error_msg)

    # Process 'masses' section
    for line in lines[masses_index + 1 : masses_index + 1 + len(types)]:
        parts = line.split()
        if len(parts) != 2:
            error_msg = f"Line '{line}' does not have two parts in your 'masses' section. Check your properties file."
            raise ValueError(error_msg)
        else:
            try:
                masses[parts[0]] = float(parts[1])
            except ValueError:
                error_msg = f"Second part of line '{line}' in your 'masses' section is not an float. Check your properties file."
                raise ValueError(error_msg)

    if set(types) != set(masses):
        error_msg = f"Number of types and masses do not match. Check your properties file."
        raise ValueError(error_msg)

    # Combining types and masses into one dictionary
    type_mass_dictionary = {}
    for symbol, type_id in types.items():
        type_mass_dictionary[type_id] = {"symbol": symbol, "mass": masses[symbol]}

    return type_mass_dictionary


# TODO: Add tests for this function
@catch_errors_decorator
def check_lmp_properties(lmp_file: Path, properties: Dict) -> bool:
    """
    Validates that the properties in a LAMMPS data file match those specified in a properties dictionary.

    Parameters
    ----------
    lmp_file : Path
        The path to the LAMMPS data file.
    properties : Dict
        A dictionary of properties with atom types as keys and properties (such as mass) as values.

    Returns
    -------
    bool
        True if the LAMMPS file properties match the provided properties dictionary, False otherwise.

    Raises
    ------
    ValueError
        If there's a mismatch in atom types or their properties between the LAMMPS file and the properties dictionary.
    """

    num_atoms, num_atom_types, cell, masses, atoms = read_lammps_data(lmp_file)
    del num_atoms, cell, atoms

    if num_atom_types > len(properties):
        error_msg = f"In LMP file {lmp_file}, there are more atom types compared to the properties file."
        raise ValueError(error_msg)

    for atom_type_id in masses:
        if atom_type_id not in properties:
            error_msg = f"Atom type {atom_type_id} is not in the properties file but is present in the LMP file {lmp_file}."
            raise ValueError(error_msg)

        # Skipping placeholder masses
        if masses[atom_type_id] == 0.1:
            continue

        if not np.isclose(masses[atom_type_id], properties[atom_type_id]["mass"], atol=1e-2):
            error_msg = f"Masses do not match for atom type {atom_type_id} between the LMP file {lmp_file} and properties file."
            raise ValueError(error_msg)

    return True


# TODO: Add tests for this function
@catch_errors_decorator
def check_dptrain_properties(user_files_path: Path, properties_dict: Dict):
    dptrain_list = []
    for file in user_files_path.iterdir():
        if file.suffix != ".json":
            continue
        if "dptrain" not in file.stem:
            continue
        dptrain_list.append(file)
    if dptrain_list == []:
        error_msg = f"No dptrain_DEEPMDVERSION.json files found in {user_files_path}"
        raise FileNotFoundError(error_msg)

    for dptrain in dptrain_list:
        dptrain_dict = load_json_file(dptrain)
        if len(dptrain_dict["model"]["type_map"]) != len(properties_dict):
            error_msg = f"Number of types in {dptrain} does not match the number of types in properties file"
            raise ValueError(error_msg)
        for idx, type_dptrain in enumerate(dptrain_dict["model"]["type_map"]):
            if type_dptrain != properties_dict[idx + 1]["symbol"]:
                error_msg = f"Type {type_dptrain} not in properties or order is incorrect. Check your {dptrain}"
                raise ValueError(error_msg)


# TODO: Add tests for this function
@catch_errors_decorator
def check_typeraw_properties(type_raw_path, properties_dict):
    type_raw = np.genfromtxt(type_raw_path, dtype=np.int32)
    unique_types = np.unique(type_raw)
    # Add one because type.raw start from 0
    unique_types = unique_types + 1
    for type_val in unique_types:
        if type_val not in properties_dict:
            error_msg = f"Type {type_val} is not in properties file but is present in {type_raw_path}"
            raise ValueError(error_msg)
