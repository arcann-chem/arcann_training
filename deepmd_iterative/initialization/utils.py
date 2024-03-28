"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/28

generate_main_json(user_input_json: Dict, default_input_json: Dict) -> Tuple[Dict, Dict, str]
    A function to generate the main JSON by combining values from the user input JSON and the default JSON.
"""

# Standard library modules
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple, List

# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator
from deepmd_iterative.common.lammps import read_lammps_data
from deepmd_iterative.common.json import load_json_file


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
    main_json = {}
    for key in default_input_json.keys():
        if key in user_input_json:
            if not isinstance(default_input_json[key], type(user_input_json[key])):
                error_msg = f"Type mismatch: '{key}' has type '{type(user_input_json[key])}', but should have type '{type(default_input_json[key])}'."
                raise TypeError(error_msg)
            if isinstance(user_input_json[key], List):
                for element in user_input_json[key]:
                    if not isinstance(element, type(default_input_json[key][0])):
                        error_msg = f"Type mismatch: Elements in '{key}' are of type '{type(element)}', but they should be of type '{type(default_input_json[key][0])}'."
                        raise TypeError(error_msg)
    logging.debug(f"Type check complete")

    merged_input_json = deepcopy(user_input_json)
    for key in ["systems_auto", "nnp_count"]:
        if key == "systems_auto" and key not in user_input_json:
            error_msg = f"'systems_auto' not provided, it is mandatory. It should be a list of '{type(default_input_json['systems_auto'][0])}'."
            raise ValueError(error_msg)
        else:
            main_json[key] = user_input_json[key] if key in user_input_json else default_input_json[key]
            merged_input_json[key] = merged_input_json[key] if key in merged_input_json else default_input_json[key]

    main_json["current_iteration"] = 0

    main_json["systems_auto"] = {}
    for idx, key in enumerate(user_input_json["systems_auto"]):
        main_json["systems_auto"][key] = {}
        main_json["systems_auto"][key]["index"] = idx

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
    combined = {}
    combined = {}
    for symbol, type_id in types.items():
        combined[type_id] = {"symbol": symbol, "mass": masses[symbol]}

    return combined


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
