"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/31

generate_main_json(user_input_json: Dict, default_input_json: Dict) -> Tuple[Dict, Dict, str]
    A function to generate the main JSON by combining values from the user input JSON and the default JSON.
"""
# Standard library modules
import logging
from copy import deepcopy
from typing import Dict, Tuple, List

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


@catch_errors_decorator
def generate_main_json(
    user_input_json: Dict, default_input_json: Dict
) -> Tuple[Dict, Dict, str]:
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
    ValueError
        If a mandatory parameter is missing or if the value of 'exploration_type' is not 'lammps' or 'i-PI'.
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
    for key in ["systems_auto", "nnp_count", "exploration_type"]:
        if key == "systems_auto" and key not in user_input_json:
            error_msg = f"'systems_auto' not provided, it is mandatory. It should be a list of '{type(default_input_json['systems_auto'][0])}'."
            raise ValueError(error_msg)
        elif (
            key in user_input_json
            and key == "exploration_type"
            and not (user_input_json[key] == "lammps" or user_input_json[key] == "i-PI")
        ):
            error_msg = f"'{key}' should be either 'lammps' or 'i-PI'."
            raise ValueError(error_msg)
        else:
            main_json[key] = (
                user_input_json[key]
                if key in user_input_json
                else default_input_json[key]
            )
            merged_input_json[key] = (
                merged_input_json[key]
                if key in merged_input_json
                else default_input_json[key]
            )

    main_json["current_iteration"] = 0
    padded_curr_iter = str(main_json["current_iteration"]).zfill(3)

    main_json["systems_auto"] = {}
    for key in user_input_json["systems_auto"]:
        main_json["systems_auto"][key] = {}

    return main_json, merged_input_json, padded_curr_iter
