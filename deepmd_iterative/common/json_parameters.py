"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/16

Functions
---------
get_key_in_dict(key: str, input_json: Dict, previous_json: Dict, default_json: Dict) -> Any
    Get the value of the key from input JSON, previous JSON or default JSON, and validate its type.

get_machine_keyword(input_json: Dict, previous_json: Dict, default_json: Dict) -> Union[bool, str, List[str]]
    Get the value of the "user_machine_keyword" key from input JSON, previous JSON or default JSON, and validate its type.

set_main_config(user_config: Dict, default_config: Dict) -> Tuple[Dict, Dict, str]
    Set the main configuration (JSON) by validating the input JSON with a default JSON. If the input JSON is invalid, an error is raised and the script is terminated.
"""
# Standard library modules
import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


@catch_errors_decorator
def get_key_in_dict(
    key: str, input_json: Dict, previous_json: Dict, default_json: Dict
) -> Any:
    """
    Get the value of the key from input JSON, previous JSON or default JSON, and validate its type.

    Parameters
    ----------
    key : str
        The key to look up.
    input_json : dict
        The input JSON containing user-defined parameters.
    previous_json : dict
        The previous JSON containing previously defined parameters.
    default_json : dict
        The default JSON containing default parameters.

    Returns
    -------
    Any
        The value of the key, if it exists and is of the correct type.
    """

    # Check if the key is present in any of the JSON, and set the value accordingly.
    if key in input_json:
        if input_json[key] == "default" and key in default_json:
            value = default_json[key]
        else:
            value = input_json[key]
    elif key in previous_json:
        value = previous_json[key]
    elif key in default_json:
        value = default_json[key]
    else:
        # The key is not present in any of the JSON.
        error_msg = f"'{key}' not found in any JSON."
        raise KeyError(error_msg)

    # Check if the value is of the correct type.
    if not isinstance(value, type(default_json[key])):
        # The value is not of the correct type.
        error_msg = f"Wrong type: '{key}' is {type(value)}. '{key}' should be a {type(default_json[key])}."
        raise TypeError(error_msg)

    return value


@catch_errors_decorator
def get_machine_keyword(
    input_json: Dict, previous_json: Dict, default_json: Dict
) -> Union[bool, str, List[str]]:
    """
    Get the value of the "user_machine_keyword" key from input JSON, previous JSON or default JSON, and validate its type.

    Parameters
    ----------
    input_json : dict
        The input JSON containing user-defined parameters.
    previous_json : dict
        The previous JSON containing previously defined parameters.
    default_json : dict
        The default JSON containing default parameters.

    Returns
    -------
    Union[bool, str, List[str]]
        The value of the "user_machine_keyword" key, if it exists and is of the correct type.
    """

    # The key to look up in the JSON.
    key = "user_machine_keyword"

    # Check if the key is present in any of the JSON, and set the value accordingly.
    if key in input_json:
        if input_json[key] == "default" and key in default_json:
            value = default_json[key]
        else:
            value = input_json[key]
    elif key in previous_json:
        value = previous_json[key]
    elif key in default_json:
        value = default_json[key]
    else:
        # The key is not present in any of the JSON.
        error_msg = f"'{key}' not found in any JSON."
        raise KeyError(error_msg)

    # Check if the value is of the correct type.
    if (
        not isinstance(value, bool)
        and not (isinstance(value, str) and value != "")
        and not (
            isinstance(value, List)
            and [isinstance(value[_], str) for _ in range(len(value))]
        )
    ):
        # The value is not of the correct type.
        error_msg = f"""
            Wrong type: '{key}' is {type(value)}.\n
            '{key}' should be a boolean: false or true (Meaning it is deactivated)\n
            '{key}' should be a list of strings in the form: [\"project\", \"allocation\", \"arch_name\"]"""
        raise TypeError(error_msg)

    return value


@catch_errors_decorator
def set_main_config(user_config: Dict, default_config: Dict) -> Tuple[Dict, Dict, str]:
    """
    Set the main configuration (JSON) by validating the input JSON with a default JSON. If the input JSON is invalid, an error is raised and the script is terminated.

    Parameters
    ----------
    user_config : dict
        The user-defined configuration (JSON) containing user-defined parameters.
    default_config : dict
        The default configuration (JSON) containing default parameters.

    Returns
    -------
    Tuple[Dict, Dict, str]
        A tuple containing:
        - The main configuration (JSON)
        - The current configuration (JSON)
        - A message describing the validation result.

    Raises
    ------
    TypeError
        If the type of a user-defined parameter is not the same as the type of the default parameter.
    ValueError
        If a mandatory parameter is not provided or if the value of 'exploration_type' is not 'lammps' or 'i-PI'.
    """
    main_config = {}
    for key in default_config.keys():
        if key in user_config:
            if not isinstance(default_config[key], type(user_config[key])):
                error_msg = f"Wrong type: '{key}' is {type(user_config[key])}. It should be {type(default_config[key])}."
                raise TypeError(error_msg)
            if isinstance(user_config[key], List):
                for element in user_config[key]:
                    if not isinstance(element, type(default_config[key][0])):
                        error_msg = f"Wrong type: '{key}' is a list of {type(element)}. It should be a list of {type(default_config[key][0])}."
                        raise TypeError(error_msg)
    logging.debug(f"Type check complete")

    current_config = deepcopy(user_config)
    for key in ["system", "subsys_nr", "nnp_count", "exploration_type"]:
        if key == "system" and key not in user_config:
            error_msg = f"'{key}' is not provided, it is mandatory. It should of type {type(default_config[key])}."
            raise ValueError(error_msg)
        if key == "subsys_nr" and key not in user_config:
            error_msg = f"'subsys_nr' is not provided, it is mandatory. It should be a list of {type(default_config['subsys_nr'][0])}."
            raise ValueError(error_msg)
        elif (
            key in user_config
            and key == "exploration_type"
            and not (user_config[key] == "lammps" or user_config[key] == "i-PI")
        ):
            error_msg = f"'{key}' should be a string: lammps or i-PI."
            raise ValueError(error_msg)
        else:
            main_config[key] = (
                user_config[key] if key in user_config else default_config[key]
            )
            current_config[key] = (
                current_config[key] if key in current_config else default_config[key]
            )

    main_config["current_iteration"] = 0
    padded_curr_iter = str(main_config["current_iteration"]).zfill(3)

    main_config["subsys_nr"] = {}
    for key in user_config["subsys_nr"]:
        main_config["subsys_nr"][key] = {}

    return main_config, current_config, padded_curr_iter
