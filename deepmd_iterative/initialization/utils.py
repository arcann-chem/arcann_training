"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/24

set_main_config(user_config: Dict, default_config: Dict) -> Tuple[Dict, Dict, str]
    Set the main configuration (JSON) by validating the input JSON with a default JSON. If the input JSON is invalid, an error is raised and the script is terminated.
"""
# Standard library modules
import logging
from copy import deepcopy
from typing import Dict, Tuple, List

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


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
                error_msg = f"Wrong type: '{key}' is '{type(user_config[key])}'. It should be '{type(default_config[key])}'"
                raise TypeError(error_msg)
            if isinstance(user_config[key], List):
                for element in user_config[key]:
                    if not isinstance(element, type(default_config[key][0])):
                        error_msg = f"Wrong type: '{key}' is a list of '{type(element)}'. It should be a list of '{type(default_config[key][0])}'"
                        raise TypeError(error_msg)
    logging.debug(f"Type check complete")

    current_config = deepcopy(user_config)
    for key in ["systems_auto", "nnp_count", "exploration_type"]:
        if key == "systems_auto" and key not in user_config:
            error_msg = f"'systems_auto' is not provided, it is mandatory. It should be a list of '{type(default_config['systems_auto'][0])}'"
            raise ValueError(error_msg)
        elif (
            key in user_config
            and key == "exploration_type"
            and not (user_config[key] == "lammps" or user_config[key] == "i-PI")
        ):
            error_msg = f"'{key}' should be a string: 'lammps' or 'i-PI'"
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

    main_config["systems_auto"] = {}
    for key in user_config["systems_auto"]:
        main_config["systems_auto"][key] = {}

    return main_config, current_config, padded_curr_iter
