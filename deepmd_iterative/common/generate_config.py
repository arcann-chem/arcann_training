"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/22
"""

# Standard library modules
from typing import Dict, Tuple

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator
from deepmd_iterative.common.json import read_key_input_json


@catch_errors_decorator
def set_subsys_params_deviation(
    input_json: Dict,
    new_input_json: Dict,
    default_input_json: Dict,
    config_json: Dict,
    step_name: str,
    default_present: bool,
    system_auto_index: int,
) -> Tuple[int, float, float, float, float]:
    """
    Sets candidate selection parameters for a specific system.

    Args:
        input_json (Dict): A dictionary containing the input JSON data.
        new_input_json (Dict): A dictionary containing the new input JSON data.
        default_input_json (Dict): A dictionary containing the default input JSON data.
        config_json (Dict): A dictionary containing the config JSON data.
        step_name (str): A string representing the name of the step.
        default_present (bool): A boolean indicating whether the default input JSON data is present.
        system_auto_index (int): An integer representing the system index.

    Returns:
        A tuple containing the following candidate selection parameters for the specified system:
        - maximum number of candidates (int)
        - lower limit for sigma (float)
        - upper limit for sigma (float)
        - maximum upper limit of sigma (float)
        - amount of time to ignore at start of simulation (float)
    """

    max_candidates = read_key_input_json(
        input_json,
        new_input_json,
        "max_candidates",
        default_input_json,
        step_name,
        default_present,
        system_index=system_auto_index,
        system_count=len(config_json["system_auto"]),
    )

    sigma_low = read_key_input_json(
        input_json,
        new_input_json,
        "sigma_low",
        default_input_json,
        step_name,
        default_present,
        system_index=system_auto_index,
        system_count=len(config_json["system_auto"]),
    )

    sigma_high = read_key_input_json(
        input_json,
        new_input_json,
        "sigma_high",
        default_input_json,
        step_name,
        default_present,
        system_index=system_auto_index,
        system_count=len(config_json["system_auto"]),
    )

    sigma_high_limit = read_key_input_json(
        input_json,
        new_input_json,
        "sigma_high_limit",
        default_input_json,
        step_name,
        default_present,
        system_index=system_auto_index,
        system_count=len(config_json["system_auto"]),
    )

    ignore_first_x_ps = read_key_input_json(
        input_json,
        new_input_json,
        "ignore_first_x_ps",
        default_input_json,
        step_name,
        default_present,
        system_index=system_auto_index,
        system_count=len(config_json["system_auto"]),
    )

    return max_candidates, sigma_low, sigma_high, sigma_high_limit, ignore_first_x_ps
