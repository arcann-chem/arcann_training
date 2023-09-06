"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/09/06

Functions
---------
generate_input_labeling_json(user_input_json: Dict, previous_json: Dict, default_input_json: Dict, merged_input_json: Dict, main_json: Dict) -> Dict
    Update and complete input JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

get_system_labeling(merged_input_json: Dict, system_auto_index: int) -> Tuple[float, float, int, int, int]
    Returns a tuple of system labeling parameters based on the input JSON and system number.
"""
# Standard library modules
from typing import Dict, List, Tuple

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


@catch_errors_decorator
def generate_input_labeling_json(
    user_input_json: Dict,
    previous_json: Dict,
    default_input_json: Dict,
    merged_input_json: Dict,
    main_json: Dict,
) -> Dict:
    """
    Update and complete input JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

    Parameters
    ----------
    user_input_json : dict
        The input JSON provided by the user, containing user-defined parameters.
    previous_json : dict
        The JSON from the previous iteration.
    default_input_json : dict
        The default input JSON containing default parameters.
    merged_input_json : dict
        The merged input JSON.
    main_json : dict
        The main JSON.

    Returns
    -------
    dict
        The updated merged input JSON.

    Raises
    ------
    ValueError
        If the exploration type is neither "lammps" nor "i-PI".
    KeyError
        If a key is not found in any of the JSON dictionaries.
    TypeError
        If the value type is not int/float or bool (for disturbed_start).
    ValueError
        If the length of the value list is not equal to the system count.
    """

    system_count = len(main_json.get("systems_auto", []))

    for key in [
        "walltime_first_job_h",
        "walltime_second_job_h",
        "nb_nodes",
        "nb_mpi_per_node",
        "nb_threads_per_mpi",
    ]:
        # Get the value
        default_used = False
        if key in user_input_json:
            if user_input_json[key] == "default" and key in default_input_json:
                value = default_input_json[key]
                default_used = True
            else:
                value = user_input_json[key]
        elif key in previous_json:
            value = previous_json[key]
        elif key in default_input_json:
            value = default_input_json[key]
            default_used = True
        else:
            error_msg = f"'{key}' not found in any of the JSON dictionaries"
            raise KeyError(error_msg)

        # Everything is system dependent so a list
        merged_input_json[key] = []

        # Default is used for the key
        if default_used:
            merged_input_json[key] = [value[0]] * system_count
        else:
            # Check if previous or user provided a list
            if isinstance(value, List):
                if len(value) == system_count:
                    for it_value in value:
                        if isinstance(it_value, (int, float)):
                            merged_input_json[key].append(it_value)
                        else:
                            error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}'."
                            raise TypeError(error_msg)
                else:
                    error_msg = f"Size mismatch: The length of the list should be '{system_count}' corresponding to the number of systems."
                    raise ValueError(error_msg)

            # If it is not a List
            elif isinstance(value, (int, float)):
                merged_input_json[key] = [value] * system_count
            else:
                error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}'."
                raise TypeError(error_msg)

    return merged_input_json


@catch_errors_decorator
def get_system_labeling(
    merged_input_json: Dict, system_auto_index: int
) -> Tuple[float, float, int, int, int,]:
    """
    Return a tuple of system labeling parameters based on the input JSON and system index.

    Parameters
    ----------
    merged_input_json : Dict[str, Any]
        A dictionary containing input parameters.
    system_auto_index : int
        An integer representing the system index.

    Returns
    -------
    Tuple[float, float, int, int, int]
        A tuple containing system labeling parameters:
        - walltime_first_job_h : float
            X
        - walltime_second_job_h : float
            X
        - nb_nodes : int
            X
        - nb_mpi_per_node : int
            X
        - nb_threads_per_mpi : int
            X
    """
    system_values = []

    for key in [
        "walltime_first_job_h",
        "walltime_second_job_h",
    ]:
        system_values.append(float(merged_input_json[key][system_auto_index]))

    for key in [
        "nb_nodes",
        "nb_mpi_per_node",
        "nb_threads_per_mpi",
    ]:
        system_values.append(int(merged_input_json[key][system_auto_index]))
    return tuple(system_values)
