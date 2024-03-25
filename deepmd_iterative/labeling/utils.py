"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/14

Functions
---------
generate_input_labeling_json(user_input_json: Dict, previous_json: Dict, default_input_json: Dict, merged_input_json: Dict, main_json: Dict) -> Dict
    Update and complete input JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

get_system_labeling(merged_input_json: Dict, system_auto_index: int) -> Tuple[float, float, int, int, int]
    Returns a tuple of system labeling parameters based on the input JSON and system number.
"""
# Standard library modules
import logging
from typing import Dict, List, Tuple

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator
from deepmd_iterative.common.json import convert_control_to_input


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
        If the labeling_program is not in valid_labeling_program.
    KeyError
        If a key is not found in any of the JSON dictionaries.
    TypeError
        If the value type is not str/int/float.
    ValueError
        If the length of the value list is not equal to the system count.
    """

    system_count = len(main_json.get("systems_auto", []))

    valid_labeling_program = ["cp2k", "orca"]

    previous_input_json = convert_control_to_input(previous_json, main_json)

    for key in [
        "labeling_program",
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
        elif key in previous_input_json:
            value = previous_input_json[key]
        elif key in default_input_json:
            value = default_input_json[key]
            default_used = True
        else:
            error_msg = f"'{key}' not found in any of the JSON dictionaries"
            raise KeyError(error_msg)

        # This is not system dependent and should be a string and should not change from previous iteration (but issue just a warning if it does).
        if key == "labeling_program":
            if key in previous_input_json and value != previous_input_json[key]:
                logging.critical(f"Labeling program changed from {previous_input_json[key]} to {value}!")

            if default_used:
                merged_input_json[key] = value
            else:
                if isinstance(value, str):
                    if value in valid_labeling_program:
                        merged_input_json[key] = value
                    else:
                        error_msg = f"Exploration type {key} is not known: use either {valid_labeling_program}."
                        raise ValueError(error_msg)
                else:
                    error_msg = f"Type mismatch: the type is '{type(value)}', but it should be '{type('string')}'."
                    raise TypeError(error_msg)
        else:
            # Everything else is system dependent so a list
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
) -> Tuple[str, float, float, int, int, int,]:
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
    Tuple[str, float, float, int, int, int]
        A tuple containing system labeling parameters:
        - labeling_program : str
            The labeling program.
        - walltime_first_job_h : float
            The walltime for the first job in hours.
        - walltime_second_job_h : float
            The walltime for the second job in hours.
        - nb_nodes : int
            The number of nodes.
        - nb_mpi_per_node : int
            The number of MPI processes per node.
        - nb_threads_per_mpi : int
            The number of threads per MPI process.
    """
    system_values = []
    for key in [ "labeling_program" ]:
        system_values.append(merged_input_json[key][system_auto_index])

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
