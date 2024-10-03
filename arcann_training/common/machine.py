"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/07/14

The machine module provides functions for machine operations.

Functions
---------
get_host_name() -> str
    A function to returns the fully-qualified hostname of the current machine.

assert_same_machine(expected_machine: str, machine_config: Dict) -> None
    A function to ceck if the machine name in the provided dictionary matches the expected machine name.

get_machine_keyword(input_json: Dict, previous_json: Dict, default_json: Dict) -> Union[bool, str, List[str]]
    Get the value of the "user_machine_keyword" key from input JSON, previous JSON or default JSON, and validate its type.

get_machine_config_files(deepmd_iterative_path: Path, training_path: Path) -> List[Dict]
    A function to returns a list of dictionaries containing machine configurations for all machines found in the given paths.

get_machine_from_configs(machine_configs: List[Dict], machine_short_name: str = "") -> str
    A function to returns the name of the machine that matches the current hostname or the input machine name.

get_machine_spec_for_step(deepmd_iterative_path: Path, training_path: Path, step: str, input_machine_shortname: str = None, user_machine_keyword: Union[str, List[str]] = None, check_only: bool = False) -> Tuple[str, Dict[str, Any], str, str]
    A function to returns the machine specification for a given step and machine.
"""

# TODO: Homogenize the docstrings for this module

# Standard library modules
import socket
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Local imports
from arcann_training.common.utils import catch_errors_decorator
from arcann_training.common.json import load_json_file


# Unittested
@catch_errors_decorator
def get_host_name() -> str:
    """
    Return the fully-qualified hostname of the current machine.

    This function first gets the hostname of the current machine using the 'socket.gethostname()' function. If the hostname
    contains a period, it is already fully-qualified and can be returned immediately. Otherwise, the function uses the
    'socket.gethostbyaddr()' function to look up the fully-qualified hostname.

    Parameters
    ----------
    None

    Returns
    -------
    str
        The fully-qualified hostname of the current machine.

    Raises
    ------
    None
    """
    hostname = socket.gethostname()
    if "." in hostname:
        return hostname  # Hostname is already fully-qualified
    else:
        try:
            hostname = socket.gethostbyaddr(hostname)[0]
            return hostname
        except socket.timeout:
            return hostname


# Unittested
@catch_errors_decorator
def assert_same_machine(
    user_machine_keyword: str, control_json: Dict, step: str
) -> None:
    """
    Verify that the provided machine keyword matches the expected machine keyword in the control JSON.

    This function is used for preparation and launch processes. It checks if the provided machine keyword
    matches the one specified in the control JSON. If they do not match, a ValueError is raised with an
    error message, and the execution is aborted.

    Parameters
    ----------
    user_machine_keyword : str
        The machine keyword to be checked.

    control_json : Dict
        The control JSON dictionary containing the expected machine keyword.

    step : str
        A string representing the step or stage identifier used to retrieve the expected machine keyword
        from the control JSON.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the provided machine keyword does not match the expected machine keyword specified in the control JSON.
    """
    # Check if the provided machine name matches the expected machine name
    if control_json.get(f"user_machine_keyword_{step}") != user_machine_keyword:
        # If not, log an error message and abort the execution
        error_msg = f"Provided machine '{user_machine_keyword}' does not match expected machine '{control_json.get(f'user_machine_keyword_{step}')}'"
        raise ValueError(error_msg)


# Unittested
@catch_errors_decorator
def get_machine_keyword(
    input_json: Dict, previous_json: Dict, default_json: Dict, step: str
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
    step : str
        The step for which the machine keyword is being retrieved.

    Returns
    -------
    Union[bool, str, List[str]]
        The value of the "user_machine_keyword" key, if it exists and is of the correct type.

    Raises
    ------
    KeyError
        If the key is not found in any of the provided JSON.
    TypeError
        If the value is not of the correct type.

    Notes
    -----
    The function checks if the key is present in any of the JSON dictionaries (input_json, previous_json, default_json).
    If the key is found, the value is retrieved and validated to ensure it is of the correct type.
    The correct types for the value are:
    - bool: False or True (indicating if the keyword is deactivated or not)
    - str: a keyword
    - List[str]: a list of keywords in the form ["project", "allocation", "arch_name"]
    """

    # The key to look up in the JSON.
    key = f"user_machine_keyword_{step}"

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
        error_msg = f"'{key}' not found in any JSON provided"
        raise KeyError(error_msg)

    # Check if the value is of the correct type.
    if (
        not isinstance(value, bool)
        and not (isinstance(value, str) and value != "")
        and not (
            isinstance(value, List)
            and all([isinstance(value[_], str) for _ in range(len(value))])
        )
    ):
        # The value is not of the correct type.
        error_msg = f"""
            Wrong type: '{key}' is '{type(value)}'.\n
            '{key}' should be a '{type(True)}': False or True (Meaning it is deactivated)\n
            '{key}' should be a '{type(str)}': a keyword\n
            '{key}' should be a {type([])} of {type("")} in the form: [\"project\", \"allocation\", \"arch_name\"]"""
        raise TypeError(error_msg)

    return value


# TODO: Add tests for this function
@catch_errors_decorator
def get_machine_config_files(
    deepmd_iterative_path: Path, training_path: Path
) -> List[Dict]:
    """
    Return a list of dictionaries containing machine configurations for all machines found in the given paths.

    Parameters
    ----------
    deepmd_iterative_path : Path
        The path to the 'deepmd_iterative' directory.
    training_path : Path
        The path to the training directory.

    Returns
    -------
    List[Dict]
        A list of dictionaries, each containing the contents of a 'machine.json' file.

    Raises
    ------
    FileNotFoundError
        If no 'machine.json' file is found in the given directories.
    """
    machine_configs = []

    # Check for 'machine.json' file in the training directory.
    training_config_path = training_path / "user_files" / "machine.json"
    if training_config_path.is_file():
        machine_configs.append(load_json_file(training_config_path))

    # Check for 'machine.json' file in the deepmd_iterative directory.
    # default_config_path = deepmd_iterative_path / "assets" / "machine.json"
    # if default_config_path.is_file():
    #     machine_configs.append(load_json_file(default_config_path))

    # If no 'machine.json' file is found, raise a FileNotFoundError.
    if not machine_configs:
        error_msg = f"No 'machine.json' file found. Please check the installation"
        raise FileNotFoundError(error_msg)

    return machine_configs


# Unittested
@catch_errors_decorator
def get_machine_from_configs(
    machine_configs: List[Dict], machine_short_name: str = ""
) -> str:
    """
    Given a list of machine configuration dictionaries and an optional input machine name,
    return the name of the machine that matches the current hostname or the input machine name.

    Parameters
    ----------
    machine_configs : List[Dict]
        A list of machine configuration dictionaries.
    machine_short_name : str, optional
        The name of the machine to use. If not provided, the machine with a matching hostname will be used.

    Returns
    -------
    str
        The name of the machine that matches the current hostname or input machine name.

    Raises
    ------
    ValueError
        If no machine specification is found that matches the current hostname or input machine name.
    """
    if not machine_short_name:
        machine_hostname = get_host_name()
        for machine_config in machine_configs:
            for machine_short_name in machine_config.keys():
                if machine_config[machine_short_name]["hostname"] in machine_hostname:
                    return machine_short_name
    else:
        for machine_config in machine_configs:
            if machine_short_name in machine_config:
                return machine_short_name

    error_msg = f"No matching machine found for hostname '{get_host_name()}' and no input machine specified"
    raise ValueError(error_msg)


# TODO: Add tests for this function
@catch_errors_decorator
def get_machine_spec_for_step(
    deepmd_iterative_path: Path,
    training_path: Path,
    step: str,
    input_machine_shortname: str = None,
    user_machine_keyword: Union[str, List[str]] = None,
    check_only: bool = False,
) -> Tuple[str, Dict[str, Any], str, str, str, str, str, Dict[str, Any]]:
    """
    Return the machine specification for a given step and machine.

    Parameters
    ----------
    deepmd_iterative_path : Path
        The path to the DeepMD-Iterative root directory.
    training_path : Path
        The path to the training directory.
    step : str
        The name of the step for which to get the machine specification.
    input_machine_shortname : str, optional
        The short name of the machine for which to get the specification. Defaults to None.
    user_machine_keyword : Union[str, List[str]], optional
        A keyword or list of keywords to use when searching for a matching configuration. Defaults to None.
    check_only : bool, optional
        Whether to only check for a matching configuration without returning the machine specification. Defaults to False.

    Returns
    -------
    Tuple[str, Dict[str, Any], str, str, str, str, str, Dict[str, Any]]
        A tuple containing the following elements:
            - machine_shortname: The short name of the machine.
            - machine_walltime_format: The walltime format of the machine.
            - machine_job_scheduler: The job scheduler for the machine.
            - machine_launch_command: The launch command for the machine.
            - machine_max_jobs: The maximum number of jobs that can be run on the machine.
            - machine_max_array_size: The maximum array size that can be run on the machine.
            - machine_keyword: The keyword (provided, or found) for the machine specification
            - machine_spec: The machine specification as a dictionary.

    Raises
    ------
    ValueError
        If no matching configuration is found for the given input parameters.
    """
    # Get a list of all machine configuration files
    machine_configs = get_machine_config_files(deepmd_iterative_path, training_path)

    # Get the short name of the machine to use
    machine_shortname = get_machine_from_configs(
        machine_configs, input_machine_shortname
    )

    # If check_only is True, return an empty machine specification
    if check_only:
        return machine_shortname, {}, "", "", "", "", "", {}

    # Iterate over all machine configurations
    for config in machine_configs:
        # Iterate over all keys in the configuration for the selected machine
        for config_key, config_data in config.get(machine_shortname, {}).items():
            # Skip keys that are not relevant to the machine specification
            if config_key not in [
                "hostname",
                "walltime_format",
                "job_scheduler",
                "launch_command",
                "max_jobs",
                "max_array_size",
            ]:
                # Check if the current keyword matches the user keyword
                if (
                    isinstance(user_machine_keyword, str)
                    and user_machine_keyword == config_key
                ) or (
                    isinstance(user_machine_keyword, list)
                    and len(user_machine_keyword) == 3
                    and user_machine_keyword[0] == config_data.get("project_name")
                    and user_machine_keyword[1] == config_data.get("allocation_name")
                    and user_machine_keyword[2] == config_data.get("arch_name")
                ):
                    # Check if the step is valid for the current configuration
                    if step in config_data.get("valid_for", []):
                        # Return the machine specification
                        return (
                            machine_shortname,
                            config[machine_shortname]["walltime_format"],
                            config[machine_shortname]["job_scheduler"],
                            config[machine_shortname]["launch_command"],
                            config[machine_shortname]["max_jobs"],
                            config[machine_shortname]["max_array_size"],
                            config_key,
                            config_data,
                        )
                elif user_machine_keyword is None:
                    if step in config_data.get("default", []):
                        return (
                            machine_shortname,
                            config[machine_shortname]["walltime_format"],
                            config[machine_shortname]["job_scheduler"],
                            config[machine_shortname]["launch_command"],
                            config[machine_shortname]["max_jobs"],
                            config[machine_shortname]["max_array_size"],
                            config_key,
                            config_data,
                        )

    # If no matching configuration was found, return an error
    if user_machine_keyword is not None and not (
        isinstance(user_machine_keyword, str)
        or (isinstance(user_machine_keyword, list) and len(user_machine_keyword) == 3)
    ):
        error_msg = f"Invalid 'user_machine_keyword'. Please provide either a '{type('')}' or a '{type([])}' of 3 '{type('')}"
    elif user_machine_keyword is not None and (
        isinstance(user_machine_keyword, list) and len(user_machine_keyword) == 3
    ):
        error_msg = f"User keyword '{user_machine_keyword}' not found in any configuration files"
    elif not machine_configs:
        error_msg = "No machine configuration files found"
    elif user_machine_keyword is not None and not any(
        user_machine_keyword in config for config in machine_configs
    ):
        error_msg = f"User keyword '{user_machine_keyword}' not found in any configuration files"
    elif input_machine_shortname is not None and machine_shortname not in [
        config.get("name") for config in machine_configs
    ]:
        error_msg = (
            f"No configuration found for input machine '{input_machine_shortname}'"
        )
    elif user_machine_keyword is not None and not any(
        user_machine_keyword in config for config in machine_configs
    ):
        error_msg = f"User keyword '{user_machine_keyword}' not found in any configuration files"
    else:
        error_msg = f"No default configuration found for step '{step}' and machine '{machine_shortname}'"
    raise ValueError(error_msg)
