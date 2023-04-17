"""
Created: 2023/01/01
Last modified: 2023/04/17
"""
# Standard library modules
import logging
import sys
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
        logging.error(f'"{key}" not found in any JSON')
        logging.error(f"Aborting...")
        sys.exit(1)

    # Check if the value is of the correct type.
    if not isinstance(value, type(default_json[key])):
        # The value is not of the correct type.
        logging.error(f'Wrong type: "{key}" is {type(value)}')
        logging.error(f'"{key}" should be a {type(default_json[key])}')
        logging.error(f"Aborting...")
        sys.exit(1)

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
        logging.error(f'"{key}" not found in any JSON')
        logging.error(f"Aborting...")
        sys.exit(1)

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
        logging.error(f'Wrong type: "{key}" is {type(value)}')
        logging.error(
            f'"{key}" should be a boolean: false or true (Meaning it is deactivated)'
        )
        logging.error(
            f'"{key}" should be a list of strings in the form: ["project", "allocation", "arch_name"]'
        )
        logging.error(f'"{key}" should be a non-empty string in the form: "shortcut"')
        logging.error(f"Aborting...")
        sys.exit(1)

    return value


# Used in initialization - init
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
            error_msg = f"{key} is not provided, it is mandatory. It should of type {type(default_config[key])}."
            raise ValueError(error_msg)
        if key == "subsys_nr" and key not in user_config:
            error_msg = f"subsys_nr is not provided, it is mandatory. It should be a list of {type(default_config['subsys_nr'][0])}."
            raise ValueError(error_msg)
        elif (
            key in user_config
            and key == "exploration_type"
            and not (user_config[key] == "lammps" or user_config[key] == "i-PI")
        ):
            error_msg = f"{key} should be a string: lammps or i-PI."
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


# Used in training
def set_training_config(
    user_config: Dict,
    previous_config: Dict,
    default_config: Dict,
    current_config: Dict,
) -> Tuple[Dict, Dict]:
    """
    Creates the training config (JSON) and updates the current config (JSON) using user, previous and default configs (JSON)

    Args:
    user_config (Dict): The user config (JSON) containing user-defined parameters.
    previous_config (Dict): The previous config (JSON) containing previously defined parameters.
    default_config (Dict): The default config (JSON) containing default parameters.
    current_config (Dict): The current config (JSON) containing the current parameters.

    Returns:
    Tuple(Dict, Dict):
        - the training config (JSON)
        - the current config (JSON)
    """

    training_json = {}

    # Update the training JSON configuration with values from the input JSON files
    for key in [
        "user_machine_keyword",
        "job_email",
        "use_initial_datasets",
        "use_extra_datasets",
        "deepmd_model_version",
        "deepmd_model_type_descriptor",
        "start_lr",
        "stop_lr",
        "decay_rate",
        "decay_steps",
        "decay_steps_fixed",
        "numb_steps",
        "numb_test",
    ]:
        # Check if the key is present in any of the dictionaries, and set the value accordingly.
        if key in user_config:
            if user_config[key] == "default" and key in default_config:
                training_json[key] = default_config[key]
                current_config[key] = default_config[key]
            else:
                training_json[key] = user_config[key]
        elif key in previous_config:
            training_json[key] = previous_config[key]
            current_config[key] = previous_config[key]
        elif key in default_config:
            training_json[key] = default_config[key]
            current_config[key] = default_config[key]
        else:
            # The key is not present in any of the dictionaries.
            logging.error(f'"{key}" not found in any JSON')
            logging.error(f"Aborting...")
            sys.exit(1)
        if not isinstance(training_json[key], type(default_config[key])):
            logging.error(f"Wrong type: '{key}' is a {type(training_json[key])}")
            logging.error(f"It should be a {type(default_config[key])}")
            logging.error(f"Aborting...")
            sys.exit(1)

    return training_json, current_config


###########################################
def set_new_input_explor_json(
    input_json: Dict,
    previous_json: Dict,
    default_json: Dict,
    new_input_json: Dict,
    config_json: Dict,
) -> Dict:
    """
    Updates the training JSON with input JSON, previous JSON and default JSON.

    Args:
    input_json (Dict): The input JSON containing user-defined parameters.
    previous_json (Dict): The previous JSON containing previously defined parameters.
    default_json (Dict): The default JSON containing default parameters.
    new_input_json (Dict): The inputJSON udpated/completed with previous/defaults.

    Returns:
    Dict: an input JSON udpated/completed with previous/defaults
    """

    if config_json["exploration_type"] == "lammps":
        exploration_dep = 0
    elif config_json["exploration_type"] == "i-PI":
        exploration_dep = 1
    else:
        logging.error(f"{config_json['exploration_type']} is not known")
        logging.error(f"Aborting...")
        sys.exit(1)

    subsys_count = len(config_json["subsys_nr"])

    for key in [
        "timestep_ps",
        "temperature_K",
        "exp_time_ps",
        "max_exp_time_ps",
        "job_walltime_h",
        "init_exp_time_ps",
        "init_job_walltime_h",
        "disturbed_start",
        "print_mult",
    ]:

        # Get the value
        default_used = False
        if key in input_json:
            if input_json[key] == "default" and key in default_json:
                value = default_json[key]
                default_used = True
            else:
                value = input_json[key]
        elif key in previous_json:
            value = previous_json[key]
        elif key in default_json:
            value = default_json[key]
            default_used = True
        else:
            logging.error(f'"{key}" not found in any JSON')
            logging.error(f"Aborting...")
            sys.exit(1)

        # Everything is subsys dependent so a list
        new_input_json[key] = []

        # Default is used for the key
        if default_used:
            new_input_json[key] = [value[0][exploration_dep]] * subsys_count
        else:
            # Check if previous or user provided a list
            if isinstance(value, List):
                if len(value) == subsys_count:
                    for it_value in value:
                        if (
                            isinstance(it_value, (int, float))
                            and key != "disturbed_start"
                        ) or (
                            isinstance(it_value, (bool)) and key == "disturbed_start"
                        ):
                            new_input_json[key].append(it_value)
                        else:
                            logging.error(
                                f"Wrong type: the type is {type(it_value)} it should be int/float or bool (for disturbed_start)"
                            )
                            logging.error(f"Aborting...")
                            sys.exit(1)
                else:
                    logging.error(
                        f"Wrong size: The length of the list should be {subsys_count} [Subsys]"
                    )
                    logging.error(f"Aborting...")
                    sys.exit(1)

            # If it is not a List
            elif (isinstance(value, (int, float)) and key != "disturbed_start") or (
                isinstance(value, (bool)) and key == "disturbed_start"
            ):
                new_input_json[key] = [value] * subsys_count
            else:
                logging.error(
                    f"Wrong type: the type is {type(value)} it should be int/float or bool (for disturbed_start)"
                )
                logging.error(f"Aborting...")
                sys.exit(1)

    return new_input_json


@catch_errors_decorator
def set_new_input_explordevi_json(
    input_json: Dict,
    previous_json: Dict,
    default_json: Dict,
    new_input_json: Dict,
    config_json: Dict,
) -> Dict:
    """
    Updates the training JSON with input JSON, previous JSON and default JSON.

    Args:
    input_json (Dict): The input JSON containing user-defined parameters.
    previous_json (Dict): The previous JSON containing previously defined parameters.
    default_json (Dict): The default JSON containing default parameters.
    new_input_json (Dict): The inputJSON udpated/completed with previous/defaults.

    Returns:
    Dict: an input JSON udpated/completed with previous/defaults
    """

    if config_json["exploration_type"] == "lammps":
        exploration_dep = 0
    elif config_json["exploration_type"] == "i-PI":
        exploration_dep = 1
    else:
        logging.error(f"{config_json['exploration_type']} is not known")
        logging.error(f"Aborting...")
        sys.exit(1)

    subsys_count = len(config_json["subsys_nr"])

    for key in [
        "max_candidates",
        "sigma_low",
        "sigma_high",
        "sigma_high_limit",
        "ignore_first_x_ps",
    ]:

        # Get the value
        default_used = False
        if key in input_json:
            if input_json[key] == "default" and key in default_json:
                value = default_json[key]
                default_used = True
            else:
                value = input_json[key]
        elif key in previous_json:
            value = previous_json[key]
        elif key in default_json:
            value = default_json[key]
            default_used = True
        else:
            logging.error(f'"{key}" not found in any JSON')
            logging.error(f"Aborting...")
            sys.exit(1)

        # Everything is subsys dependent so a list
        new_input_json[key] = []

        # Default is used for the key
        if default_used:
            new_input_json[key] = [value[0][exploration_dep]] * subsys_count
        else:
            # Check if previous or user provided a list
            if isinstance(value, List):
                if len(value) == subsys_count:
                    for it_value in value:
                        if isinstance(it_value, (int, float)):
                            new_input_json[key].append(it_value)
                        else:
                            logging.error(
                                f"Wrong type: the type is {type(it_value)} it should be int/float."
                            )
                            logging.error(f"Aborting...")
                            sys.exit(1)
                else:
                    logging.error(
                        f"Wrong size: The length of the list should be {subsys_count} [Subsys]."
                    )
                    logging.error(f"Aborting...")
                    sys.exit(1)

            # If it is not a List
            elif isinstance(value, (int, float)):
                new_input_json[key] = [value] * subsys_count
            else:
                logging.error(
                    f"Wrong type: the type is {type(value)} it should be int/float."
                )
                logging.error(f"Aborting...")
                sys.exit(1)

    return new_input_json
