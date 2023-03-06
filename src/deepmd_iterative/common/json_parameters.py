from pathlib import Path
from typing import Dict, List, Tuple, Union
import logging
import sys
from copy import deepcopy

from deepmd_iterative.common.json import load_json_file

# Used in initialization - init
def set_config_json(
    input_json: Dict,
    default_json: Dict
) -> Tuple[Dict, Dict, int]:
    """
    This function sets a configuration JSON object by validating the input JSON with a default JSON object.
    If the input JSON object is invalid, it throws an error and terminates the script.

    Args:
    - input_json (Dict): The input JSON object containing user-defined parameters.
    - default_json (Dict): The input JSON object containing default parameters.

    Returns:
    - A tuple containing the configuration JSON object, a input JSON completed with defaults and an integer representing
    the current iteration padded.
    """
    config_json = {}
    for key in default_json.keys():
        if key in input_json:
            if not isinstance(default_json[key], type(input_json[key])):
                logging.error(f"Wrong type: '{key}' is {type(input_json[key])}")
                logging.error(f"It should be {type(default_json[key])}")
                sys.exit(1)
            if isinstance(input_json[key], List):
                for element in input_json[key]:
                    if not isinstance(element, type(default_json[key][0])):
                        logging.error(f"Wrong type: '{key}' is a {list} of {type(element)}")
                        logging.error(f"It should be a {list} of {type(default_json[key][0])}")
                        sys.exit(1)
    logging.debug(f"Type check complete")

    new_input_json = deepcopy(input_json)
    for key in ["system", "subsys_nr", "nb_nnp", "exploration_type"]:
        if key == "system" and key not in input_json:
            logging.error(f"{key} is not provided, it is mandatory.")
            logging.error(f"It should of type {type(default_json[key])}")
            sys.exit(1)
        if key == "subsys_nr" and key not in input_json:
            logging.error(f"subsys_nr is not provided, it is mandatory.")
            logging.error(f"It should be a {list} of {type(default_json['subsys_nr'][0])}")
            sys.exit(1)
        elif key in input_json and key == "exploration_type" and not ( input_json[key] == "lammps" or input_json[key] == "i-PI"):
            logging.error(f"{key} should be a {type(str)}: lammps or i-PI.")
            sys.exit(1)
        else:
            config_json[key] = input_json[key] if key in input_json else default_json[key]
            new_input_json[key] = new_input_json[key] if key in new_input_json else default_json[key]

    config_json["current_iteration"] = 0
    padded_curr_iter = str(config_json["current_iteration"]).zfill(3)

    config_json["subsys_nr"] = {}
    for key in input_json['subsys_nr']:
        config_json["subsys_nr"][key] = {}

    return config_json, new_input_json, padded_curr_iter


def get_machine_keyword(
    input_json: Dict,
    previous_json: Dict,
    default_json: Dict
)-> Union[bool,str,List[str]]:
    """
    Get the value of the "user_machine_keyword" key from input_json, previous_json, or default_json, and validate its type.

    Args:
    input_json (Dict): The input JSON object containing user-defined parameters.
    previous_json (Dict): The previous JSON object containing user-defined parameters..
    default_json (Dict): The input JSON object containing default parameters.

    Returns:
    Union[bool, str, List[str]]: The value of the "user_machine_keyword" key, if it exists and is of the correct type.

    Raises:
    TypeError: If the value of the "user_machine_keyword" key is not of the correct type.
    """

    # The key to look up in the dictionaries.
    key = "user_machine_keyword"

    # Check if the key is present in any of the dictionaries, and set the value accordingly.
    if key in input_json:
        value = input_json[key]
    elif key in previous_json:
        value = previous_json[key]
    elif key in default_json:
        value = default_json[key]
    else:
        # The key is not present in any of the dictionaries.
        logging.error(f'"{key}" not found in input or default JSON')
        sys.exit(1)

    # Check if the value is of the correct type.
    if (
        not isinstance(value, bool) and
        not (isinstance(value, str) and value != "") and
        not (isinstance(value, List) and [isinstance(value[_], str) for _ in range(len(value))])
        ):
        # The value is not of the correct type.
        logging.error(f'Wrong type: "{key}" is {type(value)}')
        logging.error(f'"{key}" should be a boolean: false or true (Meaning it is deactivated)')
        logging.error(f'"{key}" should be a list of strings in the form: ["project", "allocation", "arch_name"]')
        logging.error(f'"{key}" should be a non-empty string in the form: "shortcut"')
        logging.error(f"Aborting...")
        sys.exit(1)

    return value

# def get_key_in_dict(
#     key: str,
#     input_json: Dict,
#     previous_json: Dict,
#     default_json: Dict
# )-> Union[bool,str,List[str]]:
#     """
#     Get the value of the "user_machine_keyword" key from input_json, previous_json, or default_json, and validate its type.

#     Args:
#     key (str): The key to look up.
#     input_json (Dict): The input JSON object containing user-defined parameters.
#     previous_json (Dict): The previous JSON object containing user-defined parameters..
#     default_json (Dict): The input JSON object containing default parameters.

#     Returns:
#     Union[bool, str, List[str]]: The value of the "user_machine_keyword" key, if it exists and is of the correct type.

#     Raises:
#     TypeError: If the value of the "user_machine_keyword" key is not of the correct type.
#     """

#     # The key to look up in the dictionaries.

#     # Check if the key is present in any of the dictionaries, and set the value accordingly.
#     if key in input_json:
#         value = input_json[key]
#     elif key in previous_json:
#         value = previous_json[key]
#     elif key in default_json:
#         value = default_json[key]
#     else:
#         # The key is not present in any of the dictionaries.
#         logging.error(f'"{key}" not found in input or default JSON')
#         sys.exit(1)

#     # Check if the value is of the correct type.
#     if (not isinstance(value, type(default_json[key]))):
#         # The value is not of the correct type.
#         logging.error(f'Wrong type: "{key}" is {type(value)}')
#         logging.error(f'"{key}" should be a {type(default_json[key])}')
#         logging.error(f"Aborting...")
#         sys.exit(1)

#     return value



def set_training_json(
    control_path: Path,
    padded_curr_iter: str,
    input_json: Dict,
    previous_json: Dict,
    default_json: Dict,
    new_input_json: Dict
) -> Tuple[Dict, Dict]:

    # Load or create the training JSON file
    training_json = load_json_file(
        (control_path / f"training_{padded_curr_iter}.json"),
        abort_on_error=False,
    )
    if not training_json:
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
        if key in input_json:
            training_json[key] = input_json[key]
        elif key in previous_json:
            training_json[key] = previous_json[key]
            new_input_json[key] = previous_json[key]
        elif key in default_json:
            training_json[key] = default_json[key]
            new_input_json[key] = default_json[key]
        else:
            # The key is not present in any of the dictionaries.
            logging.error(f'"{key}" not found in input or default JSON')
            sys.exit(1)
        if not isinstance(training_json[key], type(default_json[key])):
            logging.error(f"Wrong type: '{key}' is a {type(training_json[key])}")
            logging.error(f"It should be a {type(default_json[key])}")
            sys.exit(1)

    return training_json, new_input_json