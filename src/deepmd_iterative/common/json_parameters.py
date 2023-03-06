from pathlib import Path
from typing import Dict, List, Tuple
import logging
import sys
from copy import deepcopy

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