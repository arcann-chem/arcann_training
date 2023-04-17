"""
Created: 2023/01/01
Last modified: 2023/04/17

The json module provides functions to manipulate JSON data (as dict).

Functions
---------
add_key_value_to_dict(dictionary: Dict, key: str, value: Any) -> None
    A function to add a new key-value pair to a dictionary.

backup_and_overwrite_json_file(json_dict: Dict, file_path: Path, enable_logging: bool = True) -> None
    A function to write a dictionary to a JSON file after creating a backup of the existing file.

load_default_json_file(file_path: Path) -> Dict
    A function to load a default JSON file from the given file path and return its contents as a dictionary.

load_json_file(file_path: Path, abort_on_error: bool = True, enable_logging: bool = True) -> Dict
    A function to load a JSON file from the given file path and return its contents as a dictionary.

write_json_file(json_dict: Dict, file_path: Path, enable_logging: bool = True, **kwargs) -> None
    A function to write a dictionary to a JSON file.
"""
# Standard library modules
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Union

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator

# Unittested
@catch_errors_decorator
def add_key_value_to_dict(dictionary: Dict, key: str, value: Any) -> None:
    """
    Adds a new key-value pair to a dictionary.

    If the dictionary is empty, a new sub-dictionary will be created with the specified key and value.
    If the key does not already exist in the dictionary, a new sub-dictionary will be created with the specified key and value.
    If the key already exists in the dictionary, the existing sub-dictionary's value will be updated.

    Parameters
    ----------
    dictionary : Dict
        The dictionary to which the key-value pair should be added.
    key : str
        The key to use for the new or updated sub-dictionary.
    value : Any
        The value to be associated with the new or updated sub-dictionary.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If dictionary is not a dictionary, or key is not a string, or value is None.
    ValueError
        If key is an empty string.
    """
    if not isinstance(dictionary, dict):
        error_msg = f"The dictionary argument must be a dictionary."
        raise TypeError(error_msg)
    if not isinstance(key, str):
        error_msg = f"The key argument must be a string."
        raise TypeError(error_msg)
    if key == "":
        error_msg = f"The key argument must not be an empty string."
        raise ValueError(error_msg)
    if value is None:
        error_msg = f"The value argument cannot be None."
        raise TypeError(error_msg)

    dictionary.setdefault(key, {})["value"] = value


# Unittested
@catch_errors_decorator
def backup_and_overwrite_json_file(
    json_dict: Dict, file_path: Path, enable_logging: bool = True
) -> None:
    """
    Write a dictionary to a JSON file after creating a backup of the existing file.

    If the file already exists, it will be renamed to have a ".json.bak" extension before the new data is written. If the
    file is a symbolic link, it will be removed before the new data is written.

    Parameters
    ----------
    json_dict : Dict
        A dictionary containing data to be written to the JSON file.
    file_path : Path
        A path object representing the file to write the JSON data to.
    enable_logging : bool, optional
        Whether to log information about the writing process. Defaults to False.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If file_path is not a Path object.
    """
    if not isinstance(file_path, Path):
        raise TypeError("file_path must be a Path object.")

    backup_path = file_path.with_suffix(".json.bak")
    # Create a backup of the original file, if it exists
    if file_path.is_file() and not file_path.is_symlink() and not backup_path.is_file():
        file_path.rename(backup_path)
    elif file_path.is_file() and not file_path.is_symlink() and backup_path.is_file():
        backup_path.unlink()
        file_path.rename(backup_path)
    # If the file is a symbolic link, remove it
    elif file_path.is_symlink():
        file_path.unlink()
    # Write the new data to the original file
    write_json_file(json_dict, file_path, enable_logging)


# Unittested
@catch_errors_decorator
def load_default_json_file(file_path: Path) -> Dict:
    """
    Load a default JSON file from the given file path and return its contents as a dictionary.

    Parameters
    ----------
    file_path : Path
        The path to the default JSON file to be loaded.

    Returns
    -------
    Dict
        A dictionary containing the contents of the default JSON file.

    Raises
    ------
    TypeError
        If file_path is not a Path object.
    """
    if not isinstance(file_path, Path):
        raise TypeError("'file_path' must be a Path object.")

    # Check if the file exists and is a file
    if file_path.is_file():
        # Open the file and load the contents as a dictionary
        with file_path.open(encoding="UTF-8") as json_file:
            # Check if the file is empty
            file_content = json_file.read().strip()
            if len(file_content) == 0:
                return {}
            return json.loads(file_content)
    else:
        # If the file cannot be found, return an empty dictionary and log a warning
        logging.warning(
            f"Default file {file_path.name} not found in {file_path.parent}"
        )
        logging.warning(f"Check your installation")
        return {}


# Unittested
@catch_errors_decorator
def load_json_file(
    file_path: Path, abort_on_error: bool = True, enable_logging: bool = True
) -> Dict:
    """
    Load a JSON file from the given file path and return its contents as a dictionary.

    Parameters
    ----------
    file_path: Path
        The path to the JSON file to be loaded.
    abort_on_error: bool
        Whether to abort the program if the file cannot be found. If True, an error message is logged and the program exits with an error code. If False, an empty dictionary is returned. Defaults is True.
    enable_logging: bool
        Whether to log information about the loading process. Defaults is True.

    Returns
    -------
    Dict
        A dictionary containing the contents of the JSON file.

    Raises
    ------
    TypeError
        If file_path is not a Path object.
    FileNotFoundError
        If the file cannot be found and abort_on_error is True.
    """
    if not isinstance(file_path, Path):
        error_msg = f"'file_path' must be a Path object."
        raise TypeError(error_msg)

    # Check if the file exists and is a file
    if file_path.is_file():
        # If logging is enabled, log information about the loading process
        if enable_logging:
            logging.info(f"Loading {file_path.name} from {file_path.parent}")
        # Open the file and load the contents as a dictionary
        with file_path.open(encoding="UTF-8") as json_file:
            # Check if the file is empty
            file_content = json_file.read().strip()
            if len(file_content) == 0:
                return {}
            return json.loads(file_content)
    else:
        # If the file cannot be found and abort_on_error is True, log an error message and exit with an error code
        if abort_on_error:
            error_msg = f"File {file_path.name} not found in {file_path.parent}."
            raise FileNotFoundError(error_msg)
        # If abort_on_error is False, return an empty dictionary
        else:
            # If logging is enabled, log information about the creation of the empty dictionary
            if enable_logging:
                logging.info(
                    f"Creating an empty dictionary: {file_path.name} in {file_path.parent}"
                )
            return {}


# Unittested
@catch_errors_decorator
def write_json_file(
    json_dict: Dict, file_path: Path, enable_logging: bool = True, **kwargs
) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters
    ----------
    json_dict : Dict
        A dictionary containing data to be written to the JSON file.
    file_path : Path
        A path object representing the file to write the JSON data to.
        Must be a Path object, otherwise a TypeError will be raised.
    enable_logging : bool, optional
        If True, log a message indicating the file and path that the JSON data is being written to. Defaults to True.
    **kwargs : optional
        Optional arguments to be passed to the json.dump() function.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If file_path is not a Path object.
    Exception
        If the file cannot be written.
    """
    if not isinstance(file_path, Path):
        raise TypeError("file_path must be a Path object.")

    try:
        # Open the file specified by the file_path argument in write mode
        with file_path.open("w", encoding="UTF-8") as json_file:
            # Use the json.dump() method to write the JSON data to the file
            json.dump(json_dict, json_file, indent=kwargs.get("indent", 4))
            # If log_write is True, log a message indicating the file and path that the JSON data is being written to
            if enable_logging:
                logging.info(f"JSON data written to {file_path.absolute()}")
    except (OSError, IOError) as e:
        # Raise an exception if the file path is not valid or the file cannot be written
        error_msg = f"Error writing JSON data to file {file_path}: {e}"
        raise Exception(error_msg)


# Need to phase out
############################################
def read_key_input_json(
    input_json: dict,
    new_input_json: dict,
    key: str,
    default_inputs_json: dict,
    step: str,
    default_present: bool = True,
    subsys_index: int = -1,
    subsys_number: int = 0,
    exploration_dep: int = -1,
) -> Union[str, float, int, None]:
    """_summary_

    Args:
        input_json (dict): _description_
        new_input_json (dict): _description_
        key (str): _description_
        default_inputs_json (dict): _description_
        step (str): _description_
        default_present (bool, optional): _description_. Defaults to True.
        subsys_index (int, optional): _description_. Defaults to -1.
        subsys_number (int, optional): _description_. Defaults to 0.
        exploration_dep (int, optional): _description. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # Special keys first:
    if "system" in key and key in input_json and "value" in input_json[key]:
        if isinstance(input_json[key]["value"], str):
            add_key_value_to_dict(new_input_json, key, input_json[key]["value"])
            return input_json[key]["value"]
        else:
            logging.error(f'Wrong type: "{key}" is {type(input_json[key]["value"])}')
            logging.error(f'Wrong type: "{key}" should be {str}')
            logging.error(f"Aborting...")
            sys.exit(1)
    elif "subsys_nr" in key and key in input_json and "value" in input_json[key]:
        if isinstance(input_json[key]["value"], list) and [
            isinstance(input_json[key]["value"][_], str)
            for _ in range(len(input_json[key]["value"]))
        ]:
            add_key_value_to_dict(new_input_json, key, input_json[key]["value"])
            return input_json[key]["value"]
        else:
            logging.error(f'Wrong type: "{key}" is {type(input_json[key]["value"])}')
            logging.error(f'Wrong type: "{key}" should be a {list} of {str}')
            logging.error(f"Aborting...")
            sys.exit(1)
    elif (
        "user_cluster_keyword" in key
        and key in input_json
        and "value" in input_json[key]
    ):
        if isinstance(input_json[key]["value"], bool):
            add_key_value_to_dict(new_input_json, key, input_json[key]["value"])
            return input_json[key]["value"]
        elif isinstance(input_json[key]["value"], str):
            add_key_value_to_dict(new_input_json, key, input_json[key]["value"])
            return input_json[key]["value"]
        elif isinstance(input_json[key]["value"], list) and [
            isinstance(input_json[key]["value"][_], str)
            for _ in range(len(input_json[key]["value"]))
        ]:
            add_key_value_to_dict(new_input_json, key, input_json[key]["value"])
            return input_json[key]["value"]
        else:
            logging.error(f'Wrong type: "{key}" is {type(input_json[key]["value"])}')
            logging.error(f'Wrong type: "{key}" should be a {bool} in the form: False')
            logging.error(
                f'Wrong type: "{key}" should be a {list} of {str} in the form: ["project","allocation","arch_name"]'
            )
            logging.error(
                f'Wrong type: "{key}" should be a {str} in the form: "shortcut"'
            )
            logging.error(f"Aborting...")
            sys.exit(1)
    # Check if the key is in the user input
    if key in input_json:
        # Check if the key has a key called 'value'
        if "value" in input_json[key]:
            if input_json[key]["value"] is not None:
                # Check if the key is subsys dependent (can be a list of values)
                # If -1 it means no list
                if subsys_index == -1:
                    if exploration_dep == -1:
                        # Check if the type correspond to the default
                        if isinstance(
                            input_json[key]["value"],
                            type(default_inputs_json[step][key]),
                        ):
                            add_key_value_to_dict(
                                new_input_json, key, input_json[key]["value"]
                            )
                            return input_json[key]["value"]
                        else:
                            logging.error(
                                f"Wrong type: \"{key}\" is {type(input_json[key]['value'])}"
                            )
                            logging.error(
                                f'Wrong type: "{key}" should be {type(default_inputs_json[step][key])}'
                            )
                            logging.error(f"Aborting...")
                            sys.exit(1)
                    else:
                        # Check if the type correspond to the default
                        if isinstance(
                            input_json[key]["value"],
                            type(default_inputs_json[step][key][exploration_dep]),
                        ):
                            add_key_value_to_dict(
                                new_input_json, key, input_json[key]["value"]
                            )
                            return input_json[key]["value"]
                        else:
                            logging.error(
                                f"Wrong type: \"{key}\" is {type(input_json[key]['value'])}"
                            )
                            logging.error(
                                f'Wrong type: "{key}" should be {type(default_inputs_json[step][key][exploration_dep])}'
                            )
                            logging.error(f"Aborting...")
                            sys.exit(1)
                # If not then it is list
                else:
                    if exploration_dep == -1:
                        # Check if it has the same type (meaning same value get propagated)
                        if not isinstance(
                            input_json[key]["value"], list
                        ) and isinstance(
                            input_json[key]["value"],
                            type(default_inputs_json[step][key]),
                        ):
                            add_key_value_to_dict(
                                new_input_json, key, input_json[key]["value"]
                            )
                            return input_json[key]["value"]
                        # If not check if it is a list, and if the type inside the list matches, and return index
                        elif (
                            isinstance(input_json[key]["value"], list)
                            and [
                                isinstance(
                                    input_json[key]["value"][_],
                                    type(default_inputs_json[step][key]),
                                )
                                for _ in range(len(input_json[key]["value"]))
                            ]
                            and subsys_number == len(input_json[key]["value"])
                        ):
                            add_key_value_to_dict(
                                new_input_json, key, input_json[key]["value"]
                            )
                            return input_json[key]["value"][subsys_index]
                        elif (
                            isinstance(input_json[key]["value"], list)
                            and [
                                isinstance(
                                    input_json[key]["value"][_],
                                    type(default_inputs_json[step][key]),
                                )
                                for _ in range(len(input_json[key]["value"]))
                            ]
                            and subsys_number != len(input_json[key]["value"])
                        ):
                            logging.error(
                                f"Wrong size: The length of the list is {len(input_json[key]['value'])}"
                            )
                            logging.error(
                                f"Wrong size: The length of the list should be {subsys_number} [Subsys number]"
                            )
                            logging.error(f"Aborting...")
                            sys.exit(1)
                        elif not isinstance(input_json[key]["value"], list):
                            logging.error(
                                f'Wrong type: "{key}" is {type(input_json[key]["value"])}'
                            )
                            logging.error(
                                f'Wrong type: "{key}" should be {type(default_inputs_json[step][key])} '
                                f"[Will be repeated on all subsys] "
                            )
                            logging.error(f"Aborting...")
                            sys.exit(1)
                        elif isinstance(input_json[key]["value"], list):
                            logging.error(
                                f'Wrong type: "{key}" is a {list} of {type(input_json[key]["value"][subsys_index])}'
                            )
                            logging.error(
                                f'Wrong type: "{key}" should a {list} of {type(default_inputs_json[step][key])}'
                            )
                            logging.error(f"Aborting...")
                            sys.exit(1)

                    else:
                        # Check if it has the same type (meaning same value get propagated)
                        if not isinstance(
                            input_json[key]["value"], list
                        ) and isinstance(
                            input_json[key]["value"],
                            type(default_inputs_json[step][key][exploration_dep]),
                        ):
                            add_key_value_to_dict(
                                new_input_json, key, input_json[key]["value"]
                            )
                            return input_json[key]["value"]
                        # If not check if it is a list, and if the type inside the list matches, and return index
                        elif (
                            isinstance(input_json[key]["value"], list)
                            and [
                                isinstance(
                                    input_json[key]["value"][_],
                                    type(
                                        default_inputs_json[step][key][exploration_dep]
                                    ),
                                )
                                for _ in range(len(input_json[key]["value"]))
                            ]
                            and subsys_number == len(input_json[key]["value"])
                        ):
                            add_key_value_to_dict(
                                new_input_json, key, input_json[key]["value"]
                            )
                            return input_json[key]["value"][subsys_index]
                        elif (
                            isinstance(input_json[key]["value"], list)
                            and [
                                isinstance(
                                    input_json[key]["value"][_],
                                    type(
                                        default_inputs_json[step][key][exploration_dep]
                                    ),
                                )
                                for _ in range(len(input_json[key]["value"]))
                            ]
                            and subsys_number != len(input_json[key]["value"])
                        ):
                            logging.error(
                                f"Wrong size: The length of the list is {len(input_json[key]['value'])}"
                            )
                            logging.error(
                                f"Wrong size: The length of the list should be {subsys_number} [Subsys number]"
                            )
                            logging.error(f"Aborting...")
                            sys.exit(1)
                        elif not isinstance(input_json[key]["value"], list):
                            logging.error(
                                f'Wrong type: "{key}" is {type(input_json[key]["value"])}'
                            )
                            logging.error(
                                f'Wrong type: "{key}" should be {type(default_inputs_json[step][key][exploration_dep])} '
                                f"[Will be repeated on all subsys] "
                            )
                            logging.error(f"Aborting...")
                            sys.exit(1)
                        elif isinstance(input_json[key]["value"], list):
                            logging.error(
                                f'Wrong type: "{key}" is a {list} of {type(input_json[key]["value"][subsys_index])}'
                            )
                            logging.error(
                                f'Wrong type: "{key}" should a {list} of {type(default_inputs_json[step][key][exploration_dep])}'
                            )
                            logging.error(f"Aborting...")
                            sys.exit(1)
            else:
                pass
        else:
            logging.error(f'The key "{key}" does not have a value subkey')
            logging.error(f"Check your structure")
            logging.error("Aborting...")
            sys.exit(1)
    if not default_present:
        logging.error(f"The defaut_input.json is not present")
        logging.error(f"Check your installation or provide all correct values")
        logging.error(f"Aborting...")
        sys.exit(1)
    if key in default_inputs_json[step]:
        if default_inputs_json[step][key] is None:
            logging.error(f'There is no default value for this "{key}"')
            logging.error(f"Aborting...")
            sys.exit(1)
        else:
            if exploration_dep == -1:
                add_key_value_to_dict(
                    new_input_json, key, default_inputs_json[step][key]
                )
                return default_inputs_json[step][key]
            elif exploration_dep == 0:
                add_key_value_to_dict(
                    new_input_json, key, default_inputs_json[step][key][0]
                )
                return default_inputs_json[step][key][0]
            else:
                add_key_value_to_dict(
                    new_input_json, key, default_inputs_json[step][key][1]
                )
                return default_inputs_json[step][key][1]
