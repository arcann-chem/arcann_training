from pathlib import Path
import logging
import sys
import json

from typing import Union


def json_read(file_path: Path, abort: bool = True, is_logged: bool = False) -> dict:
    """Read a JSON file to a JSON dict

    Args:
        file_path (Path): Path object to the file
        abort (bool, optional): True to abort, False create a new dict. Defaults to True.
        is_logged (bool, optional): Logging. Defaults to False.

    Returns:
        dict: JSON dictionary
    """
    if file_path.is_file():
        if is_logged:
            logging.info(f"Loading {file_path.name} from {file_path.parent}")
        return json.load(file_path.open())
    else:
        if abort:
            logging.error(f"File {file_path.name} not found in {file_path.parent}")
            logging.error(f"Aborting...")
            sys.exit(1)
        else:
            if is_logged:
                logging.info(f"Creating a {file_path.name} in {file_path.parent}")
            return {}


def read_default_input_json(default_json_file_apath: Path) -> dict:
    """_summary_

    Args:
        default_json_file_apath (Path): _description_

    Returns:
        dict: _description_
    """
    if default_json_file_apath.is_file():
        return json.load(default_json_file_apath.open())
    else:
        logging.warning(
            f"Default file {default_json_file_apath.name} not found in {default_json_file_apath.parent}"
        )
        logging.warning(f"Check your installation")
        return {}


def json_dump(json_dict: dict, file_path: Path, is_logged: bool = False):
    """Write a JSON dict to a JSON file

    Args:
        json_dict (dict): JSON dictionary
        file_path (Path): Path object to the file
        is_logged (bool, optional): Logging. Defaults to False.
    """
    with file_path.open("w", encoding="UTF-8") as f:
        json.dump(json_dict, f, indent=4)
        if is_logged:
            logging.info(f"Writing {file_path.name} in {file_path.parent}")


def json_dump_bak(json_dict: dict, file_path: Path, is_logged: bool = False):
    """_summary_

    Args:
        json_dict (dict): _description_
        file_path (Path): _description_
        is_logged (bool, optional): _description_. Defaults to False.
    """
    if file_path.is_file() and not file_path.is_symlink():
        file_path.rename(file_path.parent / (file_path.name + ".bak"))
        json_dump(json_dict, file_path, True)
    elif file_path.is_symlink():
        file_path.unlink()
        json_dump(json_dict, file_path, True)
    else:
        json_dump(json_dict, file_path, True)


def add_key_value_to_new_input(json_dict: dict, key: str, value):
    """_summary_

    Args:
        json_dict (dict): _description_
        key (str): _description_
        value (_type_): _description_
    """
    if json_dict == {}:
        json_dict[key] = {}
        json_dict[key]["value"] = value
    elif key not in json_dict:
        json_dict[key] = {}
        json_dict[key]["value"] = value
    else:
        json_dict[key]["value"] = value


def read_key_input_json(
    input_json: dict,
    new_input_json: dict,
    key: str,
    default_inputs_json: dict,
    step: str,
    default_present: bool = True,
    subsys_index: int = -1,
    subsys_number: int = 0,
)  -> Union[str,float,int,None]:
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

    Returns:
        _type_: _description_
    """
    ## Special keys first:
    if "system" in key and key in input_json and "value" in input_json[key]:
        if isinstance(input_json[key]["value"], str):
            add_key_value_to_new_input(new_input_json, key, input_json[key]["value"])
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
            add_key_value_to_new_input(new_input_json, key, input_json[key]["value"])
            return input_json[key]["value"]
        else:
            logging.error(f'Wrong type: "{key}" is {type(input_json[key]["value"])}')
            logging.error(f'Wrong type: "{key}" should be a {list} of {str}')
            logging.error(f"Aborting...")
            sys.exit(1)
    elif "user_spec" in key and key in input_json and "value" in input_json[key]:
        if isinstance(input_json[key]["value"], bool):
            add_key_value_to_new_input(new_input_json, key, input_json[key]["value"])
            return input_json[key]["value"]
        elif isinstance(input_json[key]["value"], str):
            add_key_value_to_new_input(new_input_json, key, input_json[key]["value"])
            return input_json[key]["value"]
        elif isinstance(input_json[key]["value"], list) and [
            isinstance(input_json[key]["value"][_], str)
            for _ in range(len(input_json[key]["value"]))
        ]:
            add_key_value_to_new_input(new_input_json, key, input_json[key]["value"])
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

    ## Check if the key is in the user input
    if key in input_json:
        ## Check if the key has a key called 'value'
        if "value" in input_json[key]:
            if input_json[key]["value"] is not None:
                ## Check if the key is subsys dependent (can be a list of values)
                ## If -1 it means no list
                if subsys_index == -1:
                    ## Check if the type correspond to the default
                    if isinstance(
                        input_json[key]["value"], type(default_inputs_json[step][key])
                    ):
                        add_key_value_to_new_input(
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
                ## If not then it is list
                else:
                    ## Check if it has the same type (meaning same value get propagated)
                    if isinstance(input_json[key]["value"], list) and isinstance(
                        input_json[key]["value"], type(default_inputs_json[step][key])
                    ):
                        add_key_value_to_new_input(
                            new_input_json, key, input_json[key]["value"]
                        )
                        return input_json[key]["value"]
                    ## If not check if is a list, and if the type inside the list matches, and return index
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
                        add_key_value_to_new_input(
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
                            f'Wrong type: "{key}" should be {type(default_inputs_json[step][key])} [Will be repeated on all subsys]'
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
                True
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
            add_key_value_to_new_input(
                new_input_json, key, default_inputs_json[step][key]
            )
            return default_inputs_json[step][key]


# def json_read_input(
#     file_path: Path, default_file: Path, is_logged: bool = False
# ) -> dict:
#     """Read a JSON file to a JSON dict. Special for inputs (read a default one if not present)

#     Args:
#         file_path (Path): Path object to the file
#         default_file (Path): Path object to the default one
#         is_logged (bool, optional): _description_. Defaults to False.

#     Returns:
#         dict: _description_
#     """
#     if file_path.is_file():
#         if is_logged:
#             logging.info(f"Loading {file_path.name} from {file_path.parent}")
#         user_input_json = json.load(file_path.open())
#         if default_file.is_file():
#             default_input_json = json.load(default_file.open())
#         else:
#             default_input_json = {}
#         return user_input_json, default_input_json
#     elif default_file.is_file():
#         if is_logged:
#             logging.info(
#                 f"Loading {default_file.name} from {default_file.parent}"
#             )
#         return json.load(default_file.open())
#     else:
#         logging.error(f"No input files found (user an")
#         logging.error(f"Aborting...")
#         sys.exit(1)


# def json_check_key_and_return_default(
#     json_dict: dict,
#     new_json_dict: dict,
#     key: str,
#     default_value=None,
#     list_index: int = -1,
# ):
#     """Check a key in dict and return values or default values

#     Args:
#         json_dict (dict): the dict
#         key (str): the key (must be in the first depth of the dict (something returned by dict.keys()))
#         default_value (_type_, optional): the default value trigger. Defaults to None.
#         list_index (int, optional): the index of the list (for subsys). Defaults to -1.

#     Returns:
#         Any: the value
#     """
#     default: bool
#     if type(default_value) is int or type(default_value) is float:
#         if json_dict[key]['value'] == default_value:
#             default = True
#         else:
#             default = False
#             add_key_value_to_new_input(new_json_dict, key, json_dict[key]['value'])
#             return json_dict[key]['value']
#     elif type(default_value) is list:
#         if len(json_dict[key]['value']) == 0:
#             default = True
#         else:
#             default = False
#     elif type(default_value) is str:
#         if json_dict[key]['value'] == default_value:
#             default = True
#         else:
#             default = False
#     elif json_dict[key]['value'] is None:
#         logging.error(f"There is no default value for this {key}")
#         logging.error("Aborting...")
#         sys.exit(1)
#     else:
#         default = False
#     if default:
#         if list_index == -1:
#             add_key_value_to_new_input(new_json_dict, key, json_dict[key]['_default'])
#             return json_dict[key]['_default']
#         elif type(json_dict[key]['_default']) is not list:
#             add_key_value_to_new_input(new_json_dict, key, json_dict[key]['_default'])
#             return json_dict[key]['_default']
#         else:
#             add_key_value_to_new_input(
#                 new_json_dict, key, json_dict[key]['_default'][list_index]
#             )
#             return json_dict[key]['_default'][list_index]
#     else:
#         if list_index == -1:
#             add_key_value_to_new_input(new_json_dict, key, json_dict[key]['value'])
#             return json_dict[key]['value']
#         else:
#             add_key_value_to_new_input(
#                 new_json_dict, key, json_dict[key]['value'][list_index]
#             )
#             return json_dict[key]['value'][list_index]
