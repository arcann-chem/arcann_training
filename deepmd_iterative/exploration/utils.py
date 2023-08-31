"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/31

Functions
---------
generate_input_exploration_json(user_input_json: Dict, previous_json: Dict, default_input_json: Dict, merged_input_json: Dict, main_json: Dict) -> Dict
    Update and complete input JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

get_system_exploration(merged_input_json: Dict, system_auto_index: int) -> Tuple[Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],bool]
    Returns a tuple of system exploration parameters based on the input JSON and system number.

generate_starting_points(exploration_type: int, system_auto: int, training_path: str, padded_prev_iter: str, previous_json: Dict, input_present: bool, disturbed_start: bool) -> Tuple[List[str], List[str], bool]
    Generates a list of starting point file names.

create_models_list(main_json: Dict, previous_json: Dict, it_nnp: int, padded_prev_iter: str, training_path: Path, local_path: Path) -> Tuple[List[str], str]
    Generate a list of model file names and create symbolic links to the corresponding model files.

get_last_frame_number(model_deviation: np.ndarray, sigma_high_limit: float, disturbed_start: bool) -> int
    Returns the index of the last frame to be processed based on the given parameters.

update_system_nb_steps_factor(previous_json: Dict, system_auto_index: int) -> int
    Calculates a ratio based on information from a dictionary and returns a multiplying factor for system_nb_steps.

set_input_explordevi_json(user_input_json: Dict, previous_json: Dict, default_input_json: Dict, merged_input_json: Dict, main_json: Dict) -> Dict
    Updates the training JSON with input JSON, previous JSON and default JSON.
"""
# Standard library modules
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


@catch_errors_decorator
def generate_input_exploration_json(
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

    if main_json["exploration_type"] == "lammps":
        exploration_dep = 0
    elif main_json["exploration_type"] == "i-PI":
        exploration_dep = 1
    else:
        error_msg = f"'{main_json['exploration_type']}' is not recognized"
        raise ValueError(error_msg)

    system_count = len(main_json.get("systems_auto", []))

    for key in [
        "timestep_ps",
        "temperature_K",
        "exp_time_ps",
        "max_exp_time_ps",
        "job_walltime_h",
        "init_exp_time_ps",
        "init_job_walltime_h",
        "print_interval_mult",
        "disturbed_start",
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
            merged_input_json[key] = [value[0][exploration_dep]] * system_count
        else:
            # Check if previous or user provided a list
            if isinstance(value, List):
                if len(value) == system_count:
                    for it_value in value:
                        if (
                            isinstance(it_value, (int, float))
                            and key != "disturbed_start"
                        ) or (
                            isinstance(it_value, (bool)) and key == "disturbed_start"
                        ):
                            merged_input_json[key].append(it_value)
                        else:
                            error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}' or '{type(True)}' (for disturbed_start)"
                            raise TypeError(error_msg)
                else:
                    error_msg = f"Size mismatch: The length of the list should be '{system_count}' corresponding to the number of systems."
                    raise ValueError(error_msg)

            # If it is not a List
            elif (isinstance(value, (int, float)) and key != "disturbed_start") or (
                isinstance(value, (bool)) and key == "disturbed_start"
            ):
                merged_input_json[key] = [value] * system_count
            else:
                error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}' or '{type(True)}' (for disturbed_start)"
                raise TypeError(error_msg)

    return merged_input_json


@catch_errors_decorator
def get_system_exploration(
    merged_input_json: Dict, system_auto_index: int
) -> Tuple[
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    bool,
]:
    """
    Return a tuple of system exploration parameters based on the input JSON and system index.

    Parameters
    ----------
    merged_input_json : Dict[str, Any]
        A dictionary containing input parameters.
    system_auto_index : int
        An integer representing the system index.

    Returns
    -------
    Tuple[float, float, float, float, float, float, float, float, bool]
        A tuple containing system exploration parameters:
        - timestep_ps : float
            Simulation timestep in picoseconds.
        - temperature_K : float
            Simulation temperature in Kelvin.
        - exp_time_ps : float
            Total exploration time in picoseconds.
        - max_exp_time_ps : float
            Maximum allowed exploration time in picoseconds.
        - job_walltime_h : float
            Maximum job walltime in hours.
        - init_exp_time_ps : float
            Initial exploration time in picoseconds.
        - init_job_walltime_h : float
            Initial job walltime in hours.
        - print_interval_mult : float
            Print interval multiplier.
        - disturbed_start : bool
            Whether to start exploration from a disturbed minimum.
    """
    system_values = []
    for key in [
        "timestep_ps",
        "temperature_K",
        "exp_time_ps",
        "max_exp_time_ps",
        "job_walltime_h",
        "init_exp_time_ps",
        "init_job_walltime_h",
        "print_interval_mult",
        "disturbed_start",
    ]:
        system_values.append(merged_input_json[key][system_auto_index])
    return tuple(system_values)


@catch_errors_decorator
def generate_input_exploration_deviation_json(
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

    if main_json["exploration_type"] == "lammps":
        exploration_dep = 0
    elif main_json["exploration_type"] == "i-PI":
        exploration_dep = 1
    else:
        error_msg = f"'{main_json['exploration_type']}' is not recognized"
        raise ValueError(error_msg)

    system_count = len(main_json.get("systems_auto", []))

    for key in [
        "max_candidates",
        "sigma_low",
        "sigma_high",
        "sigma_high_limit",
        "ignore_first_x_ps",
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
            merged_input_json[key] = [value[exploration_dep]] * system_count
        else:
            # Check if previous or user provided a list
            if isinstance(value, List):
                if len(value) == system_count:
                    for it_value in value:
                        if isinstance(it_value, (int, float)):
                            merged_input_json[key].append(it_value)
                        else:
                            error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}'"
                            raise TypeError(error_msg)
                else:
                    error_msg = f"Size mismatch: The length of the list should be '{system_count}' corresponding to the number of systems."
                    raise ValueError(error_msg)

            # If it is not a List
            elif isinstance(value, (int, float)):
                merged_input_json[key] = [value] * system_count
            else:
                error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}'"
                raise TypeError(error_msg)
    return merged_input_json


@catch_errors_decorator
def get_system_deviation(
    merged_input_json: Dict, system_auto_index: int
) -> Tuple[
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
]:
    """
    Return a tuple of system exploration parameters based on the input JSON and system index.

    Parameters
    ----------
    merged_input_json : Dict[str, Any]
        A dictionary containing input parameters.
    system_auto_index : int
        An integer representing the system index.

    Returns
    -------
    Tuple[float, float, float, float, float, float, float, float, bool]
        A tuple containing system deviation parameters:
        - Max candidates : float
        - Sigma low : float
        - Sigma high : float
        - Sigma high limit : float
        - Ignore first x ps : float
    """
    system_values = []
    for key in [
        "max_candidates",
        "sigma_low",
        "sigma_high",
        "sigma_high_limit",
        "ignore_first_x_ps",
    ]:
        system_values.append(merged_input_json[key][system_auto_index])
    return tuple(system_values)


@catch_errors_decorator
def generate_input_exploration_disturbed_json(
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

    if main_json["exploration_type"] == "lammps":
        exploration_dep = 0
    elif main_json["exploration_type"] == "i-PI":
        exploration_dep = 1
    else:
        error_msg = f"'{main_json['exploration_type']}' is not recognized"
        raise ValueError(error_msg)

    system_count = len(main_json.get("systems_auto", []))

    for key in [
        "disturbed_start_value",
        "disturbed_candidate_value",
        "disturbed_candidate_indexes",
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
            merged_input_json[key] = [value[exploration_dep]] * system_count
        else:
            # Check if previous or user provided a list
            if isinstance(value, List):
                if len(value) == system_count:
                    for it_value in value:
                        if isinstance(it_value, (int, float)):
                            merged_input_json[key].append(it_value)
                        else:
                            error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}'"
                            raise TypeError(error_msg)
                else:
                    error_msg = f"Size mismatch: The length of the list should be '{system_count}' corresponding to the number of systems."
                    raise ValueError(error_msg)

            # If it is not a List
            elif isinstance(value, (int, float)):
                merged_input_json[key] = [value] * system_count
            else:
                error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}'"
                raise TypeError(error_msg)
    return merged_input_json


@catch_errors_decorator
def get_system_disturb(
    merged_input_json: Dict, system_auto_index: int
) -> Tuple[Union[float, int], Union[float, int], List[int],]:
    """
    Return a tuple of system exploration parameters based on the input JSON and system index.

    Parameters
    ----------
    merged_input_json : Dict[str, Any]
        A dictionary containing input parameters.
    system_auto_index : int
        An integer representing the system index.

    Returns
    -------
    Tuple[float, float, List]
        A tuple containing system deviation parameters:
        - disturbed_start_value : float
        - disturbed_candidate_value : float
        - disturbed_candidate_indexes : float
    """
    system_values = []
    for key in [
        "disturbed_start_value",
        "disturbed_candidate_value",
        "disturbed_candidate_indexes",
    ]:
        system_values.append(merged_input_json[key][system_auto_index])
    return tuple(system_values)


@catch_errors_decorator
def generate_starting_points(
    exploration_type: int,
    system_auto: str,
    training_path: str,
    padded_prev_iter: str,
    previous_json: Dict,
    input_present: bool,
    disturbed_start: bool,
) -> Tuple[List[str], List[str], bool]:
    """
    Generates a list of starting point file names.

    Parameters
    ----------
    exploration_type : int
        An integer representing the exploration type (1 for "lammps" or 2 for "i-PI").
    system_auto : str
        The name of the system.
    training_path : str
        The path to the training directory.
    padded_prev_iter : str
        A zero-padded string representing the previous iteration number.
    previous_json : Dict
        A dictionary containing information about the previous exploration.
    input_present : bool
        A boolean indicating whether an input file is present.
    disturbed_start : bool
        A boolean indicating whether to start from a disturbed minimum.

    Returns
    -------
    Tuple[List[str], List[str], bool]
        A tuple containing the starting point file names, the backup starting point file names,
        and a boolean indicating whether to start from a disturbed minimum.
    """

    # Determine file extension based on exploration type
    if exploration_type == 1:
        file_extension = "lmp"
    elif exploration_type == 2:
        file_extension = "xyz"

    # Get list of starting point file names for system and iteration
    starting_points_path = list(
        Path(training_path, "starting_structures").glob(
            f"{padded_prev_iter}_{system_auto}_*.{file_extension}"
        )
    )
    starting_points_all = [str(zzz).split("/")[-1] for zzz in starting_points_path]
    starting_points = [zzz for zzz in starting_points_all if "disturbed" not in zzz]
    starting_points_disturbed = [
        zzz for zzz in starting_points_all if zzz not in starting_points
    ]
    starting_points_bckp = starting_points.copy()
    starting_points_disturbed_bckp = starting_points_disturbed.copy()

    # Check if system should start from a disturbed minimum
    if input_present and disturbed_start:
        # If input file is present and disturbed start is requested, use disturbed starting points
        starting_points = starting_points_disturbed.copy()
        starting_points_bckp = starting_points_disturbed_bckp.copy()
        return starting_points, starting_points_bckp, True
    elif not input_present:
        # If input file is not present, check if system started from disturbed minimum in previous iteration
        if (
            previous_json["systems_auto"][system_auto]["disturbed_start"]
            and previous_json["systems_auto"][system_auto]["disturbed_start_value"]
        ):
            # If system started from disturbed minimum in previous iteration, use disturbed starting points
            starting_points = starting_points_disturbed.copy()
            starting_points_bckp = starting_points_disturbed_bckp.copy()
            return starting_points, starting_points_bckp, True
        else:
            # Otherwise, use regular starting points
            return starting_points, starting_points_bckp, False
    else:
        # If input file is present but disturbed start is not requested, use regular starting points
        return starting_points, starting_points_bckp, False


# Unittested
@catch_errors_decorator
def create_models_list(
    main_json: Dict,
    previous_json: Dict,
    it_nnp: int,
    padded_prev_iter: str,
    training_path: Path,
    local_path: Path,
) -> Tuple[List[str], str]:
    """
    Generate a list of model file names and create symbolic links to the corresponding model files.

    Parameters
    ----------
    main_json : Dict
        The main JSON.
    previous_json : Dict
        The JSON from the previous iteration.
    it_nnp : int
        An integer representing the index of the NNP model to start from.
    padded_prev_iter : str
        A string representing the zero-padded iteration number of the previous iteration.
    training_path : Path
        The path to the training directory.
    local_path : Path
        The path to the local directory.

    Returns
    -------
    Tuple[List[str], str]
        A tuple containing the list of model file names, and a string of space-separated model file names.
    """

    # Generate list of NNP model indices and reorder based on current model to propagate
    list_nnp = [zzz for zzz in range(1, main_json["nnp_count"] + 1)]
    reorder_nnp_list = (
        list_nnp[list_nnp.index(it_nnp) :] + list_nnp[: list_nnp.index(it_nnp)]
    )

    # Determine whether to use compressed models
    compress_str = "_compressed" if previous_json["is_compressed"] else ""

    # Generate list of model file names
    models_list = [
        "graph_" + str(f) + "_" + padded_prev_iter + compress_str + ".pb"
        for f in reorder_nnp_list
    ]

    # Create symbolic links to the model files in the local directory
    for it_sub_nnp in range(1, main_json["nnp_count"] + 1):
        nnp_apath = (
            training_path
            / "NNP"
            / (
                "graph_"
                + str(it_sub_nnp)
                + "_"
                + padded_prev_iter
                + compress_str
                + ".pb"
            )
        ).resolve()
        subprocess.call(["ln", "-nsf", str(nnp_apath), str(local_path)])

    # Join the model file names into a single string for ease of use
    models_string = " ".join(models_list)

    return models_list, models_string


# Unittested
@catch_errors_decorator
def get_last_frame_number(
    model_deviation: np.ndarray, sigma_high_limit: float, disturbed_start: bool
) -> int:
    """
    Returns the index of the last frame to be processed based on the given parameters.

    Parameters
    ----------
    model_deviation : np.ndarray
        The model deviation data, represented as a NumPy array.
    sigma_high_limit : float
        The threshold value for the deviation data. Frames with deviation values above this threshold will be ignored.
    disturbed_start : bool
        Indicates whether the first frame should be ignored because it is considered "disturbed".

    Returns
    -------
    int
        The index of the last frame to be processed, based on the input parameters.
    """
    # Ignore the first frame if it's considered "disturbed"
    if disturbed_start:
        start_frame = 1
    else:
        start_frame = 0

    # Check if any deviation values are over the sigma_high_limit threshold
    if np.any(model_deviation[start_frame:, 4] >= sigma_high_limit):
        last_frame = np.argmax(model_deviation[start_frame:, 4] >= sigma_high_limit)
    else:
        last_frame = -1

    return last_frame


# Unittested
@catch_errors_decorator
def update_system_nb_steps_factor(previous_json: Dict, system_auto_index: int) -> int:
    """
    Calculates a ratio based on information from a dictionary and returns a multiplying factor for system_nb_steps.

    Parameters
    ----------
    previous_json : Dict
        A dictionary containing information about the previous interation.
    system_auto_index : int
        An integer representing the system index.

    Returns
    -------
    int
        An integer representing the multiplying factor for system_nb_steps.
    """
    # Calculate the ratio of ill-described candidates to the total number of candidates
    ill_described_ratio = (
        previous_json["systems_auto"][system_auto_index]["nb_candidates"]
        + previous_json["systems_auto"][system_auto_index]["nb_rejected"]
    ) / previous_json["systems_auto"][system_auto_index]["nb_total"]

    # Return a multiplying factor for system_nb_steps based on the ratio of ill-described candidates
    if ill_described_ratio < 0.10:
        return 4
    elif ill_described_ratio < 0.20:
        return 2
    else:
        return 1
