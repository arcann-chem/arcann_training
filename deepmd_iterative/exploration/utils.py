"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/16

Functions
---------
set_input_explor_json(input_json: Dict, previous_json: Dict, default_json: Dict, new_input_json: Dict, config_json: Dict) -> Dict
    Update and complete input JSON with user-defined parameters, previously defined parameters, and default parameters for training.

get_subsys_exploration(new_input_json: Dict, it0_subsys_nr: int) -> Tuple[Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],bool]
    Returns a tuple of subsystem exploration parameters based on the input JSON and subsystem number.

generate_starting_points(exploration_type: int, it_subsys_nr: int, training_path: str, previous_iteration_zfill: str, prevexploration_json: Dict, input_present: bool, subsys_disturbed_start: bool) -> Tuple[List[str], List[str], bool]
    Generates a list of starting point file names.

create_models_list(config_json: Dict, prevtraining_json: Dict, it_nnp: int, previous_iteration_zfill: str, training_path: Path, local_path: Path) -> Tuple[List[str], str]
    Generate a list of model file names and create symbolic links to the corresponding model files.

get_last_frame_number(model_deviation: np.ndarray, sigma_high_limit: float, is_start_disturbed: bool) -> int
    Returns the index of the last frame to be processed based on the given parameters.

update_subsys_nb_steps_factor(previous_exploration_config: Dict, it0_subsys_nr: int) -> int
    Calculates a ratio based on information from a dictionary and returns a multiplying factor for subsys_nb_steps.

set_input_explordevi_json(input_json: Dict, previous_json: Dict, default_json: Dict, new_input_json: Dict, config_json: Dict) -> Dict
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
def set_input_explor_json(
    input_json: Dict,
    previous_json: Dict,
    default_json: Dict,
    new_input_json: Dict,
    config_json: Dict,
) -> Dict:
    """
    Update and complete input JSON with user-defined parameters, previously defined
    parameters, and default parameters for training.

    Parameters
    ----------
    input_json : dict
        Input JSON containing user-defined parameters.
    previous_json : dict
        Previous JSON containing previously defined parameters.
    default_json : dict
        Default JSON containing default parameters.
    new_input_json : dict
        Input JSON updated/completed with previous/defaults.
    config_json : dict
        Configuration JSON containing exploration type and subsystem count parameters.

    Returns
    -------
    dict
        Input JSON updated/completed with previous/defaults.

    Raises
    ------
    ValueError
        If exploration type is not "lammps" or "i-PI".
    KeyError
        If the key is not found in any JSON.
    TypeError
        If the value type is not int/float or bool (for disturbed_start).
    ValueError
        If the length of the value list is not equal to subsystem count.
    """

    if config_json["exploration_type"] == "lammps":
        exploration_dep = 0
    elif config_json["exploration_type"] == "i-PI":
        exploration_dep = 1
    else:
        error_msg = f"{config_json['exploration_type']} is not known."
        raise ValueError(error_msg)

    subsys_count = len(config_json.get("subsys_nr", []))

    for key in [
        "timestep_ps",
        "temperature_K",
        "exp_time_ps",
        "max_exp_time_ps",
        "job_walltime_h",
        "init_exp_time_ps",
        "init_job_walltime_h",
        "disturbed_start",
        "print_interval_mult",
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
            error_msg = f"'{key}' not found in any JSON."
            raise KeyError(error_msg)

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
                            error_msg = f"Wrong type: the type is {type(it_value)} it should be int/float or bool (for disturbed_start)."
                            raise TypeError(error_msg)
                else:
                    error_msg = f"Wrong size: The length of the list should be {subsys_count} [Subsys]."
                    raise ValueError(error_msg)

            # If it is not a List
            elif (isinstance(value, (int, float)) and key != "disturbed_start") or (
                isinstance(value, (bool)) and key == "disturbed_start"
            ):
                new_input_json[key] = [value] * subsys_count
            else:
                error_msg = f"Wrong type: the type is {type(it_value)} it should be int/float or bool (for disturbed_start)."
                raise TypeError(error_msg)

    return new_input_json


@catch_errors_decorator
def get_subsys_exploration(
    new_input_json: Dict, it0_subsys_nr: int
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
    Returns a tuple of subsystem exploration parameters based on the input JSON and subsystem number.

    Parameters
    ----------
    new_input_json : Dict[str, Any]
        A dictionary object containing input parameters.
    it0_subsys_nr : int
        An integer representing the subsystem index.

    Returns
    -------
    Tuple[float, float, float, float, float, float, float, float, bool]
        A tuple containing the subsystem exploration parameters:
        - timestep_ps : float
            The simulation timestep in picoseconds.
        - temperature_K : float
            The simulation temperature in Kelvin.
        - exp_time_ps : float
            The total exploration time in picoseconds.
        - max_exp_time_ps : float
            The maximum allowed exploration time in picoseconds.
        - job_walltime_h : float
            The maximum job walltime in hours.
        - init_exp_time_ps : float
            The initial exploration time in picoseconds.
        - init_job_walltime_h : float
            The initial job walltime in hours.
        - print_interval_mult : float
            The print interval multiplier.
        - disturbed_start : bool
            Whether to start the exploration from a disturbed minimum.
    """
    subsys_values = []
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
        subsys_values.append(new_input_json[key][it0_subsys_nr])
    return tuple(subsys_values)


@catch_errors_decorator
def get_subsys_deviation(
    new_input_json: Dict, it0_subsys_nr: int
) -> Tuple[
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
]:
    subsys_values = []
    for key in [
        "max_candidates",
        "sigma_low",
        "sigma_high",
        "sigma_high_limit",
        "ignore_first_x_ps",
    ]:
        subsys_values.append(new_input_json[key][it0_subsys_nr])
    return tuple(subsys_values)


@catch_errors_decorator
def get_subsys_disturb(
    new_input_json: Dict, it0_subsys_nr: int
) -> Tuple[Union[float, int], Union[float, int],]:
    subsys_values = []
    for key in [
        "disturbed_min_value",
        "distrubed_candidate_value",
    ]:
        subsys_values.append(new_input_json[key][it0_subsys_nr])
    return tuple(subsys_values)


@catch_errors_decorator
def generate_starting_points(
    exploration_type: int,
    it_subsys_nr: int,
    training_path: str,
    previous_iteration_zfill: str,
    prevexploration_json: Dict,
    input_present: bool,
    subsys_disturbed_start: bool,
) -> Tuple[List[str], List[str], bool]:
    """
    Generates a list of starting point file names.

    Parameters
    ----------
    exploration_type : int
        An integer representing the exploration type (1 for "lammps" or 2 for "i-PI").
    it_subsys_nr : int
        An integer representing the subsystem index.
    training_path : str
        The path to the training directory.
    previous_iteration_zfill : str
        A zero-padded string representing the previous iteration number.
    prevexploration_json : Dict
        A dictionary containing information about the previous exploration.
    input_present : bool
        A boolean indicating whether an input file is present.
    subsys_disturbed_start : bool
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

    # Get list of starting point file names for subsystem and iteration
    starting_points_path = list(
        Path(training_path, "starting_structures").glob(
            f"{previous_iteration_zfill}_{it_subsys_nr}_*.{file_extension}"
        )
    )
    starting_points_all = [str(zzz).split("/")[-1] for zzz in starting_points_path]
    starting_points = [zzz for zzz in starting_points_all if "disturbed" not in zzz]
    starting_points_disturbed = [
        zzz for zzz in starting_points_all if zzz not in starting_points
    ]
    starting_points_bckp = starting_points.copy()
    starting_points_disturbed_bckp = starting_points_disturbed.copy()

    # Check if subsystem should start from a disturbed minimum
    if input_present and subsys_disturbed_start:
        # If input file is present and disturbed start is requested, use disturbed starting points
        starting_points = starting_points_disturbed.copy()
        starting_points_bckp = starting_points_disturbed_bckp.copy()
        return starting_points, starting_points_bckp, True
    elif not input_present:
        # If input file is not present, check if subsystem started from disturbed minimum in previous iteration
        if (
            prevexploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"]
            and prevexploration_json["subsys_nr"][it_subsys_nr]["disturbed_min"]
        ):
            # If subsystem started from disturbed minimum in previous iteration, use disturbed starting points
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
    config_json: Dict,
    prevtraining_json: Dict,
    it_nnp: int,
    previous_iteration_zfill: str,
    training_path: Path,
    local_path: Path,
) -> Tuple[List[str], str]:
    """
    Generate a list of model file names and create symbolic links to the corresponding model files.

    Parameters
    ----------
    config_json : Dict
        A dictionary object containing configuration parameters.
    prevtraining_json : Dict
        A dictionary object containing training data from previous iterations.
    it_nnp : int
        An integer representing the index of the NNP model to start from.
    previous_iteration_zfill : str
        A string representing the zero-padded iteration number of the previous training iteration.
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
    list_nnp = [zzz for zzz in range(1, config_json["nnp_count"] + 1)]
    reorder_nnp_list = (
        list_nnp[list_nnp.index(it_nnp) :] + list_nnp[: list_nnp.index(it_nnp)]
    )

    # Determine whether to use compressed models
    compress_str = "_compressed" if prevtraining_json["is_compressed"] else ""

    # Generate list of model file names
    models_list = [
        "graph_" + str(f) + "_" + previous_iteration_zfill + compress_str + ".pb"
        for f in reorder_nnp_list
    ]

    # Create symbolic links to the model files in the local directory
    for it_sub_nnp in range(1, config_json["nnp_count"] + 1):
        nnp_apath = (
            training_path
            / "NNP"
            / (
                "graph_"
                + str(it_sub_nnp)
                + "_"
                + previous_iteration_zfill
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
    model_deviation: np.ndarray, sigma_high_limit: float, is_start_disturbed: bool
) -> int:
    """
    Returns the index of the last frame to be processed based on the given parameters.

    Parameters
    ----------
    model_deviation : np.ndarray
        The model deviation data, represented as a NumPy array.
    sigma_high_limit : float
        The threshold value for the deviation data. Frames with deviation values above this threshold will be ignored.
    is_start_disturbed : bool
        Indicates whether the first frame should be ignored because it is considered "disturbed".

    Returns
    -------
    int
        The index of the last frame to be processed, based on the input parameters.
    """
    # Ignore the first frame if it's considered "disturbed"
    if is_start_disturbed:
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
def update_subsys_nb_steps_factor(
    previous_exploration_config: Dict, it0_subsys_nr: int
) -> int:
    """
    Calculates a ratio based on information from a dictionary and returns a multiplying factor for subsys_nb_steps.

    Parameters
    ----------
    previous_exploration_config : Dict
        A dictionary containing information about a previous exploration.
    it0_subsys_nr : int
        An integer representing the subsystem index.

    Returns
    -------
    int
        An integer representing the multiplying factor for subsys_nb_steps.
    """
    # Calculate the ratio of ill-described candidates to the total number of candidates
    ill_described_ratio = (
        previous_exploration_config["subsys_nr"][it0_subsys_nr]["nb_candidates"]
        + previous_exploration_config["subsys_nr"][it0_subsys_nr]["nb_rejected"]
    ) / previous_exploration_config["subsys_nr"][it0_subsys_nr]["nb_total"]

    # Return a multiplying factor for subsys_nb_steps based on the ratio of ill-described candidates
    if ill_described_ratio < 0.10:
        return 4
    elif ill_described_ratio < 0.20:
        return 2
    else:
        return 1


@catch_errors_decorator
def set_input_explordevi_json(
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
        error_msg = f"{config_json['exploration_type']} is not known."
        raise ValueError(error_msg)

    subsys_count = len(config_json.get("subsys_nr", []))

    for key in [
        "max_candidates",
        "sigma_low",
        "sigma_high",
        "sigma_high_limit",
        "ignore_first_x_ps",
        "",
        "",
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
            error_msg = f"'{key}' not found in any JSON."
            raise KeyError(error_msg)

        # Everything is subsys dependent so a list
        new_input_json[key] = []

        # Default is used for the key
        if default_used:
            new_input_json[key] = [value[exploration_dep]] * subsys_count
        else:
            # Check if previous or user provided a list
            if isinstance(value, List):
                if len(value) == subsys_count:
                    for it_value in value:
                        if isinstance(it_value, (int, float)):
                            new_input_json[key].append(it_value)
                        else:
                            error_msg = f"Wrong type: the type is {type(it_value)} it should be int/float."
                            raise TypeError(error_msg)
                else:
                    error_msg = f"Wrong size: The length of the list should be {subsys_count} [Subsys]."
                    raise ValueError(error_msg)

            # If it is not a List
            elif isinstance(value, (int, float)):
                new_input_json[key] = [value] * subsys_count
            else:
                error_msg = (
                    f"Wrong type: the type is {type(it_value)} it should be int/float."
                )
                raise TypeError(error_msg)
    return new_input_json
