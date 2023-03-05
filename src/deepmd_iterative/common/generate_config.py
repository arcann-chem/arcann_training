from pathlib import Path
from typing import Dict, Tuple

from deepmd_iterative.common.json import read_key_input_json, load_json_file

# Used in initialization - init
def set_config_json(
    input_json: Dict,
    new_input_json: Dict,
    default_input_json: Dict,
    step_name: str,
    default_present: bool,
) -> Tuple[Dict, str]:
    """
    Generate a config JSON file (Dict).

    Args:
        input_json (Dict): The user input JSON file.
        new_input_json (Dict): A deepcopy of the user input JSON file.
        default_input_json (Dict): The default input JSON file.
        step_name (str): The name of the current step.
        default_present (bool): A flag indicating whether the default input JSON file is present.

    Returns:
        A tuple containing the config JSON file (Dict) and the current iteration, zero-padded to three digits (str).
    """
    # Create the config JSON file
    config_json = {}
    for key in ["system", "nb_nnp", "exploration_type"]:
        config_json[key] = read_key_input_json(
            input_json,
            new_input_json,
            key,
            default_input_json,
            step_name,
            default_present,
        )

    config_json["current_iteration"] = 0
    current_iteration_zfill = str(config_json["current_iteration"]).zfill(3)

    config_json["subsys_nr"] = {}
    for subsys_nr in (
        read_key_input_json(
            input_json,
            new_input_json,
            "subsys_nr",
            default_input_json,
            step_name,
            default_present,
        )
    ):
        config_json["subsys_nr"][subsys_nr] = {}

    return config_json, current_iteration_zfill

# Used in training - prepartion
def set_training_json(
    control_path: Path,
    current_iteration_zfill: str,
    input_json: Dict,
    new_input_json: Dict,
    default_input_json: Dict,
    step_name: str,
    default_present: bool,
) -> Dict:
    """
    Load or generate a training JSON file, updating various fields based on input JSON configuration files.

    Args:
        control_path (Path): The path to the control directory.
        current_iteration_zfill (str): The current iteration, zero-padded to three digits.
        input_json (Dict): The user input JSON file.
        new_input_json (Dict): The deepcopy of the user input JSON file.
        default_input_json (Dict): The default input JSON file.
        step_name (str): The name of the current step.
        default_present (bool): A flag indicating whether the default input JSON file is present.

    Returns:
        The updated training JSON file (dict).
    """

    # Load or create the training JSON file
    training_json = load_json_file(
        (control_path / f"training_{current_iteration_zfill}.json"),
        abort_on_error=False,
    )
    if not training_json:
        training_json = {}

    # Update the training JSON configuration with values from the input JSON files
    for key in ["use_initial_datasets", "use_extra_datasets", "deepmd_model_version",
                "deepmd_model_type_descriptor", "start_lr", "stop_lr", "decay_rate",
                "decay_steps_fixed", "numb_steps", "numb_test"]:
        training_json[key] = read_key_input_json(
            input_json,
            new_input_json,
            key,
            default_input_json,
            step_name,
            default_present,
        )

    return training_json


def set_subsys_params_exploration(
    input_json: Dict,
    new_input_json: Dict,
    default_input_json: Dict,
    config_json: Dict,
    step_name: str,
    default_present: bool,
    it0_subsys_nr: int,
    exploration_type: int
) -> Tuple[float, float, float, float, float, float, bool]:
    """
    Sets exploration parameters for a specific subsystem.

    Args:
        input_json (Dict): The user input JSON file.
        new_input_json (Dict): The deepcopy of the user input JSON file.
        default_input_json (Dict): The default input JSON file.
        config_json (Dict): The config JSON file.
        step_name (str): The name of the current step.
        default_present (bool): A flag indicating whether the default input JSON file is present.
        it0_subsys_nr (int): An integer representing the subsystem index.
        exploration_type (int): An integer representing the exploration type.

    Returns:
        A tuple containing the following simulation parameters for the specified subsystem:
        - timestep (ps) (float)
        - temperature (K) (float)
        - exploration time (ps) (float)
        - max exploration time (ps) (float)
        - job wall time (h) (float)
        - print multiplier (float)
        - disturbed start (bool)
    """

    ## Order is important here
    subsys_values = []
    for key in ["timestep_ps", "temperature_K", "exp_time_ps",
                "max_exp_time_ps", "job_walltime_h", "disturbed_start"]:
            subsys_values.append(
                read_key_input_json(
                    input_json,
                    new_input_json,
                    key,
                    default_input_json,
                    step_name,
                    default_present,
                    subsys_index=it0_subsys_nr,
                    subsys_number=len(config_json["subsys_nr"]),
                    exploration_dep=exploration_type,
                )
            )

    subsys_values.append(
        read_key_input_json(
            input_json,
            new_input_json,
            "print_mult",
            default_input_json,
            step_name,
            default_present,
            subsys_index=it0_subsys_nr,
            subsys_number=len(config_json["subsys_nr"]),
        )
    )

    return tuple(subsys_values)



def set_subsys_params_deviation(input_json: dict, new_input_json: dict, default_input_json: dict, config_json: dict, step_name: str, default_present: bool, it0_subsys_nr: int) -> Tuple[int, float, float, float, float]:
    """
    Sets candidate selection parameters for a specific subsystem.

    Args:
        input_json (dict): A dictionary containing the input JSON data.
        new_input_json (dict): A dictionary containing the new input JSON data.
        default_input_json (dict): A dictionary containing the default input JSON data.
        config_json (dict): A dictionary containing the config JSON data.
        step_name (str): A string representing the name of the step.
        default_present (bool): A boolean indicating whether the default input JSON data is present.
        it0_subsys_nr (int): An integer representing the subsystem index.

    Returns:
        A tuple containing the following candidate selection parameters for the specified subsystem:
        - maximum number of candidates (int)
        - lower limit for sigma (float)
        - upper limit for sigma (float)
        - maximum upper limit of sigma (float)
        - amount of time to ignore at start of simulation (float)
    """

    max_candidates = read_key_input_json(
        input_json,
        new_input_json,
        "max_candidates",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
    )

    sigma_low = read_key_input_json(
        input_json,
        new_input_json,
        "sigma_low",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
    )

    sigma_high = read_key_input_json(
        input_json,
        new_input_json,
        "sigma_high",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
    )

    sigma_high_limit = read_key_input_json(
        input_json,
        new_input_json,
        "sigma_high_limit",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
    )

    ignore_first_x_ps = read_key_input_json(
        input_json,
        new_input_json,
        "ignore_first_x_ps",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
    )

    return max_candidates, sigma_low, sigma_high, sigma_high_limit, ignore_first_x_ps
