from typing import Dict, Tuple

from deepmd_iterative.common.json import read_key_input_json, load_json_file


def generate_config_json(
    input_json: Dict,
    new_input_json: Dict,
    default_input_json: Dict,
    step_name: str,
    default_present: bool,
) -> Tuple[Dict, str]:
    """
    Generate a config JSON configuration dictionary.

    This function takes in four input JSON configuration files and a step name, and returns a dictionary
    that can be used to configure. The dictionary contains various configuration options,
    including the system, the number of neural network potentials to use, the exploration type, and the
    current iteration.

    Parameters:
    input_json: The input JSON configuration dictionary.
    new_input_json: A new input JSON configuration dictionary.
    default_input_json: The default input JSON configuration dictionary.
    step_name: The name of the current step.
    default_present: A flag indicating whether the default input JSON configuration is present.

    Returns:
    A tuple containing the training JSON configuration dictionary and the current iteration, zero-padded
    to three digits.
    """
    config_json: Dict = {
        "system": read_key_input_json(
            input_json,
            new_input_json,
            "system",
            default_input_json,
            step_name,
            default_present,
        ),
        "nb_nnp": read_key_input_json(
            input_json,
            new_input_json,
            "nb_nnp",
            default_input_json,
            step_name,
            default_present,
        ),
        "exploration_type": read_key_input_json(
            input_json,
            new_input_json,
            "exploration_type",
            default_input_json,
            step_name,
            default_present,
        ),
        "current_iteration": 0,
    }

    config_json["subsys_nr"] = {}
    for it_subsys_nr, subsys_nr in enumerate(
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

    current_iteration_zfill: str = str(config_json["current_iteration"]).zfill(3)

    return config_json, current_iteration_zfill


def get_or_generate_training_json(
    control_path: str,
    current_iteration_zfill: str,
    input_json: Dict,
    new_input_json: Dict,
    default_input_json: Dict,
    step_name: str,
    default_present: bool,
    machine: str,
    machine_spec: Dict,
    machine_launch_command: str,
) -> Dict:
    """
    Load or create a training JSON configuration file, updating various fields based on input JSON configuration files.

    This function takes in several input files, including a control path, the current iteration zero-padded to three digits,
    and various input JSON configuration dictionaries. It then loads a training JSON configuration file, if it exists,
    and updates various fields based on values from the input JSON dictionaries. If the training JSON configuration file
    does not exist, it creates a new one with default values.

    Parameters:
    control_path: The path to the control directory.
    current_iteration_zfill: The current iteration, zero-padded to three digits.
    input_json: The input JSON configuration dictionary.
    new_input_json: A new input JSON configuration dictionary.
    default_input_json: The default input JSON configuration dictionary.
    step_name: The name of the current step.
    default_present: A flag indicating whether the default input JSON configuration is present.
    machine: The name of the machine to use for training.
    machine_spec: A dictionary containing machine-specific configuration options.
    machine_launch_command: The launch command to use on the machine.

    Returns:
    A dictionary containing the updated training JSON configuration.
    """

    # Load or create the training JSON configuration file
    training_json = load_json_file(
        (control_path / f"training_{current_iteration_zfill}.json"),
        abort_on_error=False,
    )
    if not training_json:
        training_json = {}

    # Update the training JSON configuration with values from the input JSON files
    training_json["use_initial_datasets"] = read_key_input_json(
        input_json,
        new_input_json,
        "use_initial_datasets",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["use_extra_datasets"] = read_key_input_json(
        input_json,
        new_input_json,
        "use_extra_datasets",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["deepmd_model_version"] = read_key_input_json(
        input_json,
        new_input_json,
        "deepmd_model_version",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["deepmd_model_type_descriptor"] = read_key_input_json(
        input_json,
        new_input_json,
        "deepmd_model_type_descriptor",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["start_lr"] = read_key_input_json(
        input_json,
        new_input_json,
        "start_lr",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["stop_lr"] = read_key_input_json(
        input_json,
        new_input_json,
        "stop_lr",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["decay_rate"] = read_key_input_json(
        input_json,
        new_input_json,
        "decay_rate",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["decay_steps"] = read_key_input_json(
        input_json,
        new_input_json,
        "decay_steps",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["decay_steps_fixed"] = read_key_input_json(
        input_json,
        new_input_json,
        "decay_steps_fixed",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["numb_steps"] = read_key_input_json(
        input_json,
        new_input_json,
        "numb_steps",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["numb_test"] = read_key_input_json(
        input_json,
        new_input_json,
        "numb_test",
        default_input_json,
        step_name,
        default_present,
    )
    training_json["machine"] = machine
    training_json["project_name"] = machine_spec["project_name"]
    training_json["allocation_name"] = machine_spec["allocation_name"]
    training_json["arch_name"] = machine_spec["arch_name"]
    training_json["arch_type"] = machine_spec["arch_type"]
    training_json["launch_command"] = machine_launch_command

    return training_json



def set_subsys_params_exploration(input_json: dict, new_input_json: dict, default_input_json: dict, config_json: dict, step_name: str, default_present: bool, it0_subsys_nr: int, exploration_type: int) -> Tuple[float, float, float, float, float, float, bool]:
    """
    Sets exploration parameters for a specific subsystem.

    Args:
        input_json (dict): A dictionary containing the input JSON data.
        new_input_json (dict): A dictionary containing the new input JSON data.
        default_input_json (dict): A dictionary containing the default input JSON data.
        config_json (dict): A dictionary containing the config JSON data.
        step_name (str): A string representing the name of the step.
        default_present (bool): A boolean indicating whether the default JSON data is present.
        it0_subsys_nr (int): An integer representing the subsystem index.
        exploration_type (int): An integer representing the exploration type.

    Returns:
        A tuple containing the following simulation parameters for the specified subsystem:
        - timestep (float)
        - temperature (float)
        - exploration time (float)
        - max exploration time (float)
        - job wall time (float)
        - print multiplier (float)
        - disturbed start (bool)
    """
    subsys_timestep = read_key_input_json(
        input_json,
        new_input_json,
        "timestep_ps",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
        exploration_dep=exploration_type,
    )

    subsys_temp = read_key_input_json(
        input_json,
        new_input_json,
        "temperature_K",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
        exploration_dep=exploration_type,
    )

    subsys_exp_time_ps = read_key_input_json(
        input_json,
        new_input_json,
        "exp_time_ps",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
        exploration_dep=exploration_type,
    )

    subsys_max_exp_time_ps = read_key_input_json(
        input_json,
        new_input_json,
        "max_exp_time_ps",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
        exploration_dep=exploration_type,
    )

    subsys_job_walltime_h = read_key_input_json(
        input_json,
        new_input_json,
        "job_walltime_h",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
        exploration_dep=exploration_type,
    )

    subsys_print_mult = read_key_input_json(
        input_json,
        new_input_json,
        "print_mult",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
    )

    subsys_disturbed_start = read_key_input_json(
        input_json,
        new_input_json,
        "disturbed_start",
        default_input_json,
        step_name,
        default_present,
        subsys_index=it0_subsys_nr,
        subsys_number=len(config_json["subsys_nr"]),
        exploration_dep=exploration_type,
    )

    return subsys_timestep, subsys_temp, subsys_exp_time_ps, subsys_max_exp_time_ps, subsys_job_walltime_h, subsys_print_mult, subsys_disturbed_start



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
