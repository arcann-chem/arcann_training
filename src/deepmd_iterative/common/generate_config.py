

from typing import Dict, Tuple

from deepmd_iterative.common.json import read_key_input_json, load_json_file

def generate_config_json(input_json: Dict, new_input_json: Dict, default_input_json: Dict, step_name: str, default_present: bool) -> Tuple[Dict, str]:
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
    for it_subsys_nr, subsys_nr in enumerate(read_key_input_json(
            input_json,
            new_input_json,
            "subsys_nr",
            default_input_json,
            step_name,
            default_present,
        )):
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
    machine_launch_command: str
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
    training_json = load_json_file((control_path / f"training_{current_iteration_zfill}.json"), abort_on_error=False)
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