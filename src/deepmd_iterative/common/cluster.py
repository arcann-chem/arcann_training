from pathlib import Path
import logging
import sys
import socket
from typing import Any, Dict, List, Tuple, Union

# deepmd_iterative imports
from deepmd_iterative.common.json import load_json_file


import socket


def get_host_name() -> str:
    """Returns the fully-qualified hostname of the current machine.

    This function first gets the hostname of the current machine using the `socket.gethostname()` function. If the hostname
    contains a period, it is already fully-qualified and can be returned immediately. Otherwise, the function uses the
    `socket.gethostbyaddr()` function to look up the fully-qualified hostname.

    Returns:
        str: The fully-qualified hostname of the current machine.
    """
    socket.setdefaulttimeout(2)
    hostname = socket.gethostname()
    if "." in hostname:
        # Hostname is already fully-qualified
        return hostname
    else:
        # Look up fully-qualified hostname using socket.gethostbyaddr()
        try:
            hostname = socket.gethostbyaddr(hostname)[0]
            return hostname
        except socket.timeout:
            return hostname


def assert_same_cluster(expected_cluster: str, cluster_config: Dict) -> None:
    """
    Check if the cluster name in the provided dictionary matches the expected cluster name. If the names do not match,
    an error is logged and the execution is aborted.

    Args:
        expected_cluster (str): The name of the expected cluster.
        cluster_info (dict): A dictionary containing the cluster information, with a "cluster" key representing the cluster name.

    Returns:
        None

    Raises:
        SystemExit: If the cluster name in the dictionary does not match the expected cluster name.
    """
    # Check if the provided cluster name matches the expected cluster name
    if cluster_config["cluster"] != expected_cluster:
        # If not, log an error message and abort the execution
        error_msg = f"Provided cluster {cluster_config['cluster']} does not match expected cluster {expected_cluster}"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)


def get_cluster_config_files(
    deepmd_iterative_path: Path, training_path: Path
) -> List[Dict]:
    """Finds and returns a list of dictionaries containing cluster configurations for all clusters.

    Args:
        deepmd_iterative_path (Path): The path to the 'deepmd_iterative' directory.
        training_path (Path): The path to the training directory.

    Returns:
        List[Dict]: A list of dictionaries, each containing the contents of a 'cluster_config.json' file.

    Raises:
        FileNotFoundError: If no 'cluster_config.json' file is found in the given directories.
    """

    # List of dictionaries containing cluster configurations.
    cluster_configs = []

    # Check for 'cluster_config.json' file in the training directory.
    training_config_path = training_path / "files" / "cluster_config.json"
    if training_config_path.is_file():
        cluster_configs.append(load_json_file(training_config_path))

    # Check for 'cluster_config.json' file in the deepmd_iterative directory.
    default_config_path = deepmd_iterative_path / "data" / "cluster_config.json"
    if default_config_path.is_file():
        cluster_configs.append(load_json_file(default_config_path))

    # If no 'cluster_config.json' file is found, raise a FileNotFoundError.
    if not cluster_configs:
        error_msg = (
            "No 'cluster_config.json' file found. Please check the installation."
        )
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)
        # raise FileNotFoundError(error_msg)

    return cluster_configs


def get_cluster_from_configs(
    cluster_configs: List[Dict], input_cluster_shortname: str = ""
) -> str:
    """
    Given a list of cluster configuration dictionaries and an optional input cluster name,
    returns the name of the cluster that matches the current hostname or the input cluster name.

    Args:
        cluster_configs (List[Dict]): List of cluster configuration dictionaries.
        cluster_short_name (str, optional): The name of the cluster to use. If not provided, the cluster with a matching hostname will be used.

    Returns:
        str: Name of the cluster that matches the current hostname or input cluster name.

    Raises:
        ValueError: If no cluster specification is found that matches the current hostname or input cluster name.
    """
    if not input_cluster_shortname:
        cluster_hostname = get_host_name()
        for cluster_config in cluster_configs:
            for cluster_short_name in cluster_config.keys():
                if cluster_config[cluster_short_name]["hostname"] in cluster_hostname:
                    return cluster_short_name
    else:
        for cluster_config in cluster_configs:
            if input_cluster_shortname in cluster_config:
                return input_cluster_shortname

    error_msg = f"No matching cluster found for hostname {get_host_name()} and no input cluster specified"
    logging.error(f"{error_msg}\nAborting...")
    sys.exit(1)
    # raise ValueError(error_msg)


def get_cluster_spec_for_step(
    deepmd_iterative_path: Path,
    training_path: Path,
    step: str,
    input_cluster_shortname: str = None,
    user_cluster_keyword: Union[str, List[str]] = None,
    check_only: bool = False,
) -> Tuple[str, Dict[str, Any], str, str]:
    """Return the cluster specification for the given step and cluster.

    Args:
        deepmd_iterative_path (Path): The path to the DeepMD-Iterative root directory.
        training_path (Path): The path to the training directory.
        step (str): The name of the step for which to get the cluster specification.
        input_cluster_shortname (str, optional): The short name of the cluster for which to get the specification. Defaults to "".
        user_cluster_keyword (Union[str, List[str]], optional): A keyword or list of keywords to use when searching for a matching configuration. Defaults to None.
        check_only (bool, optional): Whether to only check for a matching configuration without returning the cluster specification. Defaults to False.

    Returns:
        Tuple[str, dict, str, str, int]: A tuple containing the following elements:
            - cluster_shortname: The short name of the cluster.
            - cluster_spec: The cluster specification as a dictionary.
            - cluster_walltime_format: The walltime format of the cluster.
            - cluster_launch_command: The launch command to use on the cluster.
    """

    # Get a list of all cluster configuration files
    cluster_configs = get_cluster_config_files(deepmd_iterative_path, training_path)

    # Get the short name of the cluster to use
    cluster_shortname = get_cluster_from_configs(
        cluster_configs, input_cluster_shortname
    )

    # If check_only is True, return an empty cluster specification
    if check_only:
        return cluster_shortname, [], "", ""

    # Iterate over all cluster configurations
    for config in cluster_configs:
        # Iterate over all keys in the configuration for the selected cluster
        for config_key, config_data in config.get(cluster_shortname, {}).items():
            # Skip keys that are not relevant to the cluster specification
            if config_key not in ["hostname", "walltime_format", "launch_command"]:
                # Check if the current keyword matches the user keyword
                if (
                    user_cluster_keyword is None
                    or (
                        isinstance(user_cluster_keyword, str)
                        and user_cluster_keyword == config_key
                    )
                    or (
                        isinstance(user_cluster_keyword, list)
                        and len(user_cluster_keyword) == 3
                        and user_cluster_keyword[0] == config_data.get("project_name")
                        and user_cluster_keyword[1]
                        == config_data.get("allocation_name")
                        and user_cluster_keyword[2] == config_data.get("arch_name")
                    )
                ):
                    # Check if the step is valid for the current configuration
                    if step in config_data.get("valid_for", []):
                        # Return the cluster specification
                        return (
                            cluster_shortname,
                            config_data,
                            config[cluster_shortname]["walltime_format"],
                            config[cluster_shortname]["launch_command"],
                        )

    # If no matching configuration was found, return an error
    if user_cluster_keyword is not None and not (
        isinstance(user_cluster_keyword, str)
        or (isinstance(user_cluster_keyword, list) and len(user_cluster_keyword) == 3)
    ):
        error_msg = "Invalid user_cluster_keyword. Please provide either a string or a list of length 3."
    elif user_cluster_keyword is not None and (
        isinstance(user_cluster_keyword, list) and len(user_cluster_keyword) == 3
    ):
        error_msg = f"User keyword '{user_cluster_keyword}' not found in any configuration files."
    elif not cluster_configs:
        error_msg = "No cluster configuration files found."
    elif user_cluster_keyword is not None and not any(
        user_cluster_keyword in config for config in cluster_configs
    ):
        error_msg = f"User keyword '{user_cluster_keyword}' not found in any configuration files."
    elif input_cluster_shortname is not None and cluster_shortname not in [
        config.get("name") for config in cluster_configs
    ]:
        error_msg = (
            f"No configuration found for input cluster {input_cluster_shortname}."
        )
    elif user_cluster_keyword is not None and not any(
        user_cluster_keyword in config for config in cluster_configs
    ):
        error_msg = f"User keyword '{user_cluster_keyword}' not found in any configuration files."
    else:
        error_msg = f"No default configuration found for step '{step}' and cluster '{cluster_shortname}'."
    logging.error(f"{error_msg}\nAborting...")
    sys.exit(1)
