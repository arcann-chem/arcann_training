"""
Created: 2023/01/01
Last modified: 2023/03/27

The slurm module provides functions to manipulate SLURM data (as list of strings).

Functions
---------
replace_in_slurm_file_general(slurm_file_master: List[str], machine_spec: Dict, walltime_approx_s: int, machine_walltime_format: str, slurm_email: str) -> List[str]
    A function to return a modified version of the provided Slurm file, with certain strings replaced based on the specified machine specifications and walltime.
"""
# Standard library modules
import logging
from copy import deepcopy
from typing import Dict, List

# Local imports
from deepmd_iterative.common.utils import (
    catch_errors_decorator,
    convert_seconds_to_hh_mm_ss,
)
from deepmd_iterative.common.list import (
    exclude_substring_from_string_list,
    replace_substring_in_string_list,
)


@catch_errors_decorator
def replace_in_slurm_file_general(
    slurm_file_master: List[str],
    machine_spec: Dict,
    walltime_approx_s: int,
    machine_walltime_format: str,
    slurm_email: str,
) -> List[str]:
    """
    Return a modified version of the provided Slurm file, with certain strings replaced based on the specified machine
    specifications and walltime.

    Parameters
    ----------
    slurm_file_master : List[str]
        The original Slurm file to modify.
    machine_spec : Dict
        A dictionary containing the machine specifications, including the project name, allocation name,
        partition, subpartition, and QoS.
    walltime_approx_s : int
        The approximate walltime to use for the job, in seconds.
    machine_walltime_format : str
        The format to use for the walltime specification in the Slurm file.
    slurm_email : str
        The email address to use for Slurm notifications.

    Returns
    -------
    List[str]
        A modified version of the original Slurm file, with certain strings replaced based on the specified machine
        specifications and walltime.

    Raises
    ------
    None
    """
    slurm_file = deepcopy(slurm_file_master)

    slurm_file = replace_substring_in_string_list(
        slurm_file, "_R_PROJECT_", machine_spec["project_name"]
    )

    slurm_file = replace_substring_in_string_list(
        slurm_file, "_R_ALLOC_", machine_spec["allocation_name"]
    )

    slurm_file = (
        exclude_substring_from_string_list(slurm_file, "_R_PARTITION_")
        if machine_spec["partition"] is None
        else replace_substring_in_string_list(
            slurm_file, "_R_PARTITION_", machine_spec["partition"]
        )
    )

    slurm_file = (
        exclude_substring_from_string_list(slurm_file, "_R_SUBPARTITION_")
        if machine_spec["subpartition"] is None
        else replace_substring_in_string_list(
            slurm_file, "_R_SUBPARTITION_", machine_spec["subpartition"]
        )
    )

    max_qos_time = 0
    max_qos = 0
    for it_qos in machine_spec["qos"]:
        if machine_spec["qos"][it_qos] >= walltime_approx_s:
            slurm_file = replace_substring_in_string_list(slurm_file, "_R_QOS_", it_qos)
            qos_ok = True
        else:
            max_qos = it_qos if machine_spec["qos"][it_qos] > max_qos_time else max_qos
            qos_ok = False
    del it_qos

    if not qos_ok:
        logging.warning(
            "Approximate wall time superior than the maximun time allowed by the QoS"
        )
        logging.warning("Settign the maximum QoS time as walltime")
        slurm_file = (
            replace_substring_in_string_list(
                slurm_file, "_R_WALLTIME_", convert_seconds_to_hh_mm_ss(max_qos_time)
            )
            if "hours" in machine_walltime_format
            else replace_substring_in_string_list(
                slurm_file, "_R_WALLTIME_", str(max_qos_time)
            )
        )
    else:
        slurm_file = (
            replace_substring_in_string_list(
                slurm_file,
                "_R_WALLTIME_",
                convert_seconds_to_hh_mm_ss(walltime_approx_s),
            )
            if "hours" in machine_walltime_format
            else replace_substring_in_string_list(
                slurm_file, "_R_WALLTIME_", str(walltime_approx_s)
            )
        )

        if slurm_email != "":
            slurm_file = replace_substring_in_string_list(
                slurm_file, "_R_EMAIL_", slurm_email
            )
        else:
            slurm_file = exclude_substring_from_string_list(slurm_file, "_R_EMAIL_")
            slurm_file = exclude_substring_from_string_list(slurm_file, "mail")
        del slurm_email

    return slurm_file
