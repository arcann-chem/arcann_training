import logging
from typing import Dict, List
# Others
from copy import deepcopy
# deepmd_iterative imports
from deepmd_iterative.common.list import (
    remove_strings_containing_substring_in_list_of_strings,
    replace_substring_in_list_of_strings,
)
from deepmd_iterative.common.utils import convert_seconds_to_hh_mm_ss


def replace_in_slurm_file_general(
    slurm_file_master: List[str],
    machine_spec: Dict,
    walltime_approx_s: int,
    machine_walltime_format: str,
    slurm_email: str,
) -> List[str]:

    slurm_file = deepcopy(slurm_file_master)

    slurm_file = replace_substring_in_list_of_strings(
        slurm_file, "_R_PROJECT_", machine_spec["project_name"]
    )

    slurm_file = replace_substring_in_list_of_strings(
        slurm_file, "_R_ALLOC_", machine_spec["allocation_name"]
    )

    slurm_file = (
        remove_strings_containing_substring_in_list_of_strings(
            slurm_file, "_R_PARTITION_"
        )
        if machine_spec["partition"] is None
        else replace_substring_in_list_of_strings(
            slurm_file, "_R_PARTITION_", machine_spec["partition"]
        )
    )

    slurm_file = (
        remove_strings_containing_substring_in_list_of_strings(
            slurm_file, "_R_SUBPARTITION_"
        )
        if machine_spec["subpartition"] is None
        else replace_substring_in_list_of_strings(
            slurm_file, "_R_SUBPARTITION_", machine_spec["subpartition"]
        )
    )

    max_qos_time = 0
    max_qos = 0
    for it_qos in machine_spec["qos"]:
        if machine_spec["qos"][it_qos] >= walltime_approx_s:
            slurm_file = replace_substring_in_list_of_strings(
                slurm_file, "_R_QOS_", it_qos
            )
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
            replace_substring_in_list_of_strings(
                slurm_file, "_R_WALLTIME_", convert_seconds_to_hh_mm_ss(max_qos_time)
            )
            if "hours" in machine_walltime_format
            else replace_substring_in_list_of_strings(
                slurm_file, "_R_WALLTIME_", str(max_qos_time)
            )
        )
    else:
        slurm_file = (
            replace_substring_in_list_of_strings(
                slurm_file,
                "_R_WALLTIME_",
                convert_seconds_to_hh_mm_ss(walltime_approx_s),
            )
            if "hours" in machine_walltime_format
            else replace_substring_in_list_of_strings(
                slurm_file, "_R_WALLTIME_", str(walltime_approx_s)
            )
        )

        if slurm_email != "":
            slurm_file = replace_substring_in_list_of_strings(
                slurm_file, "_R_EMAIL_", slurm_email
            )
        else:
            slurm_file = remove_strings_containing_substring_in_list_of_strings(
                slurm_file, "_R_EMAIL_"
            )
            slurm_file = remove_strings_containing_substring_in_list_of_strings(
                slurm_file, "mail"
            )
        del slurm_email

    return slurm_file
