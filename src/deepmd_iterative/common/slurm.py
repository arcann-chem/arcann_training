import logging
import copy


# deepmd_iterative imports
from deepmd_iterative.common.list import (
    replace_substring_in_list,
    remove_strings_containing_substring_in_list,
)
from deepmd_iterative.common.tools import convert_seconds_to_hh_mm_ss


def replace_in_slurm_file(
    slurm_file_master,
    training_json,
    cluster_spec,
    walltime_approx_s,
    cluster_walltime_format,
    slurm_email,
):
    slurm_file = copy.deepcopy(slurm_file_master)

    slurm_file = replace_substring_in_list(
        slurm_file, "_R_DEEPMD_VERSION_", str(training_json["deepmd_model_version"])
    )

    slurm_file = replace_substring_in_list(
        slurm_file, "_R_PROJECT_", cluster_spec["project_name"]
    )

    slurm_file = replace_substring_in_list(
        slurm_file, "_R_ALLOC_", cluster_spec["allocation_name"]
    )

    slurm_file = (
        remove_strings_containing_substring_in_list(slurm_file, "_R_PARTITION_")
        if cluster_spec["partition"] is None
        else replace_substring_in_list(
            slurm_file, "_R_PARTITION_", cluster_spec["partition"]
        )
    )

    slurm_file = (
        remove_strings_containing_substring_in_list(slurm_file, "_R_SUBPARTITION_")
        if cluster_spec["subpartition"] is None
        else replace_substring_in_list(
            slurm_file, "_R_SUBPARTITION_", cluster_spec["subpartition"]
        )
    )

    max_qos_time = 0
    max_qos = 0
    for it_qos in cluster_spec["qos"]:
        if cluster_spec["qos"][it_qos] >= walltime_approx_s:
            slurm_file = replace_substring_in_list(slurm_file, "_R_QOS_", it_qos)
            qos_ok = True
        else:
            max_qos = it_qos if cluster_spec["qos"][it_qos] > max_qos_time else max_qos
            qos_ok = False
    del it_qos

    if not qos_ok:
        logging.warning(
            "Approximate wall time superior than the maximun time allowed by the QoS"
        )
        logging.warning("Settign the maximum QoS time as walltime")
        slurm_file = (
            replace_substring_in_list(
                slurm_file, "_R_WALLTIME_", convert_seconds_to_hh_mm_ss(max_qos_time)
            )
            if "hours" in cluster_walltime_format
            else replace_substring_in_list(
                slurm_file, "_R_WALLTIME_", str(max_qos_time)
            )
        )
    else:
        slurm_file = (
            replace_substring_in_list(
                slurm_file,
                "_R_WALLTIME_",
                convert_seconds_to_hh_mm_ss(walltime_approx_s),
            )
            if "hours" in cluster_walltime_format
            else replace_substring_in_list(
                slurm_file, "_R_WALLTIME_", str(walltime_approx_s)
            )
        )

        if slurm_email != "":
            slurm_file = replace_substring_in_list(slurm_file, "_R_EMAIL_", slurm_email)
        else:
            slurm_file = remove_strings_containing_substring_in_list(
                slurm_file, "_R_EMAIL_"
            )
            slurm_file = remove_strings_containing_substring_in_list(slurm_file, "mail")
        del slurm_email

    return slurm_file
