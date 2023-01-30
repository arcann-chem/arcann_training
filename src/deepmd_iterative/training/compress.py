from pathlib import Path
import logging
import sys
import copy
import subprocess

### deepmd_iterative imports
from deepmd_iterative.common.json import (
    json_read,
    json_dump,
    json_dump_bak,
    read_default_input_json,
    read_key_input_json,
)
from deepmd_iterative.common.lists import replace_in_list, delete_in_list
from deepmd_iterative.common.clusters import clusterize
from deepmd_iterative.common.files import (
    check_file,
    file_to_strings,
    change_dir,
    write_file,
)
from deepmd_iterative.common.tools import seconds_to_walltime


def main(
    step_name,
    phase_name,
    deepmd_iterative_apath,
    fake_cluster=None,
    input_fn="input.json",
):
    current_apath = Path(".").resolve()
    training_iterative_apath = current_apath.parent

    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()}")
    logging.debug(f"Current path :{current_apath}")
    logging.debug(f"Training path: {training_iterative_apath}")
    logging.debug(f"Program path: {deepmd_iterative_apath}")
    logging.info(f"-" * 88)

    ### Check if correct folder
    if step_name not in current_apath.name:
        logging.error(f"The folder doesn't seems to be for this step: {step_name.capitalize()}")
        logging.error(f"Aborting...")
        return 1

    ### Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)

    ### Get default inputs json
    default_present = False
    default_input_json = read_default_input_json(
        deepmd_iterative_apath / "data" / "inputs.json"
    )
    if bool(default_input_json):
        default_present = True

    ### Get input json (user one)
    if (current_apath / input_fn).is_file():
        input_json = json_read((current_apath / input_fn), True, True)
    else:
        input_json = {}
    new_input_json = copy.deepcopy(input_json)

    ### Get control path and config_json
    control_apath = training_iterative_apath / "control"
    config_json = json_read((control_apath / "config.json"), True, True)
    training_json = json_read(
        (control_apath / (f"training_{current_iteration_zfill}.json")), True, True
    )
    jobs_apath = deepmd_iterative_apath / "data" / "jobs" / "training"


### Get machine info
    user_spec = read_key_input_json(
        input_json,
        new_input_json,
        "user_spec",
        default_input_json,
        step_name,
        default_present,
    )
    user_spec = None if isinstance(user_spec, bool) else user_spec

    ### Read cluster info
    (
        cluster,
        cluster_spec,
        cluster_walltime_format,
        cluster_launch_command,
        cluster_error,
    ) = clusterize(
        deepmd_iterative_apath,
        training_iterative_apath,
        step="training",
        input_cluster=fake_cluster,
        user_keyword=user_spec,
    )
    if fake_cluster is not None:
        logging.info(f"Pretending to be on {fake_cluster}")
    else:
        logging.info(f"Cluster is {cluster}")
    del fake_cluster
    if cluster_error != 0:
        ### #FIXME: Better errors for clusterize
        logging.error(f"Error in machine_file.json")
        logging.error(f"Aborting...")
        return 1
    del cluster_error

    ### Checks
    if not training_json["is_frozen"]:
        logging.error(f"Lock found. Run/Check first: training checkfreeze")
        logging.error(f"Aborting...")
        return 1

    check_file(
        jobs_apath / (f"job_deepmd_compress_{cluster_spec['arch_type']}_{cluster}.sh"),
        True,
        True,
        f"No SLURM file present for {step_name.capitalize()} / {phase_name.capitalize()} on this cluster.",
    )
    slurm_file_master = file_to_strings(
        jobs_apath / (f"job_deepmd_compress_{cluster_spec['arch_type']}_{cluster}.sh")
    )
    del jobs_apath

    ### Prep and launch DP Compress
    check = 0
    walltime_approx_s = 7200
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_apath = Path(".").resolve()/str(it_nnp)


        check_file(local_apath/"model.ckpt.index",True,True)

        slurm_file = copy.deepcopy(slurm_file_master)
        slurm_file = replace_in_list(
            slurm_file, "_R_DEEPMD_VERSION_", str(training_json["deepmd_model_version"])
        )

        slurm_file = replace_in_list(
            slurm_file, "_R_DEEPMD_MODEL_", f"graph_{it_nnp}_{current_iteration_zfill}"
        )

        slurm_file = replace_in_list(
            slurm_file, "_R_PROJECT_", cluster_spec["project_name"]
        )
        slurm_file = replace_in_list(
            slurm_file, "_R_ALLOC_", cluster_spec["allocation_name"]
        )
        slurm_file = (
            delete_in_list(slurm_file, "_R_PARTITON_")
            if cluster_spec["partition"] is None
            else replace_in_list(slurm_file, "_R_PARTITION_", cluster_spec["partition"])
        )
        slurm_file = (
            delete_in_list(slurm_file, "_R_SUBPARTITION_")
            if cluster_spec["subpartition"] is None
            else replace_in_list(
                slurm_file, "_R_SUBPARTITION_", cluster_spec["subpartition"]
            )
        )
        max_qos_time = 0
        max_qos = 0
        for it_qos in cluster_spec["qos"]:
            if cluster_spec["qos"][it_qos] >= walltime_approx_s:
                slurm_file = replace_in_list(slurm_file, "_R_QOS_", it_qos)
                qos_ok = True
            else:
                max_qos = (
                    it_qos if cluster_spec["qos"][it_qos] > max_qos_time else max_qos
                )
                qos_ok = False
        del it_qos
        if not qos_ok:
            logging.warning(
                "Approximate wall time superior than the maximun time allowed by the QoS"
            )
            logging.warning("Settign the maximum QoS time as walltime")
            slurm_file = (
                replace_in_list(
                    slurm_file, "_R_WALLTIME_", seconds_to_walltime(max_qos_time)
                )
                if "hours" in cluster_walltime_format
                else replace_in_list(slurm_file, "_R_WALLTIME_", str(max_qos_time))
            )
        else:
            slurm_file = (
                replace_in_list(
                    slurm_file, "_R_WALLTIME_", seconds_to_walltime(walltime_approx_s)
                )
                if "hours" in cluster_walltime_format
                else replace_in_list(slurm_file, "_R_WALLTIME_", str(walltime_approx_s))
            )
        del qos_ok, max_qos_time, max_qos

        slurm_email = read_key_input_json(
            input_json,
            new_input_json,
            "slurm_email",
            default_input_json,
            step_name,
            default_present,
        )
        if slurm_email != "":
            slurm_file = replace_in_list(slurm_file, "_R_EMAIL_", slurm_email)
        else:
            slurm_file = delete_in_list(slurm_file, "_R_EMAIL_")
            slurm_file = delete_in_list(slurm_file, "mail")
        del slurm_email

        write_file(
            local_apath
            / (f"job_deepmd_freeze_{cluster_spec['arch_type']}_{cluster}.sh"),
            slurm_file,
        )
        del slurm_file

        with (local_apath/"checkpoint").open("w") as f:
            f.write("model_checkpoint_path: \"model.ckpt\"\n")
            f.write("all_model_checkpoint_paths: \"model.ckpt\"\n")
        del f
        if (local_apath/("job_deepmd_compress_"+cluster_spec["arch_type"]+"_"+cluster+".sh")).is_file():
            change_dir(local_apath)
            subprocess.call(["sbatch","./job_deepmd_compress_"+cluster_spec["arch_type"]+"_"+cluster+".sh"])
            change_dir(local_apath.parent)
            logging.info(f"DP Compress - {it_nnp} launched")
            check = check + 1
        else:
            logging.critical(f"DP Compress - {it_nnp} NOT launched")
        del local_apath

    del it_nnp, slurm_file, slurm_file_master


    ## Dump the dicts
    logging.info(f"-" * 88)
    json_dump(config_json, (control_apath / "config.json"), True)
    json_dump(
        training_json,
        (control_apath / (f"training_{current_iteration_zfill}.json")),
        True,
    )
    json_dump_bak(new_input_json, (current_apath / input_fn), True)
    logging.info(f"-" * 88)
    if check == config_json["nb_nnp"]:
        pass
    else:
        logging.critical(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is semi-succes !"
        )
        logging.critical(f"Some SLURM jobs did not launch correctly")
        logging.critical(f"Please launch manually before continuing to the next step")
    del check

    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
    )

    #### Cleaning
    del control_apath
    del config_json
    del current_iteration, current_iteration_zfill
    del training_json
    del training_iterative_apath, current_apath

    return 0

if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "compress",
            Path(sys.argv[1]),
            fake_cluster=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
