from pathlib import Path
import logging
import sys
import copy
import subprocess

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
    backup_and_overwrite_json_file,
    load_default_json_file,
    read_key_input_json,
)
from deepmd_iterative.common.cluster import get_cluster_spec_for_step
from deepmd_iterative.common.file import (
    check_file_existence,
    file_to_strings,
    change_directory,
    write_list_to_file,
)
from deepmd_iterative.common.check import validate_step_folder
from deepmd_iterative.common.slurm import replace_in_slurm_file


def main(
    step_name,
    phase_name,
    deepmd_iterative_path,
    fake_cluster=None,
    input_fn="input.json",
):
    current_path = Path(".").resolve()
    training_path = current_path.parent

    logging.info(f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()}")
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # ### Check if correct folder
    validate_step_folder(step_name)

    # ### Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)

    # ### Get default inputs json
    default_present = False
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "data" / "inputs.json"
    )
    if bool(default_input_json):
        default_present = True

    # ### Get input json (user one)
    if (current_path / input_fn).is_file():
        input_json = load_json_file((current_path / input_fn), True, True)
    else:
        input_json = {}
    new_input_json = copy.deepcopy(input_json)

    # ### Get control path and config_json
    control_apath = training_path / "control"
    config_json = load_json_file((control_apath / "config.json"), True, True)
    training_json = load_json_file(
        (control_apath / f"training_{current_iteration_zfill}.json"), True, True
    )
    jobs_apath = deepmd_iterative_path / "data" / "jobs" / "training"

    # ### Get user cluster keyword
    user_cluster_keyword = read_key_input_json(
        input_json,
        new_input_json,
        "user_cluster_keyword",
        default_input_json,
        step_name,
        default_present,
    )
    user_cluster_keyword = (
        None if isinstance(user_cluster_keyword, bool) else user_cluster_keyword
    )

    # ### Read cluster spec
    (
        cluster,
        cluster_spec,
        cluster_walltime_format,
        cluster_launch_command,
    ) = get_cluster_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "training",
        fake_cluster,
        user_cluster_keyword,
    )
    if fake_cluster is not None:
        logging.info(f"Pretending to be on {fake_cluster}")
    else:
        logging.info(f"Cluster is {cluster}")
    del fake_cluster

    # ### Checks
    if not training_json["is_frozen"]:
        logging.error(f"Lock found. Execute first: training check_freeze")
        logging.error(f"Aborting...")
        return 1

    check_file_existence(
        jobs_apath / f"job_deepmd_compress_{cluster_spec['arch_type']}_{cluster}.sh",
        True,
        True,
        f"No SLURM file present for {step_name.capitalize()} / {phase_name.capitalize()} on this cluster.",
    )
    slurm_file_master = file_to_strings(
        jobs_apath / f"job_deepmd_compress_{cluster_spec['arch_type']}_{cluster}.sh"
    )
    del jobs_apath

    # ### Prep and launch DP Compress
    check = 0
    walltime_approx_s = 7200
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_apath = Path(".").resolve() / str(it_nnp)

        check_file_existence(local_apath / "model.ckpt.index", True, True)

        slurm_email = read_key_input_json(
            input_json,
            new_input_json,
            "slurm_email",
            default_input_json,
            step_name,
            default_present,
        )
        slurm_file = replace_in_slurm_file(
            slurm_file_master,
            training_json,
            cluster_spec,
            walltime_approx_s,
            cluster_walltime_format,
            slurm_email,
        )

        write_list_to_file(
            local_apath
            / f"job_deepmd_compress_{cluster_spec['arch_type']}_{cluster}.sh",
            slurm_file,
        )
        del slurm_file

        with (local_apath / "checkpoint").open("w") as f:
            f.write('model_checkpoint_path: "model.ckpt"\n')
            f.write('all_model_checkpoint_paths: "model.ckpt"\n')
        del f
        if (
            local_apath
            / (
                "job_deepmd_compress_"
                + cluster_spec["arch_type"]
                + "_"
                + cluster
                + ".sh"
            )
        ).is_file():
            change_directory(local_apath)
            try:
                subprocess.call(
                    [
                        cluster_launch_command,
                        f"./job_deepmd_compress_{cluster_spec['arch_type']}_{cluster}.sh",
                    ]
                )
                logging.info(f"DP Compress - {it_nnp} launched")
                check = check + 1
            except FileNotFoundError:
                logging.critical(
                    f"DP Compress - {it_nnp} NOT launched - {training_json['launch_command']} not found"
                )
            change_directory(local_apath.parent)
        else:
            logging.critical(f"DP Compress - {it_nnp} NOT launched - No job file")
        del local_apath

    del it_nnp, slurm_file_master

    # ### Dump the dicts
    logging.info(f"-" * 88)
    write_json_file(config_json, (control_apath / "config.json"), True)
    write_json_file(
        training_json,
        (control_apath / f"training_{current_iteration_zfill}.json"),
        True,
    )
    backup_and_overwrite_json_file(new_input_json, (current_path / input_fn))
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

    # ### Cleaning
    del control_apath
    del config_json
    del current_iteration, current_iteration_zfill
    del training_json
    del training_path, current_path

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
