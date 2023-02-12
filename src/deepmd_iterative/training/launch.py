from pathlib import Path
import logging
import sys
import copy
import subprocess

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    json_read,
    json_dump,
    read_default_input_json,
    read_key_input_json,
)
from deepmd_iterative.common.files import (
    change_dir,
)

from deepmd_iterative.common.clusters import clusterize, check_same_cluster


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

    # ### Check if correct folder
    if step_name not in current_apath.name:
        logging.error(f"The folder doesn't seems to be for this step: {step_name.capitalize()}")
        logging.error(f"Aborting...")
        return 1

    # ### Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)

    # ### Get default inputs json
    default_present = False
    default_input_json = read_default_input_json(
        deepmd_iterative_apath / "data" / "inputs.json"
    )
    if bool(default_input_json):
        default_present = True

    # ### Get input json (user one)
    if (current_apath / input_fn).is_file():
        input_json = json_read((current_apath / input_fn), True, True)
    else:
        input_json = {}
    new_input_json = copy.deepcopy(input_json)

    # ### Get control path and config_json
    control_apath = training_iterative_apath / "control"
    config_json = json_read((control_apath / "config.json"), True, True)
    training_json = json_read(
        (control_apath / f"training_{current_iteration_zfill}.json"), True, True
    )

    # ### Get machine info
    user_spec = read_key_input_json(
        input_json,
        new_input_json,
        "user_spec",
        default_input_json,
        step_name,
        default_present,
    )
    user_spec = None if isinstance(user_spec, bool) else user_spec

    # ### Read cluster info
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
        check_only=True,
    )
    if fake_cluster is not None:
        logging.info(f"Pretending to be on {fake_cluster}")
    else:
        logging.info(f"Cluster is {cluster}")
    del fake_cluster

    # ### Check prep/launch
    check_same_cluster(cluster, training_json)

    # ### Checks
    if training_json["is_launched"]:
        logging.critical(f"Already launched.")
        continuing = input(
            f"Should it be run again? (Y for Yes, anything else to abort)"
        )
        if continuing == "Y":
            del continuing
        else:
            logging.error(f"Aborting...")
            return 1
    if not training_json["is_locked"]:
        logging.error(f"Lock found. Execute first: training preparation")
        logging.error(f"Aborting...")
        return 1

    # ### Launch the jobs
    check = 0
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_apath = Path(".").resolve() / str(it_nnp)
        if (
            local_apath
            / f"job_deepmd_train_{training_json['arch_type']}_{cluster}.sh"
        ).is_file():
            change_dir(local_apath)
            try:
                subprocess.call(
                    [
                        training_json["launch_command"],
                        f"./job_deepmd_train_{training_json['arch_type']}_{cluster}.sh",
                    ]
                )
                logging.info(f"DP Train - {it_nnp} launched")
                check = check + 1
            except FileNotFoundError:
                logging.critical(f"DP Train - {it_nnp} NOT launched - {training_json['launch_command']} not found")
            change_dir(local_apath.parent)
        else:
            logging.critical(f"DP Train - {it_nnp} NOT launched - No job file")
        del local_apath
    del it_nnp

    if check == config_json["nb_nnp"]:
        training_json["is_launched"] = True

    json_dump(
        training_json,
        (control_apath / f"training_{current_iteration_zfill}.json"),
        True,
    )

    logging.info(f"-" * 88)
    if check == config_json["nb_nnp"]:
        logging.info(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
        )
    else:
        logging.critical(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is semi-succes !"
        )
        logging.critical(f"Some SLURM jobs did not launch correctly")
        logging.critical(f"Please launch manually before continuing to the next step")
        logging.critical(
            f'Replace the key "is_launched" to True in the training_{current_iteration_zfill}.json.'
        )
    del check

    # ### Cleaning
    del control_apath
    del input_json, default_input_json, default_present, new_input_json
    del config_json
    del current_iteration, current_iteration_zfill
    del training_json
    del cluster
    del training_iterative_apath, current_apath

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "launch",
            Path(sys.argv[1]),
            fake_cluster=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
