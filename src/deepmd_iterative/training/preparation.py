from pathlib import Path
import logging
import sys
import copy
import subprocess
import random

# ### Non-standard imports
import numpy as np

# ### deepmd_iterative imports
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
    check_dir,
    write_file,
)
from deepmd_iterative.common.training import (
    get_decay_rate,
    get_decay_steps,
    check_initial_datasets,
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

    # ### Check if correct folder
    if step_name not in current_apath.name:
        logging.error(f"The folder doesn't seems to be for this step: {step_name.capitalize()}")
        logging.critical("Aborting...")
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

    # ### Get extra needed paths
    jobs_apath = deepmd_iterative_apath / "data" / "jobs" / "training"

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
    )
    if fake_cluster is not None:
        logging.info(f"Pretending to be on {fake_cluster}")
    else:
        logging.info(f"Cluster is {cluster}")
    del fake_cluster
    if cluster_error != 0:
        # ### #FIXME: Better errors for clusterize
        logging.error(f"Error in machine_file.json")
        logging.error(f"Aborting...")
        return 1
    del cluster_error

    # ### Checks
    if current_iteration > 0:
        labeling_json = json_read(
            (control_apath / f"labeling_{current_iteration_zfill}.json"), True, True
        )
        if not labeling_json["is_extracted"]:
            logging.error("Lock found. Run/Check first: labeling extract")
            logging.error("Aborting...")
            return 1

    # ### Get/Create training parameters
    training_json = json_read(
        (control_apath / f"training_{current_iteration_zfill}.json"), False, True
    )
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

    training_json["cluster"] = cluster
    training_json["project_name"] = cluster_spec["project_name"]
    training_json["allocation_name"] = cluster_spec["allocation_name"]
    training_json["arch_name"] = cluster_spec["arch_name"]
    training_json["arch_type"] = cluster_spec["arch_type"]
    training_json["launch_command"] = cluster_launch_command

    check_file(
        jobs_apath / f"job_deepmd_train_{cluster_spec['arch_type']}_{cluster}.sh",
        True,
        True,
        f"No SLURM file present for {step_name.capitalize()} / {phase_name.capitalize()} on this cluster.",
    )
    slurm_file_master = file_to_strings(
        jobs_apath / f"job_deepmd_train_{cluster_spec['arch_type']}_{cluster}.sh"
    )
    del jobs_apath

    # ### Check DeePMD version
    if training_json["deepmd_model_version"] not in [2.0, 2.1]:
        logging.critical(
            f"Invalid deepmd model version (2.0 or 2.1): {training_json['deepmd_model_version']}"
        )
        logging.critical("Aborting...")
        return 1

    # ### Check DeePMD descriptor type
    if training_json["deepmd_model_type_descriptor"] not in ["se_e2_a"]:
        logging.critical(
            f"Invalid deepmd type descriptor (se_e2_a): {training_json['deepmd_model_type_descriptor']}"
        )
        logging.critical("Aborting...")
        return 1

    # ### Check mismatch between cluster/arch_name/arch and DeePMD
    if training_json["deepmd_model_version"] < 2.0:
        logging.critical("Only version >= 2.0 on Jean Zay!")
        logging.critical("Aborting...")
        return 1
    if (
            training_json["deepmd_model_version"] < 2.1
            and training_json["arch_name"] == "a100"
    ):
        logging.critical("Only version >= 2.1 on Jean Zay A100 !")
        logging.critical("Aborting...")
        return 1

    # ### Check if the default input json file exists
    input_file_fpath = (
            training_iterative_apath
            / "files"
            / (
                f"dptrain_{training_json['deepmd_model_version']}_{training_json['deepmd_model_type_descriptor']}.json"
            )
    ).resolve()
    training_input_json = json_read(input_file_fpath, True, True)
    config_json["type_map"] = {}
    config_json["type_map"] = training_input_json["model"]["type_map"]
    del input_file_fpath

    # ### Check the initial sets json file
    datasets_initial_json = check_initial_datasets(training_iterative_apath)

    # ### Let us find what is in data
    data_apath = training_iterative_apath / "data"
    check_dir(data_apath, True)
    subsys_name = []

    datasets_extra = []
    datasets_validation = []
    for it_data_folders in data_apath.iterdir():
        if it_data_folders.is_dir():
            # ### Escape initial/extra sets, because initial get added first and extra as last, and also escape init_
            # not in initial_json (in case of removal)
            if (
                    it_data_folders.name not in datasets_initial_json.keys()
                    and "extra_" != it_data_folders.name[:6]
                    and "init_" != it_data_folders.name[:5]
            ):
                # ### Escape test sets
                if "test_" != it_data_folders.name[:5]:
                    # ### Escape if set iter is superior as iter, it is only for reprocessing old stuff.
                    try:
                        if (
                                int(it_data_folders.name.rsplit("_", 1)[-1])
                                <= current_iteration
                        ):
                            subsys_name.append(it_data_folders.name.rsplit("_", 1)[0])
                    # ### #TODO: Better except clause
                    except:
                        pass
                else:
                    datasets_validation.append(it_data_folders.name)
            # ### Get the extra sets !
            elif "extra_" == it_data_folders.name[:6]:
                datasets_extra.append(it_data_folders.name)
    del it_data_folders

    del datasets_validation

    # ### Training sets list construction
    datasets_training = []
    datasets_training_json = []
    # ### Initial
    nb_initial = 0
    if training_json["use_initial_datasets"]:
        for it_datasets_initial_json in datasets_initial_json.keys():
            if (data_apath / it_datasets_initial_json).is_dir():
                # ### #TODO: Here we don't Path because too complex
                datasets_training.append(f"{(Path(data_apath.parts[-1]) / 'it_datasets_initial_json' / '_')}"[:-1])
                # datasets_training.append(f"data/{it_datasets_initial_json}/")
                datasets_training_json.append(it_datasets_initial_json)
                nb_initial = (
                        nb_initial + datasets_initial_json[it_datasets_initial_json]
                )
        del it_datasets_initial_json
    del datasets_initial_json

    # ### Non-Reactive (aka subsys_nr in the initialization first) && all the others are REACTIVE !
    # ### Total and what is added just for this iteration
    nb_added_nr = 0
    nb_added_r = 0
    nb_added_nr_iter = 0
    nb_added_r_iter = 0

    # ### This trick remove duplicates from list via set
    subsys_name = list(set(subsys_name))
    subsys_name = [i for i in subsys_name if i not in config_json["subsys_nr"]]
    subsys_name = [
        i
        for i in subsys_name
        if i not in [zzz + "-disturbed" for zzz in config_json["subsys_nr"]]
    ]
    subsys_name = sorted(subsys_name)
    config_json["subsys_r"] = subsys_name
    del subsys_name

    if current_iteration > 0:
        for it_iteration in np.arange(1, current_iteration + 1):
            it_iteration_zfill = str(it_iteration).zfill(3)
            try:
                for system_it in config_json["subsys_nr"]:
                    if (
                            data_apath / f"{system_it}_{it_iteration_zfill}"
                    ).is_dir():
                        # ### #TODO: Here we don't Path because too complex
                        datasets_training.append(
                            f"data/{system_it}_{it_iteration_zfill}/"
                        )
                        datasets_training_json.append(
                            f"{system_it}_{it_iteration_zfill}"
                        )
                        nb_added_nr = (
                            nb_added_nr
                            + np.load(
                                str(
                                    data_apath
                                    / f"{system_it}_{it_iteration_zfill}"
                                    / "set.000"
                                    / "box.npy"
                                )
                            ).shape[0]
                        )
                        if it_iteration == current_iteration:
                            nb_added_nr_iter = (
                                nb_added_nr_iter
                                + np.load(
                                    str(
                                        data_apath
                                        / f"{system_it}_{it_iteration_zfill}"
                                        / "set.000"
                                        / "box.npy"
                                    )
                                ).shape[0]
                            )
                del system_it
            except (KeyError, NameError):
                pass
            try:
                for system_it in [
                    zzz + "-disturbed" for zzz in config_json["subsys_nr"]
                ]:
                    if (
                            data_apath / f"{system_it}_{it_iteration_zfill}"
                    ).is_dir():
                        # ### #TODO: Here we don't Path because too complex
                        datasets_training.append(
                            f"data/{system_it}_{it_iteration_zfill}/"
                        )
                        datasets_training_json.append(
                            f"{system_it}_{it_iteration_zfill}"
                        )
                        nb_added_nr = (
                            nb_added_nr
                            + np.load(
                                str(
                                    data_apath
                                    / f"{system_it}_{it_iteration_zfill}"
                                    / "set.000"
                                    / "box.npy"
                                )
                            ).shape[0]
                        )
                        if it_iteration == current_iteration:
                            nb_added_nr_iter = (
                                nb_added_nr_iter
                                + np.load(
                                    str(
                                        data_apath
                                        / f"{system_it}_{it_iteration_zfill}"
                                        / "set.000"
                                        / "box.npy"
                                    )
                                ).shape[0]
                            )
                del system_it
            except (KeyError, NameError):
                pass
            try:
                for system_it in config_json["subsys_r"]:
                    if (
                            data_apath / f"{system_it}_{it_iteration_zfill}"
                    ).is_dir():
                        # ### #TODO: Here we don't Path because too complex
                        datasets_training.append(
                            f"data/{system_it}_{it_iteration_zfill}/"
                        )
                        datasets_training_json.append(
                            f"{system_it}_{it_iteration_zfill}"
                        )
                        nb_added_nr = (
                            nb_added_nr
                            + np.load(
                                str(
                                    data_apath
                                    / f"{system_it}_{it_iteration_zfill}"
                                    / "set.000"
                                    / "box.npy"
                                )
                            ).shape[0]
                        )
                        if it_iteration == current_iteration:
                            nb_added_nr_iter = (
                                nb_added_nr_iter
                                + np.load(
                                    str(
                                        data_apath
                                        / f"{system_it}_{it_iteration_zfill}"
                                        / "set.000"
                                        / "box.npy"
                                    )
                                ).shape[0]
                            )
                del system_it
            except (KeyError, NameError):
                pass
        del it_iteration, it_iteration_zfill

    # ### Finally the extra sets !
    nb_extra = 0
    if training_json["use_extra_datasets"]:
        config_json["datasets_extra"] = datasets_extra
        del datasets_extra
        for it_datasets_extra in config_json["datasets_extra"]:
            # ### #TODO: Here we don't Path because too complex
            datasets_training.append("data/" + it_datasets_extra + "/")
            datasets_training_json.append(it_datasets_extra)
            nb_extra = (
                nb_extra
                + np.load(
                    str(data_apath / it_datasets_extra / "set.000" / "box.npy")
                ).shape[0]
            )
        del it_datasets_extra
    else:
        del datasets_extra

    # ### Total
    nb_trained = nb_initial + nb_added_nr + nb_added_r + nb_extra

    training_input_json["training"]["training_data"]["systems"] = datasets_training

    training_json["training_data"] = datasets_training_json
    training_json["nb_trained"] = nb_trained
    training_json["nb_initial"] = nb_initial
    training_json["nb_added_nr"] = nb_added_nr
    training_json["nb_added_r"] = nb_added_r
    training_json["nb_added_nr_iter"] = nb_added_nr_iter
    training_json["nb_added_r_iter"] = nb_added_r_iter
    training_json["nb_extra"] = nb_extra

    del datasets_training_json
    del nb_trained, nb_initial, nb_extra
    del nb_added_nr, nb_added_r, nb_added_nr_iter, nb_added_r_iter

    if default_present:
        if (
                default_input_json["training"]["decay_steps"]
                == training_json["decay_steps"]
                and not training_json["decay_steps_fixed"]
        ):
            decay_steps = int(
                get_decay_steps(
                    training_json["nb_trained"], training_json["decay_steps"]
                )
            )
        else:
            decay_steps = training_json["decay_steps"]
    else:
        decay_steps = training_json["decay_steps"]

    numb_steps = training_json["numb_steps"]
    decay_rate_new = get_decay_rate(
        numb_steps,
        training_json["start_lr"],
        training_json["stop_lr"],
        training_json["decay_steps"],
    )
    while decay_rate_new < training_json["decay_rate"]:
        numb_steps = numb_steps + 1e5
        decay_rate_new = get_decay_rate(
            numb_steps,
            training_json["start_lr"],
            training_json["stop_lr"],
            training_json["decay_steps"],
        )
    training_json["numb_steps"] = int(numb_steps)
    training_json["decay_rate"] = decay_rate_new

    del decay_steps, numb_steps, decay_rate_new

    training_input_json["training"]["numb_steps"] = training_json["numb_steps"]
    training_input_json["learning_rate"]["decay_steps"] = training_json["decay_steps"]
    training_input_json["learning_rate"]["stop_lr"] = training_json["stop_lr"]

    # ### Set frozen/compressed bool !
    training_json["is_locked"] = True
    training_json["is_launched"] = False
    training_json["is_checked"] = False
    training_json["is_frozen"] = False
    training_json["is_compressed"] = False

    logging.debug(training_json)
    logging.debug(datasets_training)

    # ### Rsync data to local data
    localdata_apath = Path(".").resolve() / "data"
    localdata_apath.mkdir(exist_ok=True)
    for it_datasets_training in datasets_training:
        subprocess.call(
            [
                "rsync",
                "-a", f"{training_iterative_apath}/{it_datasets_training.rsplit('/', 1)[0]}",
                str(localdata_apath),
            ]
        )
    del it_datasets_training, localdata_apath, datasets_training

    # ### Change some inside output
    training_input_json["training"]["disp_file"] = "lcurve.out"
    training_input_json["training"]["save_ckpt"] = "model.ckpt"

    # ### Create the inputs/jobfiles for each NNP with random SEED inf the form of NNP_number + random(0,
    # 1000) + current_iteration.zfil(3) so between 10000 and unlimited1000999 (at iteration 999 !!)
    if current_iteration > 0:
        previous_iteration = current_iteration - 1
        previous_iteration_zfill = str(previous_iteration).zfill(3)
        prevtraining_json = json_read(
            (control_apath / f"training_{previous_iteration_zfill}.json"),
            # (control_apath / ("training_" + previous_iteration_zfill + ".json")),
            True,
            True,
        )
        walltime_approx_s = int(
            np.ceil(
                (training_json["numb_steps"] * (prevtraining_json["s_per_step"] * 1.50))
            )
        )
        del previous_iteration, previous_iteration_zfill, prevtraining_json
    else:
        s_per_step = read_key_input_json(
            input_json,
            new_input_json,
            "s_per_step",
            default_input_json,
            step_name,
            default_present,
        )
        walltime_approx_s = int(np.ceil((training_json["numb_steps"] * s_per_step)))
        del s_per_step

    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_apath = Path(".").resolve() / str(it_nnp)
        local_apath.mkdir(exist_ok=True)
        check_dir(local_apath, True)

        random.seed()
        random_0_1000 = random.randrange(0, 1000)
        if training_json["deepmd_model_type_descriptor"] == "se_ar":
            training_input_json["model"]["descriptor"]["a"]["seed"] = int(
                str(it_nnp) + str(random_0_1000) + current_iteration_zfill
            )
            training_input_json["model"]["descriptor"]["r"]["seed"] = int(
                str(it_nnp) + str(random_0_1000) + current_iteration_zfill
            )
        else:
            training_input_json["model"]["descriptor"]["seed"] = int(
                str(it_nnp) + str(random_0_1000) + current_iteration_zfill
            )

        training_input_json["model"]["fitting_net"]["seed"] = int(
            str(it_nnp) + str(random_0_1000) + current_iteration_zfill
        )

        training_input_json["training"]["seed"] = int(
            str(it_nnp) + str(random_0_1000) + current_iteration_zfill
        )

        training_input_json_fpath = Path(str(it_nnp) + "/training.json").resolve()
        json_dump(training_input_json, training_input_json_fpath, False)

        slurm_file = copy.deepcopy(slurm_file_master)
        slurm_file = replace_in_list(
            slurm_file, "_R_DEEPMD_VERSION_", str(training_json["deepmd_model_version"])
        )

        slurm_file = replace_in_list(
            slurm_file, "_R_PROJECT_", cluster_spec["project_name"]
        )
        slurm_file = replace_in_list(
            slurm_file, "_R_ALLOC_", cluster_spec["allocation_name"]
        )
        slurm_file = (
            delete_in_list(slurm_file, "_R_PARTITION_")
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
            / f"job_deepmd_train_{cluster_spec['arch_type']}_{cluster}.sh",
            slurm_file,
        )
        del slurm_file, local_apath, training_input_json_fpath, random_0_1000

    del it_nnp, walltime_approx_s, training_input_json

    # ### Dump the dicts
    logging.info(f"-" * 88)
    json_dump(config_json, (control_apath / "config.json"), True)
    json_dump(
        training_json,
        (control_apath / f"training_{current_iteration_zfill}.json"),
        True,
    )
    json_dump_bak(new_input_json, (current_apath / input_fn))

    logging.info(f"-" * 88)
    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
    )

    # ### Cleaning
    del control_apath
    del data_apath
    del input_json, default_input_json, default_present, new_input_json
    del config_json
    del current_iteration, current_iteration_zfill
    del training_json
    del cluster, cluster_spec, cluster_walltime_format, cluster_launch_command
    del slurm_file_master
    del training_iterative_apath, current_apath

    print(globals())
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "preparation",
            Path(sys.argv[1]),
            fake_cluster=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
