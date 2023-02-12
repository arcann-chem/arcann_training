from pathlib import Path
import logging
import sys
import subprocess

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    json_read,
    json_dump,
)
from deepmd_iterative.common.files import (
    check_file,
    remove_file,
    remove_file_glob,
    remove_tree,
    check_dir,
)


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

    # ### Get control path and config_json
    control_apath = training_iterative_apath / "control"
    config_json = json_read((control_apath / "config.json"), True, True)
    training_json = json_read(
        (control_apath / f"training_{current_iteration_zfill}.json"), True, True
    )

    # ### Checks
    if not training_json["is_frozen"]:
        logging.error(f"Lock found. Execute first: training check_freeze")
        logging.error(f"Aborting...")
        return 1

    # ### Check if pb files are present and delete temp files
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_apath = Path(".").resolve() / str(it_nnp)
        check_file(local_apath / ("graph_" + str(it_nnp) + "_" + current_iteration_zfill + ".pb"), True, True)
        if training_json["is_compressed"]:
            check_file(local_apath / ("graph_" + str(it_nnp) + "_" + current_iteration_zfill + "_compressed.pb"), True,
                       True)
        remove_file(local_apath / "checkpoint")
        remove_file(local_apath / "input_v2_compat")
        logging.info("Deleting SLURM out/error files...")
        remove_file_glob(local_apath, "DeepMD_*")
        logging.info("Deleting the previous model.ckpt...")
        remove_file_glob(local_apath, "model.ckpt-*")
        if (local_apath / "model-compression").is_dir():
            logging.info("Deleting the temp model-compression folder...")
            remove_tree(local_apath / "model-compression")

    # ### Prepare the test folder
    (training_iterative_apath / (current_iteration_zfill + "-test")).mkdir(exist_ok=True)
    check_dir((training_iterative_apath / (current_iteration_zfill + "-test")), True)

    subprocess.call(["rsync", "-a", str(training_iterative_apath / "data"),
                     str(training_iterative_apath / (current_iteration_zfill + "-test"))])

    # ### Copy the pb files to the NNP meta folder
    (training_iterative_apath / "NNP").mkdir(exist_ok=True)
    check_dir(training_iterative_apath / "NNP", True)

    local_apath = Path(".").resolve()

    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        if training_json["is_compressed"]:
            subprocess.call(["rsync", "-a", str(local_apath / (
                        str(it_nnp) + "/graph_" + str(it_nnp) + "_" + current_iteration_zfill + "_compressed.pb")),
                             str((training_iterative_apath / "NNP"))])
        subprocess.call(["rsync", "-a", str(local_apath / (
                    str(it_nnp) + "/graph_" + str(it_nnp) + "_" + current_iteration_zfill + ".pb")),
                         str((training_iterative_apath / "NNP"))])
    del it_nnp

    # ### Next iteration
    current_iteration = current_iteration + 1
    config_json["current_iteration"] = current_iteration
    current_iteration_zfill = str(current_iteration).zfill(3)

    for it_steps in ["exploration", "reactive", "labeling", "training"]:
        (training_iterative_apath / (current_iteration_zfill + "-" + it_steps)).mkdir(exist_ok=True)
        check_dir(training_iterative_apath / (current_iteration_zfill + "-" + it_steps), True)
    del it_steps

    # ### Delete the temp data folder
    if (local_apath / "data").is_dir():
        logging.info("Deleting the temp data folder...")
        remove_tree(local_apath / "data")
        logging.info("Cleaning done!")
    del local_apath

    # ### Update the config.json
    json_dump(config_json, (control_apath / "config.json"), True)

    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
    )

    # ### Cleaning
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
            "update_iter",
            Path(sys.argv[1]),
            fake_cluster=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
