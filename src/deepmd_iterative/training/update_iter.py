from pathlib import Path
import logging
import sys
import subprocess

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.file import (
    check_file_existence,
    remove_file,
    remove_files_matching_glob,
    remove_tree,
    check_directory,
)
from deepmd_iterative.common.check import validate_step_folder


def main(
    step_name,
    phase_name,
    deepmd_iterative_path,
    fake_machine=None,
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

    # ### Get control path and config_json
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))
    training_json = load_json_file(
        (control_path / f"training_{current_iteration_zfill}.json")
    )

    # ### Checks
    if not training_json["is_frozen"]:
        logging.error(f"Lock found. Execute first: training check_freeze")
        logging.error(f"Aborting...")
        return 1

    # ### Check if pb files are present and delete temp files
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_path = Path(".").resolve() / str(it_nnp)
        check_file_existence(
            local_path
            / ("graph_" + str(it_nnp) + "_" + current_iteration_zfill + ".pb")
        )
        if training_json["is_compressed"]:
            check_file_existence(
                local_path
                / (
                    "graph_"
                    + str(it_nnp)
                    + "_"
                    + current_iteration_zfill
                    + "_compressed.pb"
                )
            )
        remove_file(local_path / "checkpoint")
        remove_file(local_path / "input_v2_compat")
        logging.info("Deleting SLURM out/error files...")
        remove_files_matching_glob(local_path, "DeepMD_*")
        logging.info("Deleting the previous model.ckpt...")
        remove_files_matching_glob(local_path, "model.ckpt-*")
        if (local_path / "model-compression").is_dir():
            logging.info("Deleting the temp model-compression folder...")
            remove_tree(local_path / "model-compression")

    # ### Prepare the test folder
    (training_path / (current_iteration_zfill + "-test")).mkdir(exist_ok=True)
    check_directory((training_path / (current_iteration_zfill + "-test")))

    subprocess.run(
        [
            "rsync",
            "-a",
            str(training_path / "data"),
            str(training_path / (current_iteration_zfill + "-test")),
        ]
    )

    # ### Copy the pb files to the NNP meta folder
    (training_path / "NNP").mkdir(exist_ok=True)
    check_directory(training_path / "NNP")

    local_path = Path(".").resolve()

    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        if training_json["is_compressed"]:
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    str(
                        local_path
                        / (
                            str(it_nnp)
                            + "/graph_"
                            + str(it_nnp)
                            + "_"
                            + current_iteration_zfill
                            + "_compressed.pb"
                        )
                    ),
                    str((training_path / "NNP")),
                ]
            )
        subprocess.run(
            [
                "rsync",
                "-a",
                str(
                    local_path
                    / (
                        str(it_nnp)
                        + "/graph_"
                        + str(it_nnp)
                        + "_"
                        + current_iteration_zfill
                        + ".pb"
                    )
                ),
                str((training_path / "NNP")),
            ]
        )
    del it_nnp

    # ### Next iteration
    current_iteration = current_iteration + 1
    config_json["current_iteration"] = current_iteration
    current_iteration_zfill = str(current_iteration).zfill(3)

    for it_steps in ["exploration", "reactive", "labeling", "training"]:
        (training_path / (current_iteration_zfill + "-" + it_steps)).mkdir(
            exist_ok=True
        )
        check_directory(training_path / (current_iteration_zfill + "-" + it_steps))
    del it_steps

    # ### Delete the temp data folder
    if (local_path / "data").is_dir():
        logging.info("Deleting the temp data folder...")
        remove_tree(local_path / "data")
        logging.info("Cleaning done!")
    del local_path

    # ### Update the config.json
    write_json_file(config_json, (control_path / "config.json"))

    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
    )

    # ### Cleaning
    del control_path
    del config_json
    del current_iteration, current_iteration_zfill
    del training_json
    del training_path, current_path

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "update_iter",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
