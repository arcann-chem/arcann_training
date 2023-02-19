from pathlib import Path
import logging
import sys

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
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
    training_json = load_json_file((control_path / f"training_{current_iteration_zfill}.json"))

    # ### Checks
    if not training_json["is_frozen"]:
        logging.error(f"Lock found. Execute first: training check_freeze")
        logging.error(f"Aborting...")
        return 1

    check = 0
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_path = Path(".").resolve() / str(it_nnp)
        if (
            local_path
            / (
                "graph_"
                + str(it_nnp)
                + "_"
                + current_iteration_zfill
                + "_compressed.pb"
            )
        ).is_file():
            check = check + 1
        else:
            logging.critical("DP Compress - " + str(it_nnp) + " not finished/failed")
        del local_path
    del it_nnp

    if check == config_json["nb_nnp"]:
        training_json["is_compressed"] = True
    else:
        logging.error(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a failure !"
        )
        logging.error("Some DP Compress did not finished correctly")
        logging.error("Please check manually before relaunching this step")
        logging.error(f"Aborting...")
        return 1
    del check

    write_json_file(
        training_json,
        (control_path / f"training_{current_iteration_zfill}.json")
    )

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
            "check_compress",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
