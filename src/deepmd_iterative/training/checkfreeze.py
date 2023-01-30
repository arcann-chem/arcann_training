from pathlib import Path
import logging
import sys

# ### Non-standard imports
import numpy as np

# ### deepmd_iterative imports
from deepmd_iterative.common.json import (
    json_read,
    json_dump,
)
from deepmd_iterative.common.files import (
    file_to_strings,
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
    if not training_json["is_checked"]:
        logging.error(f"Lock found. Run/Check first: training check")
        logging.error(f"Aborting...")
        return 1

    check = 0
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_apath = Path(".").resolve()/str(it_nnp)
        if (local_apath/("graph_"+str(it_nnp)+"_"+current_iteration_zfill+".pb")).is_file():
            check = check + 1
        else:
            logging.critical("DP Freeze - "+str(it_nnp)+" not finished/failed")
        del local_apath
    del it_nnp

    if check == config_json["nb_nnp"]:
        training_json["is_frozen"] = True
    else:
        logging.error(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a failure !"
        )
        logging.error("Some DP Freeze did not finished correctly")
        logging.error("Please check manually before relaunching this step")
        logging.error(f"Aborting...")
        return 1
    del check

    json_dump(training_json, (control_apath/f"training_{current_iteration_zfill}.json"), True)

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
            "checkfreeze",
            Path(sys.argv[1]),
            fake_cluster=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
