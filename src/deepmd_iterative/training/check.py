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

    if not training_json["is_launched"]:
        logging.error(f"Lock found. Execute first: training preparation")
        logging.error(f"Aborting...")
        return 1

    # ### Check the normal termination of the training phase
    s_per_step_per_step_size = []
    check = 0
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_apath = Path(".").resolve()/str(it_nnp)
        if (local_apath/"training.out").is_file():
            training_out = file_to_strings((local_apath/"training.out"))
            if any("finished training" in s for s in training_out):
                training_out_time = [s for s in training_out if "training time" in s]
                training_out_time_split = []
                for n in range(0, len(training_out_time)):
                    training_out_time_split.append(training_out_time[n].split(" "))
                    training_out_time_split[n] = " ".join(training_out_time_split[n]).split()
                if (local_apath/f"model.ckpt-{training_out_time_split[-1][3]}.index").is_file():
                    (local_apath/f"model.ckpt-{training_out_time_split[-1][3]}.index").rename(
                        local_apath/"model.ckpt.index")
                    (local_apath/f"model.ckpt-{training_out_time_split[-1][3]}.meta").rename(
                        local_apath/"model.ckpt.meta")
                    (local_apath/f"model.ckpt-{training_out_time_split[-1][3]}.data-00000-of-00001").rename(
                        local_apath/"model.ckpt.data-00000-of-00001")
                for n in range(0, len(training_out_time_split)):
                    s_per_step_per_step_size.append(float(training_out_time_split[n][6]))
                del n
                step_size = float(training_out_time_split[-1][3])-float(training_out_time_split[-2][3])
                check = check + 1
            else:
                logging.critical(f"DP Train - {it_nnp} not finished/failed")
            del training_out, training_out_time, training_out_time_split
        else:
            logging.critical(f"DP Train - {it_nnp} still running/no outfile")
        del local_apath
    del it_nnp

    logging.info(f"-" * 88)
    if check == config_json["nb_nnp"]:
        training_json["is_checked"] = True
    else:
        logging.critical(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a failure !"
        )
        logging.critical(f"Some DP Train did not finished correctly")
        logging.critical(f"Please check manually before relaunching this step")
        logging.critical(f"Aborting...")
        return 1
    del check

    if ("s_per_step_per_step_size" in globals()) and ("step_size" in globals()):
        training_json["s_per_step"] = np.average(s_per_step_per_step_size)/step_size
        del s_per_step_per_step_size, step_size

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
            "check",
            Path(sys.argv[1]),
            fake_cluster=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
