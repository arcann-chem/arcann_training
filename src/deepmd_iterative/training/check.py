from pathlib import Path
import logging
import sys

# Non-standard imports
import numpy as np

# deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.file import (
    file_to_list_of_strings,
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

    # Check if correct folder
    validate_step_folder(step_name)

    # Get iteration
    current_iteration_zfill = Path().resolve().parts[-1].split("-")[0]
    current_iteration = int(current_iteration_zfill)

    # Get control path and config_json
    control_path = training_path / "control"
    config_json = load_json_file((control_path / "config.json"))
    training_json = load_json_file(
        (control_path / f"training_{current_iteration_zfill}.json")
    )

    if not training_json["is_launched"]:
        logging.error(f"Lock found. Execute first: training preparation")
        logging.error(f"Aborting...")
        return 1

    # Check the normal termination of the training phase
    s_per_step_per_step_size = []
    completed_count = 0
    for it_nnp in range(1, config_json["nb_nnp"] + 1):
        local_path = Path(".").resolve() / str(it_nnp)
        if (local_path / "training.out").is_file():
            training_out = file_to_list_of_strings((local_path / "training.out"))
            if any("finished training" in s for s in training_out):
                training_out_time = [s for s in training_out if "training time" in s]
                training_out_time_split = []
                for n in range(0, len(training_out_time)):
                    training_out_time_split.append(training_out_time[n].split(" "))
                    training_out_time_split[n] = " ".join(
                        training_out_time_split[n]
                    ).split()
                if (
                    local_path / f"model.ckpt-{training_out_time_split[-1][3]}.index"
                ).is_file():
                    (
                        local_path
                        / f"model.ckpt-{training_out_time_split[-1][3]}.index"
                    ).rename(local_path / "model.ckpt.index")
                    (
                        local_path / f"model.ckpt-{training_out_time_split[-1][3]}.meta"
                    ).rename(local_path / "model.ckpt.meta")
                    (
                        local_path
                        / f"model.ckpt-{training_out_time_split[-1][3]}.data-00000-of-00001"
                    ).rename(local_path / "model.ckpt.data-00000-of-00001")
                for n in range(0, len(training_out_time_split)):
                    s_per_step_per_step_size.append(
                        float(training_out_time_split[n][6])
                    )
                del n
                step_size = float(training_out_time_split[-1][3]) - float(
                    training_out_time_split[-2][3]
                )
                completed_count += 1
            else:
                logging.critical(f"DP Train - {it_nnp} not finished/failed")
            del training_out, training_out_time, training_out_time_split
        else:
            logging.critical(f"DP Train - {it_nnp} still running/no outfile")
        del local_path
    del it_nnp

    logging.info(f"-" * 88)
    if completed_count == config_json["nb_nnp"]:
        training_json["is_checked"] = True
    else:
        logging.critical(
            f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a failure !"
        )
        logging.critical(f"Some DP Train did not finished correctly")
        logging.critical(f"Please check manually before relaunching this step")
        logging.critical(f"Aborting...")
        return 1
    del completed_count

    if ("s_per_step_per_step_size" in locals()) and ("step_size" in locals()):
        training_json["s_per_step"] = np.average(s_per_step_per_step_size) / step_size
        del s_per_step_per_step_size, step_size

    write_json_file(
        training_json, (control_path / f"training_{current_iteration_zfill}.json")
    )

    logging.info(
        f"Step: {step_name.capitalize()} - Phase: {phase_name.capitalize()} is a succes !"
    )
    # Cleaning
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
            "check",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            input_fn=sys.argv[3],
        )
    else:
        pass
