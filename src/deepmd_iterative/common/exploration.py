from pathlib import Path
import logging
import sys
from typing import List, Tuple
import subprocess


def generate_starting_points(
    exploration_type: int,
    it_subsys_nr: int,
    training_path: str,
    previous_iteration_zfill: str,
    prevexploration_json: dict,
    input_present: bool,
    subsys_disturbed_start: bool,
) -> Tuple[List[str], List[str], bool]:
    """
    Generates a list of starting point file names, given the exploration type, subsystem index, and iteration number.

    Args:
        exploration_type (int): An integer representing the exploration type (1 for "lammps" or 2 for "i-PI").
        it_subsys_nr (int): An integer representing the subsystem index.
        training_path (str): The path to the training directory.
        previous_iteration_zfill (str): A zero-padded string representing the previous iteration number.
        prevexploration_json (dict): A dictionary containing information about the previous exploration.
        input_present (bool): A boolean indicating whether an input file is present.
        subsys_disturbed_start (bool): A boolean indicating whether to start from a disturbed minimum.

    Returns:
        A tuple containing the starting point file names, the backup starting point file names, and a boolean indicating whether to start from a disturbed minimum.
    """

    # Determine file extension based on exploration type
    if exploration_type == 1:
        file_extension = "lmp"
    elif exploration_type == 2:
        file_extension = "xyz"

    # Get list of starting point file names for subsystem and iteration
    starting_points_path = list(
        Path(training_path, "starting_structures").glob(
            f"{previous_iteration_zfill}_{it_subsys_nr}_*.{file_extension}"
        )
    )
    starting_points_all = [str(zzz).split("/")[-1] for zzz in starting_points_path]
    starting_points = [zzz for zzz in starting_points_all if "disturbed" not in zzz]
    starting_points_disturbed = [
        zzz for zzz in starting_points_all if zzz not in starting_points
    ]
    starting_points_bckp = starting_points.copy()
    starting_points_disturbed_bckp = starting_points_disturbed.copy()

    # Check if subsystem should start from a disturbed minimum
    if input_present and subsys_disturbed_start:
        # If input file is present and disturbed start is requested, use disturbed starting points
        starting_points = starting_points_disturbed.copy()
        starting_points_bckp = starting_points_disturbed_bckp.copy()
        return starting_points, starting_points_bckp, True
    elif not input_present:
        # If input file is not present, check if subsystem started from disturbed minimum in previous iteration
        if (
            prevexploration_json["subsys_nr"][it_subsys_nr]["disturbed_start"]
            and prevexploration_json["subsys_nr"][it_subsys_nr]["disturbed_min"]
        ):
            # If subsystem started from disturbed minimum in previous iteration, use disturbed starting points
            starting_points = starting_points_disturbed.copy()
            starting_points_bckp = starting_points_disturbed_bckp.copy()
            return starting_points, starting_points_bckp, True
        else:
            # Otherwise, use regular starting points
            return starting_points, starting_points_bckp, False
    else:
        # If input file is present but disturbed start is not requested, use regular starting points
        return starting_points, starting_points_bckp, False


def create_models_list(
    config_json: dict,
    prevtraining_json: dict,
    it_nnp: int,
    previous_iteration_zfill: str,
    training_path: Path,
    local_path: Path,
) -> Tuple[List[str], str]:
    """
    Generate a list of model file names and create symbolic links to the corresponding model files.

    Args:
        config_json (dict): A dictionary object containing configuration parameters.
        prevtraining_json (dict): A dictionary object containing training data from previous iterations.
        it_nnp (int): An integer representing the index of the NNP model to start from.
        previous_iteration_zfill (str): A string representing the zero-padded iteration number of the previous training iteration.
        training_path (Path): The path to the training directory.
        local_path (Path): The path to the local directory.

    Returns:
        A tuple containing the list of model file names, and a string of space-separated model file names.
    """

    # Generate list of NNP model indices and reorder based on current model to propagate
    list_nnp = [zzz for zzz in range(1, config_json["nb_nnp"] + 1)]
    reorder_nnp_list = (
        list_nnp[list_nnp.index(it_nnp) :] + list_nnp[: list_nnp.index(it_nnp)]
    )

    # Determine whether to use compressed models
    compress_str = "_compressed" if prevtraining_json["is_compressed"] else ""

    # Generate list of model file names
    models_list = [
        "graph_" + str(f) + "_" + previous_iteration_zfill + compress_str + ".pb"
        for f in reorder_nnp_list
    ]

    # Create symbolic links to the model files in the local directory
    for it_sub_nnp in range(1, config_json["nb_nnp"] + 1):
        nnp_apath = (
            training_path
            / "NNP"
            / (
                "graph_"
                + str(it_sub_nnp)
                + "_"
                + previous_iteration_zfill
                + compress_str
                + ".pb"
            )
        ).resolve()
        subprocess.call(["ln", "-nsf", str(nnp_apath), str(local_path)])

    # Join the model file names into a single string for ease of use
    models_string = " ".join(models_list)

    return models_list, models_string


def update_nb_steps_factor(prevexploration_json: dict, it_subsys_nr: int) -> int:
    """
    Calculates a ratio based on information from a dictionary and returns a multiplying factor for subsys_nb_steps.

    Args:
        prevexploration_json (dict): A dictionary containing information about a previous exploration.
        it_subsys_nr (int): An integer representing the subsystem index.

    Returns:
        An integer representing the multiplying factor for subsys_nb_steps.
    """
    # Calculate the ratio of ill-described candidates to the total number of candidates
    ratio_ill_described = (
        prevexploration_json["subsys_nr"][it_subsys_nr]["nb_candidates"]
        + prevexploration_json["subsys_nr"][it_subsys_nr]["nb_rejected"]
    ) / prevexploration_json["subsys_nr"][it_subsys_nr]["nb_total"]

    # Return a multiplying factor for subsys_nb_steps based on the ratio of ill-described candidates
    if ratio_ill_described < 0.10:
        return 4
    elif ratio_ill_described < 0.20:
        return 2
    else:
        return 1
