"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/09/13
"""
# Standard library modules
import importlib
import logging
import re
import sys
from pathlib import Path

# Non-standard library imports
import numpy as np

# Local imports
from deepmd_iterative.common.json import (
    load_json_file,
    write_json_file,
)
from deepmd_iterative.common.list import (
    textfile_to_string_list,
    string_list_to_textfile,
)
from deepmd_iterative.common.filesystem import (
    check_file_existence,
    remove_file,
    remove_files_matching_glob,
)
from deepmd_iterative.common.check import validate_step_folder

# Import constants
try:
    importlib.import_module("scipy")
    from scipy import constants

    Ha_to_eV = constants.physical_constants["atomic unit of electric potential"][0]
    Bohr_to_A = constants.physical_constants["Bohr radius"][0] / constants.angstrom
    au_to_eV_per_A = np.float64(Ha_to_eV / Bohr_to_A)
    eV_per_A3_to_GPa = np.float64(
        constants.eV / constants.angstrom**3 / constants.giga
    )
except ImportError:
    import numpy as np

    Ha_to_eV = np.float64(27.211386245988)
    Bohr_to_A = np.float64(0.529177210903)
    au_to_eV_per_A = np.float64(Ha_to_eV / Bohr_to_A)
    eV_per_A3_to_GPa = np.float64(160.21766208)
except Exception:
    import numpy as np

    Ha_to_eV = np.float64(27.211386245988)
    Bohr_to_A = np.float64(0.529177210903)
    au_to_eV_per_A = np.float64(Ha_to_eV / Bohr_to_A)
    eV_per_A3_to_GPa = np.float64(160.21766208)


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_input_json_filename: str = "input.json",
):
    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}."
    )
    logging.debug(f"Current path :{current_path}")
    logging.debug(f"Training path: {training_path}")
    logging.debug(f"Program path: {deepmd_iterative_path}")
    logging.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Get control path, load the main JSON and the exploration JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    labeling_json = load_json_file((control_path / f"labeling_{padded_curr_iter}.json"))

    # Check if we can continue
    if not labeling_json["is_checked"]:
        logging.error(f"Lock found. Execute first: labeling launch.")
        logging.error(f"Aborting...")
        return 1

    # Create if it doesn't exists the data path.
    (training_path / "data").mkdir(exist_ok=True)

    candidates_expected_count = 0
    for system_auto_index, system_auto in enumerate(labeling_json["systems_auto"]):
        logging.info(
            f"Processing system: {system_auto} ({system_auto_index + 1}/{len(main_json['systems_auto'])})"
        )

        system_candidates_count = labeling_json["systems_auto"][system_auto][
            "candidates_count"
        ]
        system_candidates_skipped_count = labeling_json["systems_auto"][system_auto][
            "candidates_skipped_count"
        ]
        system_path = current_path / system_auto

        data_path = training_path / "data" / (system_auto + "_" + padded_curr_iter)
        data_path.mkdir(exist_ok=True)
        (data_path / "set.000").mkdir(exist_ok=True)

        energy_array_raw = np.zeros(
            (system_candidates_count - system_candidates_skipped_count)
        )
        coord_array_raw = np.zeros(
            (
                system_candidates_count - system_candidates_skipped_count,
                main_json["systems_auto"][system_auto]["nb_atm"] * 3,
            )
        )
        box_array_raw = np.zeros(
            (system_candidates_count - system_candidates_skipped_count, 9)
        )
        volume_array_raw = np.zeros(
            (system_candidates_count - system_candidates_skipped_count)
        )
        force_array_raw = np.zeros(
            (
                system_candidates_count - system_candidates_skipped_count,
                main_json["systems_auto"][system_auto]["nb_atm"] * 3,
            )
        )
        virial_array_raw = np.zeros(
            (system_candidates_count - system_candidates_skipped_count, 9)
        )

        # Options
        is_virial = False
        is_wannier = False

        # Wannier
        wannier_not_converged = ["#Indexes start at 0\n"]

        # counter for non skipped configurations
        system_candidates_not_skipped_counter = 0

        for labeling_step in range(system_candidates_count):
            padded_labeling_step = str(labeling_step).zfill(5)
            labeling_step_path = system_path / padded_labeling_step

            if not (labeling_step_path / "skip").is_file():
                system_candidates_not_skipped_counter += 1

                # With the first, we create a type.raw and get the CP2K version
                if system_candidates_not_skipped_counter == 1:
                    check_file_existence(
                        training_path / "user_files" / f"{system_auto}.lmp",
                        True,
                        True,
                        "Input data file (lmp) not present.",
                    )

                    lammps_data = textfile_to_string_list(
                        training_path / "user_files" / f"{system_auto}.lmp"
                    )
                    indexes = [idx for idx, s in enumerate(lammps_data) if "Atoms" in s]
                    if len(indexes) > 1:
                        for index in [
                            idx for idx, s in enumerate(lammps_data) if "Atoms" in s
                        ]:
                            atom_list = [
                                line.strip().split()
                                for line in lammps_data[index + 2 : index + 4]
                            ]
                            if (
                                len(atom_list[0]) == len(atom_list[1])
                                and lammps_data[index + 1] == " \n"
                                and atom_list[0][0] == "1"
                                and atom_list[1][0] == "2"
                            ):
                                idx = index
                                break
                    else:
                        idx = indexes[0]
                    del lammps_data[0 : idx + 2]
                    lammps_data = lammps_data[
                        0 : main_json["systems_auto"][system_auto]["nb_atm"] + 1
                    ]
                    lammps_data = [
                        " ".join(f.replace("\n", "").split()) for f in lammps_data
                    ]
                    lammps_data = [g.split(" ")[1:2] for g in lammps_data]
                    type_atom_array = np.asarray(lammps_data, dtype=np.int64).flatten()
                    type_atom_array = type_atom_array - 1
                    np.savetxt(
                        f"{system_path}/type.raw",
                        type_atom_array,
                        delimiter=" ",
                        newline=" ",
                        fmt="%d",
                    )
                    np.savetxt(
                        f"{data_path}/type.raw",
                        type_atom_array,
                        delimiter=" ",
                        newline=" ",
                        fmt="%d",
                    )

                    # Get the CP2K version
                    output_cp2k = textfile_to_string_list(
                        labeling_step_path / f"2_labeling_{padded_labeling_step}.out"
                    )
                    output_cp2k = [
                        _ for _ in output_cp2k if "CP2K| version string:" in _
                    ]
                    output_cp2k = [
                        " ".join(_.replace("\n", "").split()) for _ in output_cp2k
                    ]
                    output_cp2k = [_.split(" ")[-1] for _ in output_cp2k]
                    cp2k_version = float(output_cp2k[0])

                # TODO Patterns should be in parser
                # Energy
                energy_cp2k = textfile_to_string_list(
                    labeling_step_path
                    / f"2_labeling_{padded_labeling_step}-Force_Eval.fe"
                )
                pattern = r"ENERGY\|.*?([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)"
                for line in energy_cp2k:
                    match = re.search(pattern, line)
                    if match:
                        energy_float = float(match.group(1))
                del line, match, pattern
                energy_array = np.asarray(energy_float, dtype=np.float64).flatten()
                energy_array_raw[system_candidates_not_skipped_counter - 1] = (
                    energy_array[0] * Ha_to_eV
                )
                del energy_cp2k, energy_array, energy_float

                # Coordinates
                coordinate_xyz = textfile_to_string_list(
                    labeling_step_path / f"labeling_{padded_labeling_step}.xyz"
                )
                del coordinate_xyz[0:2]
                coordinate_xyz = [
                    " ".join(_.replace("\n", "").split()) for _ in coordinate_xyz
                ]
                coordinate_xyz = [_.split(" ")[1:] for _ in coordinate_xyz]
                coord_array = np.asarray(coordinate_xyz, dtype=np.float64).flatten()
                coord_array_raw[
                    system_candidates_not_skipped_counter - 1, :
                ] = coord_array
                del coord_array, coordinate_xyz

                # Box / Volume
                input_cp2k = textfile_to_string_list(
                    labeling_step_path / f"1_labeling_{padded_labeling_step}.inp"
                )
                cell = [
                    float(_)
                    for _ in re.findall(
                        r"\d+\.\d+", [_ for _ in input_cp2k if "ABC" in _][0]
                    )
                ]
                box_array_raw[system_candidates_not_skipped_counter - 1, 0] = cell[0]
                box_array_raw[system_candidates_not_skipped_counter - 1, 4] = cell[1]
                box_array_raw[system_candidates_not_skipped_counter - 1, 8] = cell[2]
                volume_array_raw[system_candidates_not_skipped_counter - 1] = (
                    cell[0] * cell[1] * cell[2]
                )
                del input_cp2k, cell

                # Forces
                force_cp2k = textfile_to_string_list(
                    labeling_step_path / f"2_labeling_{padded_labeling_step}-Forces.for"
                )
                del force_cp2k[0:4]
                del force_cp2k[-1]
                force_cp2k = [" ".join(_.replace("\n", "").split()) for _ in force_cp2k]
                force_cp2k = [_.split(" ")[3:] for _ in force_cp2k]
                force_array = np.asarray(force_cp2k, dtype=np.float64).flatten()
                force_array_raw[system_candidates_not_skipped_counter - 1, :] = (
                    force_array * au_to_eV_per_A
                )
                del force_array, force_cp2k

                # Virial
                if (
                    labeling_step_path
                    / f"2_labeling_{padded_labeling_step}-Stress_Tensor.st"
                ).is_file():
                    stress_cp2k = textfile_to_string_list(
                        labeling_step_path
                        / f"2_labeling_{padded_labeling_step}-Stress_Tensor.st"
                    )

                    if cp2k_version < 8 and cp2k_version >= 6:
                        matching_index = None
                        for index, line in enumerate(stress_cp2k):
                            if re.search(r"\bX\b.*\bY\b.*\bZ\b", line):
                                matching_index = index
                                break
                        del index, line
                        if matching_index is not None:
                            # Extract tensor values for X, Y, Z directions
                            x_values = re.findall(
                                r"X\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)",
                                stress_cp2k[matching_index + 1],
                            )
                            y_values = re.findall(
                                r"Y\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)",
                                stress_cp2k[matching_index + 2],
                            )
                            z_values = re.findall(
                                r"Z\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)",
                                stress_cp2k[matching_index + 3],
                            )

                            # Combine tensor values into a single array
                            tensor_values = np.vstack(
                                (x_values, y_values, z_values)
                            ).astype(np.float64)

                            # Flatten the array
                            stress_xyz_array = tensor_values.flatten()
                            virial_array_raw[
                                system_candidates_not_skipped_counter - 1, :
                            ] = (
                                stress_xyz_array
                                * volume_array_raw[
                                    system_candidates_not_skipped_counter - 1
                                ]
                                / eV_per_A3_to_GPa
                            )
                            is_virial = True
                            del (
                                x_values,
                                y_values,
                                z_values,
                                tensor_values,
                                stress_xyz_array,
                            )
                        else:
                            is_virial = False
                        del matching_index
                        del stress_cp2k
                    elif cp2k_version < 2024 and cp2k_version >= 8:
                        matching_index = None
                        for index, line in enumerate(stress_cp2k):
                            if re.search(r"\bx\b.*\by\b.*\bz\b", line):
                                matching_index = index
                                break
                        del index, line
                        if matching_index is not None:
                            # Extract tensor values for X, Y, Z directions
                            x_values = re.findall(
                                r"x\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)",
                                stress_cp2k[matching_index + 1],
                            )
                            y_values = re.findall(
                                r"y\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)",
                                stress_cp2k[matching_index + 2],
                            )
                            z_values = re.findall(
                                r"z\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)",
                                stress_cp2k[matching_index + 3],
                            )

                            # Combine tensor values into a single array
                            tensor_values = np.vstack(
                                (x_values, y_values, z_values)
                            ).astype(np.float64)

                            # Flatten the array
                            stress_xyz_array = tensor_values.flatten()
                            virial_array_raw[
                                system_candidates_not_skipped_counter - 1, :
                            ] = (
                                stress_xyz_array
                                * volume_array_raw[
                                    system_candidates_not_skipped_counter - 1
                                ]
                                / eV_per_A3_to_GPa
                            )
                            is_virial = True
                            del (
                                x_values,
                                y_values,
                                z_values,
                                tensor_values,
                                stress_xyz_array,
                            )
                        else:
                            is_virial = False
                        del matching_index
                        del stress_cp2k
                    else:
                        logging.info(
                            f"This version of CP2K is not supported for tensor: {cp2k_version}"
                        )

                # Wannier
                if (
                    labeling_step_path
                    / f"2_labeling_{padded_labeling_step}-Wannier.xyz"
                ).is_file():
                    stress_cp2k = textfile_to_string_list(
                        labeling_step_path
                        / f"2_labeling_{padded_labeling_step}-Wannier.xyz"
                    )

                    del wannier_xyz[
                        0 : 2 + main_json["system_auto"][system_auto]["nb_atm"]
                    ]
                    wannier_xyz = [
                        " ".join(_.replace("\n", "").split()) for _ in wannier_xyz
                    ]
                    wannier_xyz = [_.split(" ")[1:] for _ in wannier_xyz]
                    wannier_array = np.asarray(wannier_xyz, dtype=np.float64).flatten()
                    if system_candidates_not_skipped_counter == 1:
                        wannier_array_raw = np.zeros(
                            (
                                system_candidates_count
                                - system_candidates_skipped_count,
                                len(wannier_xyz) * 3,
                            )
                        )

                    wannier_array_raw[
                        system_candidates_not_skipped_counter - 1, :
                    ] = wannier_array
                    is_wannier = True
                    del wannier_array, wannier_xyz

                    # Check if wannier centers are not converged
                    output_cp2k = textfile_to_string_list(
                        labeling_step_path / f"2_labeling_{padded_labeling_step}.out"
                    )
                    if any(
                        "LOCALIZATION! loop did not converge within the maximum number of iterations"
                        in _
                        for _ in output_cp2k
                    ):
                        wannier_not_converged.append(
                            f"{system_candidates_not_skipped_counter - 1}\n"
                        )
        del padded_labeling_step, labeling_step, labeling_step_path

        np.savetxt(system_path / "energy.raw", energy_array_raw, delimiter=" ")
        np.save(data_path / "set.000" / "energy", energy_array_raw)
        del energy_array_raw

        np.savetxt(system_path / "coord.raw", coord_array_raw, delimiter=" ")
        np.save(data_path / "set.000" / "coord", coord_array_raw)
        del coord_array_raw

        np.savetxt(system_path / "box.raw", box_array_raw, delimiter=" ")
        np.save(data_path / "set.000" / "box", box_array_raw)
        del box_array_raw, volume_array_raw

        np.savetxt(system_path / "force.raw", force_array_raw, delimiter=" ")
        np.save(data_path / "set.000" / "force", force_array_raw)
        del force_array_raw

        if is_virial:
            np.savetxt(system_path / "virial.raw", virial_array_raw, delimiter=" ")
            np.save(data_path / "set.000" / "virial", virial_array_raw)
        del virial_array_raw, is_virial

        if is_wannier:
            np.savetxt(system_path / "wannier.raw", wannier_array_raw, delimiter=" ")
            np.save(data_path / "set.000" / "wannier", wannier_array_raw)
            if len(wannier_not_converged) > 1:
                string_list_to_textfile(
                    data_path / "set.000" / "wannier_not-converged.txt",
                    wannier_not_converged,
                )
            del wannier_not_converged, is_wannier

        system_disturbed_candidates_count = labeling_json["systems_auto"][system_auto][
            "disturbed_candidates_count"
        ]
        system_disturbed_candidates_skipped_count = labeling_json["systems_auto"][
            system_auto
        ]["disturbed_candidates_skipped_count"]

        if (
            system_disturbed_candidates_count
            - system_disturbed_candidates_skipped_count
            > 0
        ):
            data_path = (
                training_path
                / "data"
                / (system_auto + "-disturbed_" + padded_curr_iter)
            )
            data_path.mkdir(exist_ok=True)
            (data_path / "set.000").mkdir(exist_ok=True)

            energy_array_raw = np.zeros(
                (
                    system_disturbed_candidates_count
                    - system_disturbed_candidates_skipped_count
                )
            )
            coord_array_raw = np.zeros(
                (
                    system_disturbed_candidates_count
                    - system_disturbed_candidates_skipped_count,
                    main_json["systems_auto"][system_auto]["nb_atm"] * 3,
                )
            )
            box_array_raw = np.zeros(
                (
                    system_disturbed_candidates_count
                    - system_disturbed_candidates_skipped_count,
                    9,
                )
            )
            volume_array_raw = np.zeros(
                (
                    system_disturbed_candidates_count
                    - system_disturbed_candidates_skipped_count
                )
            )
            force_array_raw = np.zeros(
                (
                    system_disturbed_candidates_count
                    - system_disturbed_candidates_skipped_count,
                    main_json["systems_auto"][system_auto]["nb_atm"] * 3,
                )
            )
            virial_array_raw = np.zeros(
                (
                    system_disturbed_candidates_count
                    - system_disturbed_candidates_skipped_count,
                    9,
                )
            )

            # Options
            is_virial = False
            is_wannier = False

            # Wannier
            wannier_not_converged = ["#Indexes start at 0\n"]

            # counter for non skipped configurations
            system_disturbed_candidates_not_skipped_counter = 0

            for labeling_step in range(
                system_candidates_count,
                system_candidates_count + system_disturbed_candidates_count,
            ):
                padded_labeling_step = str(labeling_step).zfill(5)
                labeling_step_path = system_path / padded_labeling_step

                if not (labeling_step_path / "skip").is_file():
                    system_disturbed_candidates_not_skipped_counter += 1

                    # With the first, we create a type.raw and get the CP2K version
                    if system_disturbed_candidates_not_skipped_counter == 1:
                        check_file_existence(
                            training_path / "user_files" / f"{system_auto}.lmp",
                            True,
                            True,
                            "Input data file (lmp) not present.",
                        )

                        lammps_data = textfile_to_string_list(
                            training_path / "user_files" / f"{system_auto}.lmp"
                        )
                        indexes = [
                            idx for idx, s in enumerate(lammps_data) if "Atoms" in s
                        ]
                        if len(indexes) > 1:
                            for index in [
                                idx for idx, s in enumerate(lammps_data) if "Atoms" in s
                            ]:
                                atom_list = [
                                    line.strip().split()
                                    for line in lammps_data[index + 2 : index + 4]
                                ]
                                if (
                                    len(atom_list[0]) == len(atom_list[1])
                                    and lammps_data[index + 1] == " \n"
                                    and atom_list[0][0] == "1"
                                    and atom_list[1][0] == "2"
                                ):
                                    idx = index
                                    break
                        else:
                            idx = indexes[0]
                        del lammps_data[0 : idx + 2]
                        lammps_data = lammps_data[
                            0 : main_json["systems_auto"][system_auto]["nb_atm"] + 1
                        ]
                        lammps_data = [
                            " ".join(f.replace("\n", "").split()) for f in lammps_data
                        ]
                        lammps_data = [g.split(" ")[1:2] for g in lammps_data]
                        type_atom_array = np.asarray(
                            lammps_data, dtype=np.int64
                        ).flatten()
                        type_atom_array = type_atom_array - 1
                        np.savetxt(
                            f"{system_path}/type.raw",
                            type_atom_array,
                            delimiter=" ",
                            newline=" ",
                            fmt="%d",
                        )
                        np.savetxt(
                            f"{data_path}/type.raw",
                            type_atom_array,
                            delimiter=" ",
                            newline=" ",
                            fmt="%d",
                        )

                        # Get the CP2K version
                        output_cp2k = textfile_to_string_list(
                            labeling_step_path
                            / f"2_labeling_{padded_labeling_step}.out"
                        )
                        output_cp2k = [
                            _ for _ in output_cp2k if "CP2K| version string:" in _
                        ]
                        output_cp2k = [
                            " ".join(_.replace("\n", "").split()) for _ in output_cp2k
                        ]
                        output_cp2k = [_.split(" ")[-1] for _ in output_cp2k]
                        cp2k_version = float(output_cp2k[0])

                    # TODO Patterns should be in parser
                    # Energy
                    energy_cp2k = textfile_to_string_list(
                        labeling_step_path
                        / f"2_labeling_{padded_labeling_step}-Force_Eval.fe"
                    )
                    pattern = r"ENERGY\|.*?([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)"
                    for line in energy_cp2k:
                        match = re.search(pattern, line)
                        if match:
                            energy_float = float(match.group(1))
                    del line, match, pattern
                    energy_array = np.asarray(energy_float, dtype=np.float64).flatten()
                    energy_array_raw[
                        system_disturbed_candidates_not_skipped_counter - 1
                    ] = (energy_array[0] * Ha_to_eV)
                    del energy_cp2k, energy_array, energy_float

                    # Coordinates
                    coordinate_xyz = textfile_to_string_list(
                        labeling_step_path / f"labeling_{padded_labeling_step}.xyz"
                    )
                    del coordinate_xyz[0:2]
                    coordinate_xyz = [
                        " ".join(_.replace("\n", "").split()) for _ in coordinate_xyz
                    ]
                    coordinate_xyz = [_.split(" ")[1:] for _ in coordinate_xyz]
                    coord_array = np.asarray(coordinate_xyz, dtype=np.float64).flatten()
                    coord_array_raw[
                        system_disturbed_candidates_not_skipped_counter - 1, :
                    ] = coord_array
                    del coord_array, coordinate_xyz

                    # Box / Volume
                    input_cp2k = textfile_to_string_list(
                        labeling_step_path / f"1_labeling_{padded_labeling_step}.inp"
                    )
                    cell = [
                        float(_)
                        for _ in re.findall(
                            r"\d+\.\d+", [_ for _ in input_cp2k if "ABC" in _][0]
                        )
                    ]
                    box_array_raw[
                        system_disturbed_candidates_not_skipped_counter - 1, 0
                    ] = cell[0]
                    box_array_raw[
                        system_disturbed_candidates_not_skipped_counter - 1, 4
                    ] = cell[1]
                    box_array_raw[
                        system_disturbed_candidates_not_skipped_counter - 1, 8
                    ] = cell[2]
                    volume_array_raw[
                        system_disturbed_candidates_not_skipped_counter - 1
                    ] = (cell[0] * cell[1] * cell[2])
                    del input_cp2k, cell

                    # Forces
                    force_cp2k = textfile_to_string_list(
                        labeling_step_path
                        / f"2_labeling_{padded_labeling_step}-Forces.for"
                    )
                    del force_cp2k[0:4]
                    del force_cp2k[-1]
                    force_cp2k = [
                        " ".join(_.replace("\n", "").split()) for _ in force_cp2k
                    ]
                    force_cp2k = [_.split(" ")[3:] for _ in force_cp2k]
                    force_array = np.asarray(force_cp2k, dtype=np.float64).flatten()
                    force_array_raw[
                        system_disturbed_candidates_not_skipped_counter - 1, :
                    ] = (force_array * au_to_eV_per_A)
                    del force_array, force_cp2k

                    # Virial
                    if (
                        labeling_step_path
                        / f"2_labeling_{padded_labeling_step}-Stress_Tensor.st"
                    ).is_file():
                        stress_cp2k = textfile_to_string_list(
                            labeling_step_path
                            / f"2_labeling_{padded_labeling_step}-Stress_Tensor.st"
                        )

                        if cp2k_version < 8 and cp2k_version >= 6:
                            matching_index = None
                            for index, line in enumerate(stress_cp2k):
                                if re.search(r"\bX\b.*\bY\b.*\bZ\b", line):
                                    matching_index = index
                                    break
                            del index, line
                            if matching_index is not None:
                                # Extract tensor values for X, Y, Z directions
                                x_values = re.findall(
                                    r"X\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)",
                                    stress_cp2k[matching_index + 1],
                                )
                                y_values = re.findall(
                                    r"Y\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)",
                                    stress_cp2k[matching_index + 2],
                                )
                                z_values = re.findall(
                                    r"Z\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)",
                                    stress_cp2k[matching_index + 3],
                                )

                                # Combine tensor values into a single array
                                tensor_values = np.vstack(
                                    (x_values, y_values, z_values)
                                ).astype(np.float64)

                                # Flatten the array
                                stress_xyz_array = tensor_values.flatten()
                                virial_array_raw[
                                    system_disturbed_candidates_not_skipped_counter - 1,
                                    :,
                                ] = (
                                    stress_xyz_array
                                    * volume_array_raw[
                                        system_disturbed_candidates_not_skipped_counter
                                        - 1
                                    ]
                                    / eV_per_A3_to_GPa
                                )
                                is_virial = True
                                del (
                                    x_values,
                                    y_values,
                                    z_values,
                                    tensor_values,
                                    stress_xyz_array,
                                )
                            else:
                                is_virial = False
                            del matching_index
                            del stress_cp2k
                        elif cp2k_version < 2024 and cp2k_version >= 8:
                            matching_index = None
                            for index, line in enumerate(stress_cp2k):
                                if re.search(r"\bx\b.*\by\b.*\bz\b", line):
                                    matching_index = index
                                    break
                            del index, line
                            if matching_index is not None:
                                # Extract tensor values for X, Y, Z directions
                                x_values = re.findall(
                                    r"x\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)",
                                    stress_cp2k[matching_index + 1],
                                )
                                y_values = re.findall(
                                    r"y\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)",
                                    stress_cp2k[matching_index + 2],
                                )
                                z_values = re.findall(
                                    r"z\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)",
                                    stress_cp2k[matching_index + 3],
                                )

                                # Combine tensor values into a single array
                                tensor_values = np.vstack(
                                    (x_values, y_values, z_values)
                                ).astype(np.float64)

                                # Flatten the array
                                stress_xyz_array = tensor_values.flatten()
                                virial_array_raw[
                                    system_disturbed_candidates_not_skipped_counter - 1,
                                    :,
                                ] = (
                                    stress_xyz_array
                                    * volume_array_raw[
                                        system_disturbed_candidates_not_skipped_counter
                                        - 1
                                    ]
                                    / eV_per_A3_to_GPa
                                )
                                is_virial = True
                                del (
                                    x_values,
                                    y_values,
                                    z_values,
                                    tensor_values,
                                    stress_xyz_array,
                                )
                            else:
                                is_virial = False
                            del matching_index
                            del stress_cp2k
                        else:
                            logging.info(
                                f"This version of CP2K is not supported for tensor: {cp2k_version}"
                            )

                    # Wannier
                    if (
                        labeling_step_path
                        / f"2_labeling_{padded_labeling_step}-Wannier.xyz"
                    ).is_file():
                        stress_cp2k = textfile_to_string_list(
                            labeling_step_path
                            / f"2_labeling_{padded_labeling_step}-Wannier.xyz"
                        )

                        del wannier_xyz[
                            0 : 2 + main_json["system_auto"][system_auto]["nb_atm"]
                        ]
                        wannier_xyz = [
                            " ".join(_.replace("\n", "").split()) for _ in wannier_xyz
                        ]
                        wannier_xyz = [_.split(" ")[1:] for _ in wannier_xyz]
                        wannier_array = np.asarray(
                            wannier_xyz, dtype=np.float64
                        ).flatten()
                        if system_disturbed_candidates_not_skipped_counter == 1:
                            wannier_array_raw = np.zeros(
                                (
                                    system_disturbed_candidates_count
                                    - system_candidates_disturbed_skipped,
                                    len(wannier_xyz) * 3,
                                )
                            )

                        wannier_array_raw[
                            system_disturbed_candidates_not_skipped_counter - 1, :
                        ] = wannier_array
                        is_wannier = True
                        del wannier_array, wannier_xyz

                        # Check if wannier centers are not converged
                        output_cp2k = textfile_to_string_list(
                            labeling_step_path
                            / f"2_labeling_{padded_labeling_step}.out"
                        )
                        if any(
                            "LOCALIZATION! loop did not converge within the maximum number of iterations"
                            in _
                            for _ in output_cp2k
                        ):
                            wannier_not_converged.append(
                                f"{system_disturbed_candidates_not_skipped_counter - 1}\n"
                            )

                del padded_labeling_step, labeling_step, labeling_step_path

                np.savetxt(system_path / "energy.raw", energy_array_raw, delimiter=" ")
                np.save(data_path / "set.000" / "energy", energy_array_raw)
                del energy_array_raw

                np.savetxt(system_path / "coord.raw", coord_array_raw, delimiter=" ")
                np.save(data_path / "set.000" / "coord", coord_array_raw)
                del coord_array_raw

                np.savetxt(system_path / "box.raw", box_array_raw, delimiter=" ")
                np.save(data_path / "set.000" / "box", box_array_raw)
                del box_array_raw, volume_array_raw

                np.savetxt(system_path / "force.raw", force_array_raw, delimiter=" ")
                np.save(data_path / "set.000" / "force", force_array_raw)
                del force_array_raw

                if is_virial:
                    np.savetxt(
                        system_path / "virial.raw", virial_array_raw, delimiter=" "
                    )
                    np.save(data_path / "set.000" / "virial", virial_array_raw)
                del virial_array_raw

                if is_wannier:
                    np.savetxt(
                        system_path / "wannier.raw", wannier_array_raw, delimiter=" "
                    )
                    np.save(data_path / "set.000" / "wannier", wannier_array_raw)
                    if len(wannier_not_converged) > 1:
                        string_list_to_textfile(
                            data_path / "set.000" / "wannier_not-converged.txt",
                            wannier_not_converged,
                        )
                    del wannier_not_converged

        logging.info(
            f"Processed system: {system_auto} ({system_auto_index + 1}/{len(main_json['systems_auto'])})"
        )

    logging.info(f"-" * 88)
    # Update the booleans in the exploration JSON
    # if candidates_expected_count == (candidates_step_count[1] + candidates_skipped_count):
    labeling_json["is_extracted"] = True
    # del candidates_expected_count, candidates_skipped_count, candidates_step_count

    # Dump the JSON files (exploration JSONN)
    write_json_file(labeling_json, (control_path / f"labeling_{padded_curr_iter}.json"))

    # End
    logging.info(f"-" * 88)
    logging.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del (user_input_json_filename,)
    del main_json, labeling_json
    del curr_iter, padded_curr_iter

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "labeling",
            "extract",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass
