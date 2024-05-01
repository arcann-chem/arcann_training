"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/01
"""

# Standard library modules
import importlib
import logging
import sys
from pathlib import Path

# Non-standard library imports
import numpy as np

# Local imports
from deepmd_iterative.common.json import load_json_file, write_json_file
from deepmd_iterative.common.list import textfile_to_string_list, string_list_to_textfile
from deepmd_iterative.common.filesystem import check_file_existence
from deepmd_iterative.common.parsing_labeling import extract_and_convert_energy, extract_and_convert_forces, extract_and_convert_virial, extract_and_convert_wannier, extract_and_convert_box_volume, extract_and_convert_coordinates
from deepmd_iterative.common.check import validate_step_folder

# Import constants
try:
    importlib.import_module("scipy")
    from scipy import constants

    Ha_to_eV = constants.physical_constants["atomic unit of electric potential"][0]
    Bohr_to_A = constants.physical_constants["Bohr radius"][0] / constants.angstrom
    au_to_eV_per_A = np.float64(Ha_to_eV / Bohr_to_A)
    eV_per_A3_to_GPa = np.float64(constants.eV / constants.angstrom**3 / constants.giga)
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
    # Get the logger
    arcann_logger = logging.getLogger("ArcaNN")

    # Get the current path and set the training path as the parent of the current path
    current_path = Path(".").resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    arcann_logger.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}.")
    arcann_logger.debug(f"Current path :{current_path}")
    arcann_logger.debug(f"Training path: {training_path}")
    arcann_logger.debug(f"Program path: {deepmd_iterative_path}")
    arcann_logger.info(f"-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)

    # Get control path, load the main JSON and the exploration JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    labeling_json = load_json_file((control_path / f"labeling_{padded_curr_iter}.json"))

    labeling_program = labeling_json["labeling_program"]
    arcann_logger.debug(f"labeling_program: {labeling_program}")

    # Check if we can continue
    if not labeling_json["is_checked"]:
        arcann_logger.error(f"Lock found. Execute first: labeling launch.")
        arcann_logger.error(f"Aborting...")
        return 1

    # Create if it doesn't exists the data path.
    (training_path / "data").mkdir(exist_ok=True)

    for system_auto_index, system_auto in enumerate(labeling_json["systems_auto"]):
        arcann_logger.info(f"Processing system: {system_auto} ({system_auto_index + 1}/{len(main_json['systems_auto'])})")

        system_candidates_count = labeling_json["systems_auto"][system_auto]["candidates_count"]
        system_candidates_skipped_count = labeling_json["systems_auto"][system_auto]["candidates_skipped_count"]

        if system_candidates_count - system_candidates_skipped_count == 0:
            arcann_logger.debug(f"No label for this system {system_auto}, skipping")
            continue

        system_path = current_path / system_auto

        data_path = training_path / "data" / (system_auto + "_" + padded_curr_iter)
        data_path.mkdir(exist_ok=True)
        (data_path / "set.000").mkdir(exist_ok=True)

        energy_array_raw = np.zeros((system_candidates_count - system_candidates_skipped_count), dtype=np.float64)
        coord_array_raw = np.zeros((system_candidates_count - system_candidates_skipped_count, main_json["systems_auto"][system_auto]["nb_atm"] * 3), dtype=np.float64)
        box_array_raw = np.zeros((system_candidates_count - system_candidates_skipped_count, 9), dtype=np.float64)
        volume_array_raw = np.zeros((system_candidates_count - system_candidates_skipped_count), dtype=np.float64)
        force_array_raw = np.zeros((system_candidates_count - system_candidates_skipped_count, main_json["systems_auto"][system_auto]["nb_atm"] * 3,), dtype=np.float64)
        virial_array_raw = np.zeros((system_candidates_count - system_candidates_skipped_count, 9), dtype=np.float64)
        print(energy_array_raw)
        # Options
        is_virial = False
        is_wannier = False

        # Wannier
        wannier_not_converged = ["#Indexes start at 0\n"]

        # counter for non skipped configurations
        system_candidates_not_skipped_counter = 0

        arcann_logger.debug("Starting extraction...")
        for labeling_step in range(system_candidates_count):
            padded_labeling_step = str(labeling_step).zfill(5)
            labeling_step_path = system_path / padded_labeling_step

            if not (labeling_step_path / "skip").is_file():
                system_candidates_not_skipped_counter += 1
                # With the first, we create a type.raw and get the CP2K version
                if system_candidates_not_skipped_counter == 1:
                    check_file_existence(training_path / "user_files" / f"{system_auto}.lmp", True, True, "Input data file (lmp) not present.")

                    lammps_data = textfile_to_string_list(training_path / "user_files" / f"{system_auto}.lmp")
                    indexes = [idx for idx, s in enumerate(lammps_data) if "Atoms" in s]
                    if len(indexes) > 1:
                        for index in [idx for idx, s in enumerate(lammps_data) if "Atoms" in s]:
                            atom_list = [line.strip().split() for line in lammps_data[index + 2 : index + 4]]
                            if len(atom_list[0]) == len(atom_list[1]) and lammps_data[index + 1] == " \n" and atom_list[0][0] == "1" and atom_list[1][0] == "2":
                                idx = index
                                break
                    else:
                        idx = indexes[0]
                    del lammps_data[0 : idx + 2]
                    lammps_data = lammps_data[0 : main_json["systems_auto"][system_auto]["nb_atm"] + 1]
                    lammps_data = [" ".join(f.replace("\n", "").split()) for f in lammps_data]
                    lammps_data = [g.split(" ")[1:2] for g in lammps_data]
                    type_atom_array = np.asarray(lammps_data, dtype=np.int64).flatten()
                    type_atom_array = type_atom_array - 1
                    np.savetxt(f"{system_path}/type.raw", type_atom_array, delimiter=" ", newline=" ", fmt="%d")
                    np.savetxt(f"{data_path}/type.raw", type_atom_array, delimiter=" ", newline=" ", fmt="%d")
    
                    # Get the CP2K/Orca version
                    if labeling_program == "cp2k":
                        output_cp2k = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}.out")
                        output_cp2k = [_ for _ in output_cp2k if "CP2K| version string:" in _]
                        output_cp2k = [" ".join(_.replace("\n", "").split()) for _ in output_cp2k]
                        output_cp2k = [_.split(" ")[-1] for _ in output_cp2k]
                        program_version = float(output_cp2k[0])
                    elif labeling_program == "orca":
                        output_orca = textfile_to_string_list(labeling_step_path / f"1_labeling_{padded_labeling_step}.out")
                        output_orca = [_ for _ in output_orca if "Program Version" in _]
                        program_version = float(output_orca[0].split(" ")[2][0])

                # Coordinates
                coordinate_xyz = textfile_to_string_list(labeling_step_path / f"labeling_{padded_labeling_step}.xyz")
                coord_array_raw = extract_and_convert_coordinates(coordinate_xyz, coord_array_raw, system_candidates_not_skipped_counter)
                del coordinate_xyz

                if labeling_program == "cp2k":

                    # Energy
                    energy_cp2k = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}-Force_Eval.fe")
                    energy_array_raw = extract_and_convert_energy(energy_cp2k, energy_array_raw, system_candidates_not_skipped_counter, Ha_to_eV, labeling_program, program_version)
                    del energy_cp2k

                    # Box / Volume
                    input_cp2k = textfile_to_string_list(labeling_step_path / f"1_labeling_{padded_labeling_step}.inp")
                    box_array_raw, volume_array_raw = extract_and_convert_box_volume(input_cp2k, box_array_raw, volume_array_raw, system_candidates_not_skipped_counter, 1.0, labeling_program, program_version)
                    del input_cp2k

                    # Forces
                    force_cp2k = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}-Forces.for")
                    force_array_raw = extract_and_convert_forces(force_cp2k, force_array_raw, system_candidates_not_skipped_counter, au_to_eV_per_A, labeling_program, program_version)
                    del force_cp2k

                    # Virial
                    if (labeling_step_path / f"2_labeling_{padded_labeling_step}-Stress_Tensor.st").is_file():
                        stress_cp2k = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}-Stress_Tensor.st")
                        virial_array_raw, is_virial = extract_and_convert_virial(stress_cp2k, virial_array_raw, system_candidates_not_skipped_counter, volume_array_raw, eV_per_A3_to_GPa, labeling_program, program_version)
                        del stress_cp2k

                    # Wannier
                    if (labeling_step_path / f"2_labeling_{padded_labeling_step}-Wannier.xyz").is_file():
                        output_cp2k = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}.out")
                        wannier_xyz = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}-Wannier.xyz")
                        if system_candidates_not_skipped_counter == 1:
                            wannier_array_raw = np.zeros((system_candidates_count - system_candidates_skipped_count, (len(wannier_xyz)-2-main_json["systems_auto"][system_auto]["nb_atm"]) * 3), dtype=np.float64)
                        wannier_array_raw, is_wannier = extract_and_convert_wannier(wannier_xyz, wannier_array_raw, system_candidates_not_skipped_counter, main_json["systems_auto"][system_auto]["nb_atm"], 1.0, labeling_program, program_version)
                        if any("LOCALIZATION! loop did not converge within the maximum number of iterations" in _ for _ in output_cp2k):
                            wannier_not_converged.append(f"{system_candidates_not_skipped_counter - 1}\n")
                        del wannier_xyz, output_cp2k

                elif labeling_program == "orca":
                    # Energy
                    energy_orca = textfile_to_string_list(labeling_step_path / f"1_labeling_{padded_labeling_step}.engrad")
                    energy_array_raw = extract_and_convert_energy(energy_orca, energy_array_raw, system_candidates_not_skipped_counter, Ha_to_eV, labeling_program, program_version)
                    del energy_orca
                    
                    # Box / Volume
                    # TODO
                    system_cell = main_json["systems_auto"][system_auto]["cell"]
                    box_array_raw, volume_array_raw = extract_and_convert_box_volume(system_cell, box_array_raw, volume_array_raw, system_candidates_not_skipped_counter, 1.0, labeling_program, program_version)
                    del system_cell

                    # Forces
                    force_orca = textfile_to_string_list(labeling_step_path / f"1_labeling_{padded_labeling_step}.engrad")
                    force_array_raw = extract_and_convert_forces(force_orca, force_array_raw, system_candidates_not_skipped_counter, au_to_eV_per_A, labeling_program, program_version)
                    del force_orca

                    # Virial
                    # No virial in ORCA

                    # Wannier
                    # No wannier in ORCA

        del padded_labeling_step, labeling_step, labeling_step_path, system_candidates_not_skipped_counter

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
                string_list_to_textfile(data_path / "set.000" / "wannier_not-converged.txt", wannier_not_converged)
            del wannier_not_converged, wannier_array_raw, is_wannier

        arcann_logger.debug("Extraction done.")

        system_disturbed_candidates_count = labeling_json["systems_auto"][system_auto]["disturbed_candidates_count"]
        system_disturbed_candidates_skipped_count = labeling_json["systems_auto"][system_auto]["disturbed_candidates_skipped_count"]

        if system_disturbed_candidates_count - system_disturbed_candidates_skipped_count > 0:
            arcann_logger.debug("Starting extraction for disturbed...")
            data_path = training_path / "data" / (system_auto + "-disturbed_" + padded_curr_iter)
            data_path.mkdir(exist_ok=True)
            (data_path / "set.000").mkdir(exist_ok=True)

            energy_array_raw = np.zeros((system_disturbed_candidates_count - system_disturbed_candidates_skipped_count), dtype=np.float64)
            coord_array_raw = np.zeros((system_disturbed_candidates_count - system_disturbed_candidates_skipped_count, main_json["systems_auto"][system_auto]["nb_atm"] * 3), dtype=np.float64)
            box_array_raw = np.zeros((system_disturbed_candidates_count - system_disturbed_candidates_skipped_count, 9), dtype=np.float64)
            volume_array_raw = np.zeros((system_disturbed_candidates_count - system_disturbed_candidates_skipped_count), dtype=np.float64)
            force_array_raw = np.zeros((system_disturbed_candidates_count - system_disturbed_candidates_skipped_count, main_json["systems_auto"][system_auto]["nb_atm"] * 3), dtype=np.float64)
            virial_array_raw = np.zeros((system_disturbed_candidates_count - system_disturbed_candidates_skipped_count, 9), dtype=np.float64)

            # Options
            is_virial = False
            is_wannier = False

            # Wannier
            wannier_not_converged = ["#Indexes start at 0\n"]

            # counter for non skipped configurations
            system_disturbed_candidates_not_skipped_counter = 0

            for labeling_step in range(system_candidates_count, system_candidates_count + system_disturbed_candidates_count):
                padded_labeling_step = str(labeling_step).zfill(5)
                labeling_step_path = system_path / padded_labeling_step

                if not (labeling_step_path / "skip").is_file():
                    system_disturbed_candidates_not_skipped_counter += 1

                    # With the first, we create a type.raw and get the CP2K version
                    if system_disturbed_candidates_not_skipped_counter == 1:
                        check_file_existence(training_path / "user_files" / f"{system_auto}.lmp", True, True, "Input data file (lmp) not present.")

                        lammps_data = textfile_to_string_list(training_path / "user_files" / f"{system_auto}.lmp")
                        indexes = [idx for idx, s in enumerate(lammps_data) if "Atoms" in s]
                        if len(indexes) > 1:
                            for index in [idx for idx, s in enumerate(lammps_data) if "Atoms" in s]:
                                atom_list = [line.strip().split() for line in lammps_data[index + 2 : index + 4]]
                                if len(atom_list[0]) == len(atom_list[1]) and lammps_data[index + 1] == " \n" and atom_list[0][0] == "1" and atom_list[1][0] == "2":
                                    idx = index
                                    break
                        else:
                            idx = indexes[0]
                        del lammps_data[0 : idx + 2]
                        lammps_data = lammps_data[0 : main_json["systems_auto"][system_auto]["nb_atm"] + 1]
                        lammps_data = [" ".join(f.replace("\n", "").split()) for f in lammps_data]
                        lammps_data = [g.split(" ")[1:2] for g in lammps_data]
                        type_atom_array = np.asarray(lammps_data, dtype=np.int64).flatten()
                        type_atom_array = type_atom_array - 1
                        np.savetxt(f"{system_path}/type.raw", type_atom_array, delimiter=" ", newline=" ", fmt="%d")
                        np.savetxt(f"{data_path}/type.raw", type_atom_array, delimiter=" ", newline=" ", fmt="%d")

                        # Get the CP2K/Orca version
                        if labeling_program == "cp2k":
                            output_cp2k = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}.out")
                            output_cp2k = [_ for _ in output_cp2k if "CP2K| version string:" in _]
                            output_cp2k = [" ".join(_.replace("\n", "").split()) for _ in output_cp2k]
                            output_cp2k = [_.split(" ")[-1] for _ in output_cp2k]
                            program_version = float(output_cp2k[0])
                        elif labeling_program == "orca":
                            output_orca = textfile_to_string_list(labeling_step_path / f"1_labeling_{padded_labeling_step}.out")
                            output_orca = [_ for _ in output_orca if "Program Version" in _]
                            program_version = float(output_orca[0].split(" ")[2][0])

                    # Coordinates
                    coordinate_xyz = textfile_to_string_list(labeling_step_path / f"labeling_{padded_labeling_step}.xyz")
                    coord_array_raw = extract_and_convert_coordinates(coordinate_xyz, coord_array_raw, system_disturbed_candidates_not_skipped_counter)
                    del coordinate_xyz

                    if labeling_program == "cp2k":
                        # Energy
                        energy_cp2k = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}-Force_Eval.fe")
                        energy_array_raw = extract_and_convert_energy(energy_cp2k, energy_array_raw, system_disturbed_candidates_not_skipped_counter, Ha_to_eV, labeling_program, program_version)
                        del energy_cp2k

                        # Box / Volume
                        input_cp2k = textfile_to_string_list(labeling_step_path / f"1_labeling_{padded_labeling_step}.inp")
                        box_array_raw, volume_array_raw = extract_and_convert_box_volume(input_cp2k, box_array_raw, volume_array_raw, system_disturbed_candidates_not_skipped_counter, 1.0, labeling_program, program_version)
                        del input_cp2k

                        # Forces
                        force_cp2k = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}-Forces.for")
                        force_array_raw = extract_and_convert_forces(force_cp2k, force_array_raw, system_disturbed_candidates_not_skipped_counter, au_to_eV_per_A, labeling_program, program_version)
                        del force_cp2k

                        # Virial
                        if (labeling_step_path / f"2_labeling_{padded_labeling_step}-Stress_Tensor.st").is_file():
                            stress_cp2k = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}-Stress_Tensor.st")
                            virial_array_raw, is_virial = extract_and_convert_virial(stress_cp2k, virial_array_raw, system_disturbed_candidates_not_skipped_counter, volume_array_raw, eV_per_A3_to_GPa, labeling_program, program_version)
                            del stress_cp2k

                        # Wannier
                        if (labeling_step_path / f"2_labeling_{padded_labeling_step}-Wannier.xyz").is_file():
                            output_cp2k = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}.out")
                            wannier_xyz = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}-Wannier.xyz")
                            if system_disturbed_candidates_not_skipped_counter == 1:
                                wannier_array_raw = np.zeros((system_candidates_count - system_candidates_skipped_count, (len(wannier_xyz)-2-main_json["systems_auto"][system_auto]["nb_atm"]) * 3), dtype=np.float64)
                            wannier_array_raw, is_wannier = extract_and_convert_wannier(wannier_xyz, wannier_array_raw, system_disturbed_candidates_not_skipped_counter, main_json["systems_auto"][system_auto]["nb_atm"], 1.0, labeling_program, program_version)
                            if any("LOCALIZATION! loop did not converge within the maximum number of iterations" in _ for _ in output_cp2k):
                                wannier_not_converged.append(f"{system_disturbed_candidates_not_skipped_counter - 1}\n")
                            del wannier_xyz, output_cp2k

                    elif labeling_program == "orca":
                        # Energy
                        energy_orca = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}.engrad")
                        energy_array_raw = extract_and_convert_energy(energy_orca, energy_array_raw, system_disturbed_candidates_not_skipped_counter, Ha_to_eV, labeling_program, program_version)
                        del energy_orca

                        # Box / Volume
                        # TODO
                        system_cell = main_json["systems_auto"][system_auto]["cell"]
                        #box_array_raw, volume_array_raw = extract_and_convert_box_volume(energy_orca, box_array_raw, volume_array_raw, system_candidates_not_skipped_counter, 1.0, labeling_program, program_version)
                        box_array_raw[system_candidates_not_skipped_counter, 0] = system_cell[0]
                        box_array_raw[system_candidates_not_skipped_counter, 4] = system_cell[1]
                        box_array_raw[system_candidates_not_skipped_counter, 8] = system_cell[2]
                        del system_cell

                        # Forces
                        force_orca = textfile_to_string_list(labeling_step_path / f"2_labeling_{padded_labeling_step}.engrad")
                        force_array_raw = extract_and_convert_forces(force_orca, force_array_raw, system_disturbed_candidates_not_skipped_counter, au_to_eV_per_A, labeling_program, program_version)
                        del force_orca

                        # Virial
                        # No virial in ORCA

                        # Wannier
                        # No wannier in ORCA

            del padded_labeling_step, labeling_step, labeling_step_path, system_disturbed_candidates_not_skipped_counter

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
                    string_list_to_textfile(data_path / "set.000" / "wannier_not-converged.txt", wannier_not_converged)
                del wannier_not_converged, wannier_array_raw, is_wannier
            arcann_logger.debug("Extraction for disturbed done.")

        arcann_logger.info(f"Processed system: {system_auto} ({system_auto_index + 1}/{len(main_json['systems_auto'])})")

    if "output_cp2k" in locals():
        del output_cp2k
    if "output_orca" in locals():
        del output_orca

    del system_auto, system_auto_index
    del system_candidates_count, system_candidates_skipped_count, system_path, data_path
    del indexes, idx, type_atom_array, lammps_data
    del program_version
    del system_disturbed_candidates_count, system_disturbed_candidates_skipped_count

    arcann_logger.info(f"-" * 88)
    # Update the booleans in the exploration JSON
    labeling_json["is_extracted"] = True

    # Dump the JSON files (exploration JSONN)
    write_json_file(labeling_json, (control_path / f"labeling_{padded_curr_iter}.json"))

    # End
    arcann_logger.info(f"-" * 88)
    arcann_logger.info(f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!")

    # Cleaning
    del current_path, control_path, training_path
    del user_input_json_filename
    del main_json, labeling_json
    del curr_iter, padded_curr_iter

    arcann_logger.debug(f"LOCAL")
    arcann_logger.debug(f"{locals()}")
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
