"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15

Test cases for the (training) utils module.

Classes
-------
TestCreateModelsList():
    Test case for the 'create_models_list' function.
TestGetLastFrameNumber():
    Test case for the 'get_last_frame_number' function.
TestUpdateNbStepsFactor():
    Test case for the 'update_system_nb_steps_factor' function.
"""

# Standard library modules
import json
import tempfile
import unittest
from pathlib import Path

# Third-party modules
import numpy as np

# Local imports
from arcann_training.exploration.utils import (
    create_models_list,
    get_last_frame_number,
    update_system_nb_steps_factor,
)


class TestCreateModelsList(unittest.TestCase):
    """
    Test case for the 'create_models_list' function.

    Methods
    -------
    test_create_models_list():
        Test the 'create_models_list' function with various inputs and validate the output.
    """

    def setUp(self):
        # Create a temporary directory for the test
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create temporary training and local directories for the test
        self.training_dir = Path(self.temp_dir.name) / "training"
        self.nnp_dir = Path(self.training_dir) / "NNP"
        self.local_dir = Path(self.temp_dir.name) / "local"
        self.training_dir.mkdir()
        self.local_dir.mkdir()
        self.nnp_dir.mkdir()

        # Create a temporary config JSON file for the test
        self.config_file = Path(self.temp_dir.name) / "config.json"
        with self.config_file.open(mode="w") as f:
            f.write('{"nnp_count": 3}')

        # Create a temporary prevtraining JSON file for the test
        self.prevtraining_file = Path(self.temp_dir.name) / "prevtraining.json"
        with self.prevtraining_file.open(mode="w") as f:
            f.write('{"is_compressed": true}')

        # Create sample NNP model files for the test
        for i in range(1, 4):
            nnp_file = self.training_dir / "NNP" / f"graph_{i}_000_compressed.pb"
            nnp_file.touch()

    def tearDown(self):
        # Clean up the temporary directories and files
        self.config_file.unlink()
        self.prevtraining_file.unlink()
        for i in range(1, 4):
            nnp_file = self.training_dir / "NNP" / f"graph_{i}_000_compressed.pb"
            nnp_file.unlink()
            nnp_file = self.local_dir / f"graph_{i}_000_compressed.pb"
            nnp_file.unlink()
        self.nnp_dir.rmdir()
        self.training_dir.rmdir()
        self.local_dir.rmdir()
        self.temp_dir.cleanup()

    def test_create_models_list(self):
        """
        Test the 'create_models_list' function with various inputs and validate the output.
        """
        # Load the config JSON and prevtraining JSON files for the test
        with self.config_file.open(mode="r") as f:
            config_json = json.load(f)
        with self.prevtraining_file.open(mode="r") as f:
            prevtraining_json = json.load(f)

        # Test the function with various inputs
        models_list, models_string = create_models_list(
            config_json, prevtraining_json, 2, "000", self.training_dir, self.local_dir
        )
        expected_models_list = [
            "graph_2_000_compressed.pb",
            "graph_3_000_compressed.pb",
            "graph_1_000_compressed.pb",
        ]
        expected_models_string = "graph_2_000_compressed.pb graph_3_000_compressed.pb graph_1_000_compressed.pb"
        self.assertListEqual(models_list, expected_models_list)
        self.assertEqual(models_string, expected_models_string)
        for i in range(1, 4):
            nnp_link = self.local_dir / f"graph_{i}_000_compressed.pb"
            self.assertTrue(nnp_link.is_symlink())
            self.assertEqual(
                nnp_link.resolve(),
                self.training_dir / "NNP" / f"graph_{i}_000_compressed.pb",
            )


class TestGetLastFrameNumber(unittest.TestCase):
    """
    Test case for the 'get_last_frame_number' function.

    Methods
    -------
    test_get_last_frame_number():
        Test the 'get_last_frame_number' function with various inputs and validate the output.
    """

    def test_get_last_frame_number(self):
        """
        Test the 'get_last_frame_number' function with various inputs and validate the output.
        """
        # Test the function with various inputs
        model_deviation = np.array(
            [
                [1, 2, 3, 4, 0.1],
                [2, 3, 4, 5, 0.2],
                [3, 4, 5, 6, 0.3],
                [4, 5, 6, 7, 0.4],
                [5, 6, 7, 8, 0.5],
                [6, 7, 8, 9, 0.6],
            ]
        )
        self.assertEqual(get_last_frame_number(model_deviation, 0.2, False), 1)
        self.assertEqual(get_last_frame_number(model_deviation, 0.4, False), 3)
        self.assertEqual(get_last_frame_number(model_deviation, 0.6, False), 5)
        self.assertEqual(get_last_frame_number(model_deviation, 0.7, False), -1)
        self.assertEqual(get_last_frame_number(model_deviation, 0.2, True), 0)
        self.assertEqual(get_last_frame_number(model_deviation, 0.4, True), 2)
        self.assertEqual(get_last_frame_number(model_deviation, 0.6, True), 4)
        self.assertEqual(get_last_frame_number(model_deviation, 0.7, True), -1)


class TestUpdateSystemNbStepsFactor(unittest.TestCase):
    """
    Test case for the 'update_system_nb_steps_factor' function.

    Methods
    -------
    test_update_nb_steps_factor():
        Test the 'update_system_nb_steps_factor' function with various inputs and validate the output.
    """

    def setUp(self):
        # Create a temporary directory for the test
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a temporary JSON file for the test
        self.temp_file = Path(self.temp_dir.name) / "prevexploration.json"
        with self.temp_file.open(mode="w") as f:
            f.write(
                '{"systems_auto": [{"candidates_count": 5, "rejected_count": 0, "total_count": 100, "nb_steps": 100, "timestep_ps": 1}]}'
            )

    def tearDown(self):
        # Clean up the temporary directory and file
        self.temp_file.unlink()
        self.temp_dir.cleanup()

    def test_update_nb_steps_factor(self):
        # Load the JSON file for the test
        with self.temp_file.open(mode="r") as f:
            prevexploration_json = json.load(f)

        # Test the function for various ratios of ill-described candidates
        self.assertEqual(update_system_nb_steps_factor(prevexploration_json, 0), 400)
        prevexploration_json["systems_auto"][0]["rejected_count"] = 5
        self.assertEqual(update_system_nb_steps_factor(prevexploration_json, 0), 200)
        prevexploration_json["systems_auto"][0]["candidates_count"] = 20
        self.assertEqual(update_system_nb_steps_factor(prevexploration_json, 0), 100)


if __name__ == "__main__":
    unittest.main()
