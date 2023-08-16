"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/16

Test cases for the (training) utils module.

Class
-----
TestCalculateDecaySteps
    Test case for the calculate_decay_steps() function.
TestCalculateDecayRate
    Test case for the calculate_decay_steps() function.
TestCalculateLearningRate
    Test case for the calculate_learning_rate() function.
TestCheckInitialDatasets
    Test case for the check_initial_datasets() function.
"""
# Standard library modules
import tempfile
import unittest
from pathlib import Path

# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.training.utils import (
    calculate_decay_steps,
    calculate_decay_rate,
    calculate_learning_rate,
    check_initial_datasets,
)


class TestCalculateDecaySteps(unittest.TestCase):
    """
    Test case for the calculate_decay_steps() function.

    Methods
    -------
    test_calculate_decay_steps_valid_input():
        Tests the function with valid inputs.
    test_calculate_decay_steps_invalid_input():
        Tests the function with invalid inputs.
    test_calculate_decay_steps_output_type():
        Tests the type of output returned by the function, i.e., an integer.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_calculate_decay_steps_positive_input(self):
        self.assertEqual(calculate_decay_steps(20000), 5000)
        self.assertEqual(calculate_decay_steps(50000), 12500)
        self.assertEqual(calculate_decay_steps(60000), 15000)
        self.assertEqual(calculate_decay_steps(100000), 25000)
        self.assertEqual(calculate_decay_steps(150000), 30000)

    def test_calculate_decay_steps_invalid_input(self):
        with self.assertRaises(ValueError) as cm:
            calculate_decay_steps(0)
        error_msg = str(cm.exception)
        expected_error_msg = f"nb_structures must be a positive integer"
        self.assertEqual(error_msg, expected_error_msg)

        with self.assertRaises(ValueError) as cm:
            calculate_decay_steps(-100)
        error_msg = str(cm.exception)
        expected_error_msg = f"nb_structures must be a positive integer"
        self.assertEqual(error_msg, expected_error_msg)

        with self.assertRaises(ValueError) as cm:
            calculate_decay_steps(100, min_decay_steps=-500)
        error_msg = str(cm.exception)
        expected_error_msg = f"min_decay_steps must be a positive integer"
        self.assertEqual(error_msg, expected_error_msg)

    def test_calculate_decay_steps_output_type(self):
        self.assertIsInstance(calculate_decay_steps(20000), int)


class TestCalculateDecayRate(unittest.TestCase):
    """
    Test case for the calculate_decay_steps() function.

    Methods
    -------
    test_calculate_decay_rate_valid_input():
        Tests the function with valid inputs.
    test_calculate_decay_rate_invalid_input():
        Tests the function with invalid inputs.
    test_calculate_decay_rate_output_type():
        Tests the type of output returned by the function, i.e., an float.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_calculate_decay_rate_valid_input(self):
        self.assertAlmostEqual(
            calculate_decay_rate(50000, 0.01, 0.001, 5000), 0.7943282347242815, places=7
        )
        self.assertAlmostEqual(
            calculate_decay_rate(200000, 0.05, 0.005, 10000),
            0.8912509381337456,
            places=7,
        )
        self.assertAlmostEqual(
            calculate_decay_rate(500000, 0.1, 0.01, 25000), 0.8912509381337456, places=7
        )

    def test_calculate_decay_rate_invalid_input(self):
        with self.assertRaises(ValueError) as cm:
            calculate_decay_rate(100, -0.01, 0.001, 5000)
        error_msg = str(cm.exception)
        expected_error_msg = f"start_lr must be a positive number."
        self.assertEqual(error_msg, expected_error_msg)
        with self.assertRaises(ValueError) as cm:
            calculate_decay_rate(100, 0, 0.001, 5000)
        error_msg = str(cm.exception)
        expected_error_msg = f"start_lr must be a positive number."
        self.assertEqual(error_msg, expected_error_msg)
        with self.assertRaises(ValueError) as cm:
            calculate_decay_rate(100, 0.01, 0.001, 0)
        error_msg = str(cm.exception)
        expected_error_msg = f"decay_steps must be a positive integer."
        self.assertEqual(error_msg, expected_error_msg)
        with self.assertRaises(ValueError) as cm:
            calculate_decay_rate(100, 0.01, 0.001, 0.0003)
        error_msg = str(cm.exception)
        expected_error_msg = f"decay_steps must be a positive integer."
        self.assertEqual(error_msg, expected_error_msg)

    def test_calculate_decay_rate_output_type(self):
        self.assertIsInstance(calculate_decay_rate(100, 0.01, 0.001, 5000), float)


class TestCalculateLearningRate(unittest.TestCase):
    """
    Test case for the calculate_learning_rate() function.

    Methods
    -------
    test_calculate_learning_rate_valid_input():
        Tests the function with valid inputs.
    test_calculate_learning_rate_invalid_input():
        Tests the function with invalid inputs.
    test_calculate_learning_rate_output_type():
        Tests the type of output returned by the function, i.e., an float.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_calculate_learning_rate_valid_input(self):
        self.assertAlmostEqual(
            calculate_learning_rate(10000, 0.01, 0.7875603898650455, 5000),
            0.006202513676843825,
            places=7,
        )
        self.assertAlmostEqual(
            calculate_learning_rate(20000, 0.05, 0.8613440861579459, 10000),
            0.03709568173796334,
            places=7,
        )
        self.assertAlmostEqual(
            calculate_learning_rate(500000, 0.1, 0.9082829387412657, 25000),
            0.014602362613303355,
            places=7,
        )

    def test_calculate_learning_rate_invalid_input(self):
        with self.assertRaises(ValueError) as cm:
            calculate_learning_rate(-100, 0.01, 0.1, 5000)
        error_msg = str(cm.exception)
        expected_error_msg = f"All arguments must be positive."
        self.assertEqual(error_msg, expected_error_msg)
        with self.assertRaises(ValueError) as cm:
            calculate_learning_rate(100, -0.01, 0.1, 5000)
        error_msg = str(cm.exception)
        expected_error_msg = f"All arguments must be positive."
        self.assertEqual(error_msg, expected_error_msg)
        with self.assertRaises(ValueError) as cm:
            calculate_learning_rate(100, 0.01, -0.1, 5000)
        error_msg = str(cm.exception)
        expected_error_msg = f"All arguments must be positive."
        self.assertEqual(error_msg, expected_error_msg)
        with self.assertRaises(ValueError) as cm:
            calculate_learning_rate(1000, 0.01, 0.1, -5000)
        error_msg = str(cm.exception)
        expected_error_msg = f"All arguments must be positive."
        self.assertEqual(error_msg, expected_error_msg)
        with self.assertRaises(ValueError) as cm:
            calculate_learning_rate(100, 0.01, 0.1, 3213332.2)
        error_msg = str(cm.exception)
        expected_error_msg = f"decay_steps must be a positive integer"
        self.assertEqual(error_msg, expected_error_msg)

    def test_calculate_learning_rate_output_type(self):
        self.assertIsInstance(calculate_learning_rate(30000, 0.01, 0.1, 5000), float)


class TestCheckInitialDatasets(unittest.TestCase):
    """
    Test case for the check_initial_datasets() function.

    Methods
    -------
    test_check_initial_datasets():
        Tests if the function returns the correct dictionary of initial dataset names and number of samples.
    test_check_initial_datasets_invalid_num_samples():
        Tests if the function raises a ValueError when one of the initial datasets has an invalid number of samples.
    test_check_initial_datasets_missing_json():
        Tests if the function raises a FileNotFoundError when the initial_datasets.json file is missing.
    test_check_initial_datasets_missing_dataset():
        Tests if the function raises a FileNotFoundError when one of the initial datasets is missing.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(self.temp_dir.name)

        (temp_dir_path / "data" / "dataset1" / "set.000").mkdir(parents=True)
        (temp_dir_path / "data" / "dataset2" / "set.000").mkdir(parents=True)
        (temp_dir_path / "control").mkdir()

        with (temp_dir_path / "control" / "initial_datasets.json").open("w") as file:
            file.write('{"dataset1": 100, "dataset2": 200}')

        np.save(
            temp_dir_path / "data" / "dataset1" / "set.000" / "box.npy", np.zeros(100)
        )
        np.save(
            temp_dir_path / "data" / "dataset2" / "set.000" / "box.npy", np.zeros(200)
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_initial_datasets(self):
        expected_result = {"dataset1": 100, "dataset2": 200}
        self.assertDictEqual(
            check_initial_datasets(Path(self.temp_dir.name)), expected_result
        )

    def test_check_initial_datasets_invalid_num_samples(self):
        np.save(
            Path(self.temp_dir.name) / "data" / "dataset1" / "set.000" / "box.npy",
            np.zeros(50),
        )
        with self.assertRaises(ValueError) as cm:
            check_initial_datasets(Path(self.temp_dir.name))

    def test_check_initial_datasets_missing_json(self):
        (Path(self.temp_dir.name) / "control" / "initial_datasets.json").unlink()
        with self.assertRaises(FileNotFoundError) as cm:
            check_initial_datasets(Path(self.temp_dir.name))

    def test_check_initial_datasets_missing_dataset(self):
        (
            Path(self.temp_dir.name) / "data" / "dataset2" / "set.000" / "box.npy"
        ).unlink()
        (Path(self.temp_dir.name) / "data" / "dataset2" / "set.000").rmdir()
        (Path(self.temp_dir.name) / "data" / "dataset2").rmdir()
        with self.assertRaises(FileNotFoundError) as cm:
            check_initial_datasets(Path(self.temp_dir.name))


if __name__ == "__main__":
    unittest.main()
