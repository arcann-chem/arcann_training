"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/29

Test cases for the (training) utils module.

Class
-----
TestCalculateDecaySteps():
    Test case for the 'calculate_decay_steps' function.
TestCalculateDecayRate():
    Test case for the 'calculate_decay_steps' function.
TestCalculateLearningRate():
    Test case for the 'calculate_learning_rate' function.
TestCheckInitialDatasets():
    Test case for the 'check_initial_datasets' function.
TestDeepMDConfigValidation():
    Test case for the 'validate_deepmd_config' function.
TestGenerateTrainingJson():
    Test case for the 'generate_training_json' function.

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
    validate_deepmd_config,
    generate_training_json,
)


class TestCalculateDecaySteps(unittest.TestCase):
    """
    Test case for the 'calculate_decay_steps' function.

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
        expected_error_msg = f"The argument 'num_structures' must be a positive integer."
        self.assertEqual(error_msg, expected_error_msg)

        with self.assertRaises(ValueError) as cm:
            calculate_decay_steps(-100)
        error_msg = str(cm.exception)
        expected_error_msg = f"The argument 'num_structures' must be a positive integer."
        self.assertEqual(error_msg, expected_error_msg)

        with self.assertRaises(ValueError) as cm:
            calculate_decay_steps(100, min_decay_steps=-500)
        error_msg = str(cm.exception)
        expected_error_msg = f"The argument 'min_decay_steps' must be a positive integer."
        self.assertEqual(error_msg, expected_error_msg)

    def test_calculate_decay_steps_output_type(self):
        self.assertIsInstance(calculate_decay_steps(20000), int)


class TestCalculateDecayRate(unittest.TestCase):
    """
    Test case for the 'calculate_decay_steps' function.

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
        self.assertAlmostEqual(calculate_decay_rate(50000, 0.01, 0.001, 5000), 0.7943282347242815, places=7)
        self.assertAlmostEqual(
            calculate_decay_rate(200000, 0.05, 0.005, 10000),
            0.8912509381337456,
            places=7,
        )
        self.assertAlmostEqual(calculate_decay_rate(500000, 0.1, 0.01, 25000), 0.8912509381337456, places=7)

    def test_calculate_decay_rate_invalid_input(self):
        with self.assertRaises(ValueError) as cm:
            calculate_decay_rate(100, -0.01, 0.001, 5000)
        error_msg = str(cm.exception)
        expected_error_msg = f"The argument 'start_lr' must be a positive number."
        self.assertEqual(error_msg, expected_error_msg)
        with self.assertRaises(ValueError) as cm:
            calculate_decay_rate(100, 0, 0.001, 5000)
        error_msg = str(cm.exception)
        expected_error_msg = f"The argument 'start_lr' must be a positive number."
        self.assertEqual(error_msg, expected_error_msg)
        with self.assertRaises(ValueError) as cm:
            calculate_decay_rate(100, 0.01, 0.001, 0)
        error_msg = str(cm.exception)
        expected_error_msg = f"The argument 'decay_steps' must be a positive integer."
        self.assertEqual(error_msg, expected_error_msg)
        with self.assertRaises(ValueError) as cm:
            calculate_decay_rate(100, 0.01, 0.001, 0.0003)
        error_msg = str(cm.exception)
        expected_error_msg = f"The argument 'decay_steps' must be a positive integer."
        self.assertEqual(error_msg, expected_error_msg)

    def test_calculate_decay_rate_output_type(self):
        self.assertIsInstance(calculate_decay_rate(100, 0.01, 0.001, 5000), float)


class TestCalculateLearningRate(unittest.TestCase):
    """
    Test case for the 'calculate_learning_rate' function.

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
        expected_error_msg = f"The argument 'decay_steps' must be a positive integer."
        self.assertEqual(error_msg, expected_error_msg)

    def test_calculate_learning_rate_output_type(self):
        self.assertIsInstance(calculate_learning_rate(30000, 0.01, 0.1, 5000), float)


class TestCheckInitialDatasets(unittest.TestCase):
    """
    Test case for the 'check_initial_datasets' function.

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

        np.save(temp_dir_path / "data" / "dataset1" / "set.000" / "box.npy", np.zeros(100))
        np.save(temp_dir_path / "data" / "dataset2" / "set.000" / "box.npy", np.zeros(200))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_initial_datasets(self):
        expected_result = {"dataset1": 100, "dataset2": 200}
        self.assertDictEqual(check_initial_datasets(Path(self.temp_dir.name)), expected_result)

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
        (Path(self.temp_dir.name) / "data" / "dataset2" / "set.000" / "box.npy").unlink()
        (Path(self.temp_dir.name) / "data" / "dataset2" / "set.000").rmdir()
        (Path(self.temp_dir.name) / "data" / "dataset2").rmdir()
        with self.assertRaises(FileNotFoundError) as cm:
            check_initial_datasets(Path(self.temp_dir.name))


class TestDeepMDConfigValidation(unittest.TestCase):
    """
    Test case for the 'validate_deepmd_config' function.

    Methods
    -------
    test_valid_config():
        Tests if the function correctly validates a valid configuration.
    test_invalid_model_version():
        Tests if the function raises a ValueError for an invalid model version.
    """

    def test_valid_config(self):
        """
        Tests if the function correctly validates a valid configuration.
        """
        config = {
            "deepmd_model_version": 2.1,
            "arch_name": "a100",
        }
        self.assertIsNone(validate_deepmd_config(config))

    def test_invalid_model_version(self):
        """
        Tests if the function raises a ValueError for an invalid model version.
        """
        config = {
            "deepmd_model_version": 1.5,
        }
        with self.assertRaises(ValueError):
            validate_deepmd_config(config)


class TestGenerateTrainingJson(unittest.TestCase):
    """
    Test case for the 'generate_training_json' function.

    Methods
    -------
    test_valid_user_input():
        Tests if the function correctly validates a valid configuration.
    test_invalid_key():
        Tests if the function raises a ValueError for an invalid key in user input.
    test_type_mismatch():
        Tests if the function raises a TypeError for a type mismatch in user input.
    test_use_previous_json():
        Tests if the function correctly updates training JSON with previous JSON.
    """

    def setUp(self):
        self.default_input_json = {
            "user_machine_keyword_train": "default",
            "user_machine_keyword_freeze": "default",
            "user_machine_keyword_compress": "default",
            "job_email": "default@example.com",
            "use_initial_datasets": True,
            "use_extra_datasets": False,
            "deepmd_model_version": "1.0",
            "start_lr": 0.001,
            "stop_lr": 0.0001,
            "decay_rate": 0.9,
            "decay_steps": 100,
            "decay_steps_fixed": True,
            "numb_steps": 1000,
            "numb_test": 100,
            "job_walltime_train_h": -1,
            "mean_s_per_step": 0.1,
        }

    def test_valid_user_input(self):
        """
        Tests if the function correctly validates a valid configuration.
        """
        user_input = {
            "user_machine_keyword_train": "custom",
            "job_email": "user@example.com",
            "mean_s_per_step": 0.5,
        }
        previous_json = {}
        expected_training_json = {
            "user_machine_keyword_train": "custom",
            "user_machine_keyword_freeze": "default",
            "user_machine_keyword_compress": "default",
            "job_email": "user@example.com",
            "use_initial_datasets": True,
            "use_extra_datasets": False,
            "deepmd_model_version": "1.0",
            "start_lr": 0.001,
            "stop_lr": 0.0001,
            "decay_rate": 0.9,
            "decay_steps": 100,
            "decay_steps_fixed": True,
            "numb_steps": 1000,
            "numb_test": 100,
            "job_walltime_train_h": -1,
            "mean_s_per_step": 0.5,
        }
        expected_merged_json = {
            "user_machine_keyword_train": "custom",
            "user_machine_keyword_freeze": "default",
            "user_machine_keyword_compress": "default",
            "job_email": "user@example.com",
            "use_initial_datasets": True,
            "use_extra_datasets": False,
            "deepmd_model_version": "1.0",
            "start_lr": 0.001,
            "stop_lr": 0.0001,
            "decay_rate": 0.9,
            "decay_steps": 100,
            "decay_steps_fixed": True,
            "numb_steps": 1000,
            "numb_test": 100,
            "job_walltime_train_h": -1,
            "mean_s_per_step": 0.5,
        }
        training_json, updated_merged_json = generate_training_json(user_input, previous_json, self.default_input_json)
        self.assertDictEqual(training_json, expected_training_json)
        self.assertDictEqual(updated_merged_json, expected_merged_json)

    def test_invalid_key(self):
        """
        Tests if the function raises a ValueError for an invalid key in user input.
        """
        user_input = {"numb_steps": "dos"}
        previous_json = {}
        default_input_json = {"numb_steps": 1000}
        with self.assertRaises(TypeError):
            generate_training_json(user_input, previous_json, default_input_json)

    def test_type_mismatch(self):
        """
        Tests if the function raises a TypeError for a type mismatch in user input.
        """
        user_input = {"numb_steps": "invalid_type"}
        previous_json = {}

        with self.assertRaises(TypeError):
            generate_training_json(user_input, previous_json, self.default_input_json)

    def test_use_previous_json(self):
        """
        Tests if the function correctly updates training JSON with previous JSON.
        """
        user_input = {}
        previous_json = {
            "user_machine_keyword_train": "previous",
            "user_machine_keyword_freeze": "previous",
            "user_machine_keyword_compress": "previous",
            "job_email": "previous@example.com",
            "job_walltime_train_h": -1,
            "mean_s_per_step": 0.3,
        }

        expected_training_json = {
            "user_machine_keyword_train": "previous",
            "user_machine_keyword_freeze": "previous",
            "user_machine_keyword_compress": "previous",
            "job_email": "previous@example.com",
            "use_initial_datasets": True,
            "use_extra_datasets": False,
            "deepmd_model_version": "1.0",
            "start_lr": 0.001,
            "stop_lr": 0.0001,
            "decay_rate": 0.9,
            "decay_steps": 100,
            "decay_steps_fixed": True,
            "numb_steps": 1000,
            "numb_test": 100,
            "job_walltime_train_h": -1,
            "mean_s_per_step": 0.3,
        }

        training_json, updated_merged_json = generate_training_json(user_input, previous_json, self.default_input_json)
        self.assertDictEqual(training_json, expected_training_json)
        self.assertDictEqual(updated_merged_json, expected_training_json)


if __name__ == "__main__":
    unittest.main()
