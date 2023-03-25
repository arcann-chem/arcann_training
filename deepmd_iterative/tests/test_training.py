from pathlib import Path
import unittest
import tempfile
import os
import shutil

# Non-standard imports
import numpy as np

# deepmd_iterative imports
from deepmd_iterative.common.training import (
    calculate_decay_steps,
    calculate_decay_rate,
    calculate_learning_rate,
    check_initial_datasets,
)


class TestCalculateDecaySteps(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Delete the temporary directory and its contents
        self.temp_dir.cleanup()

    def test_calculate_decay_steps_positive_input(self):
        # Test for positive input values
        self.assertEqual(calculate_decay_steps(20000), 5000)
        self.assertEqual(calculate_decay_steps(50000), 12500)
        self.assertEqual(calculate_decay_steps(60000), 15000)
        self.assertEqual(calculate_decay_steps(100000), 25000)
        self.assertEqual(calculate_decay_steps(150000), 30000)

    def test_calculate_decay_steps_invalid_input(self):
        # Test for invalid input values
        with self.assertRaises(SystemExit) as cm:
            calculate_decay_steps(0)
        self.assertEqual(cm.exception.code, 1)
        with self.assertRaises(SystemExit) as cm:
            calculate_decay_steps(-100)
        self.assertEqual(cm.exception.code, 1)
        with self.assertRaises(SystemExit) as cm:
            calculate_decay_steps(100, min_decay_steps=-500)
        self.assertEqual(cm.exception.code, 1)

    def test_calculate_decay_steps_output_type(self):
        # Test for the correct output type
        self.assertIsInstance(calculate_decay_steps(20000), int)

    def test_calculate_decay_steps_temp_directory(self):
        # Test that the function does not create or modify any files outside of the temp directory
        initial_files = os.listdir(".")
        calculate_decay_steps(20000)
        final_files = os.listdir(".")
        self.assertListEqual(initial_files, final_files)


class TestCalculateDecayRate(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Delete the temporary directory and its contents
        self.temp_dir.cleanup()

    def test_calculate_decay_rate_valid_input(self):
        # Test for valid input values
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
        # Test for invalid input values
        with self.assertRaises(SystemExit) as cm:
            calculate_decay_rate(100, -0.01, 0.001, 5000)
        self.assertEqual(cm.exception.code, 1)
        with self.assertRaises(SystemExit) as cm:
            calculate_decay_rate(100, 0.01, 0.001, 0)
        self.assertEqual(cm.exception.code, 1)

    def test_calculate_decay_rate_output_type(self):
        # Test that the output is of type float
        self.assertIsInstance(calculate_decay_rate(100, 0.01, 0.001, 5000), float)

    def test_calculate_decay_rate_temp_directory(self):
        # Test that the function does not create or modify any files outside of the temp directory
        initial_files = os.listdir(".")
        calculate_decay_rate(50000, 0.01, 0.001, 5000)
        final_files = os.listdir(".")
        self.assertListEqual(initial_files, final_files)


class TestCalculateLearningRate(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Delete the temporary directory and its contents
        self.temp_dir.cleanup()

    def test_calculate_learning_rate_valid_input(self):
        # Test for valid input values
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
        # Test for invalid input values
        with self.assertRaises(SystemExit) as cm:
            calculate_learning_rate(-100, 0.01, 0.1, 5000)
        self.assertEqual(cm.exception.code, 1)
        with self.assertRaises(SystemExit) as cm:
            calculate_learning_rate(100, -0.01, 0.1, 5000)
        self.assertEqual(cm.exception.code, 1)
        with self.assertRaises(SystemExit) as cm:
            calculate_learning_rate(100, 0.01, -0.1, 5000)
        self.assertEqual(cm.exception.code, 1)
        with self.assertRaises(SystemExit) as cm:
            calculate_learning_rate(100, 0.01, 0.1, -5000)
        self.assertEqual(cm.exception.code, 1)

    def test_calculate_learning_rate_output_type(self):
        # Test for the correct output type
        self.assertIsInstance(calculate_learning_rate(100, 0.01, 0.1, 5000), float)

    def test_calculate_learning_rate_temp_directory(self):
        # Test that the function does not create or modify any files outside of the temp directory
        initial_files = os.listdir(".")
        calculate_learning_rate(100, 0.01, 0.1, 5000)
        final_files = os.listdir(".")
        self.assertListEqual(initial_files, final_files)


class TestCheckInitialDatasets(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        # Create 'data' and 'control' subfolders inside the temporary directory
        os.makedirs(os.path.join(self.temp_dir.name, "data"))
        os.makedirs(os.path.join(self.temp_dir.name, "control"))
        # Create a sample 'initial_datasets.json' file in the 'control' subfolder
        with open(
            os.path.join(self.temp_dir.name, "control", "initial_datasets.json"), "w"
        ) as file:
            file.write('{"dataset1": 100, "dataset2": 200}')
        # Create sample dataset folders in the 'data' subfolder
        os.makedirs(os.path.join(self.temp_dir.name, "data", "dataset1", "set.000"))
        os.makedirs(os.path.join(self.temp_dir.name, "data", "dataset2", "set.000"))
        # Create sample box.npy files for each dataset
        np.save(
            os.path.join(self.temp_dir.name, "data", "dataset1", "set.000", "box.npy"),
            np.zeros(100),
        )
        np.save(
            os.path.join(self.temp_dir.name, "data", "dataset2", "set.000", "box.npy"),
            np.zeros(200),
        )

    def tearDown(self):
        # Delete the temporary directory and its contents
        self.temp_dir.cleanup()

    def test_check_initial_datasets(self):
        # Test the function with valid inputs
        expected_result = {"dataset1": 100, "dataset2": 200}
        self.assertDictEqual(
            check_initial_datasets(Path(self.temp_dir.name)), expected_result
        )

    def test_check_initial_datasets_missing_json(self):
        # Test the function when the initial_datasets.json file is missing
        os.remove(os.path.join(self.temp_dir.name, "control", "initial_datasets.json"))
        with self.assertRaises(SystemExit) as cm:
            check_initial_datasets(Path(self.temp_dir.name))
        self.assertEqual(cm.exception.code, 2)

    def test_check_initial_datasets_missing_dataset(self):
        # Test the function when one of the initial datasets is missing
        shutil.rmtree(os.path.join(self.temp_dir.name, "data", "dataset2"))
        with self.assertRaises(SystemExit) as cm:
            check_initial_datasets(Path(self.temp_dir.name))
        self.assertEqual(cm.exception.code, 2)

    def test_check_initial_datasets_invalid_num_samples(self):
        # Test the function when the number of samples in one of the initial datasets is incorrect
        np.save(
            os.path.join(self.temp_dir.name, "data", "dataset1", "set.000", "box.npy"),
            np.zeros(50),
        )
        with self.assertRaises(SystemExit) as cm:
            check_initial_datasets(Path(self.temp_dir.name))
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
