from pathlib import Path

# Unittest imports
import unittest
import tempfile

# Non-standard library imports
import numpy as np

# deepmd_iterative imports
from deepmd_iterative.common.xyz import (
    read_xyz_trajectory,
    write_xyz_frame_to_file,
)


class TestReadXYZTrajectory(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and file with a constant number of atoms
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.file_oneframe_path = Path(self.tmp_dir.name) / "test_oneframe.xyz"
        with self.file_oneframe_path.open("w") as f:
            f.write("4\n")
            f.write("Test molecule\n")
            f.write("C 0.0 0.0 0.0\n")
            f.write("C 1.2 0.0 0.0\n")
            f.write("C 0.0 1.2 0.0\n")
            f.write("C 0.0 0.0 1.2\n")

        self.file_path = Path(self.tmp_dir.name) / "test.xyz"
        with self.file_path.open("w") as f:
            f.write("4\n")
            f.write("Test molecule frame 1\n")
            f.write("C 0.0 0.0 0.0\n")
            f.write("C 1.2 0.0 0.0\n")
            f.write("C 0.0 1.2 0.0\n")
            f.write("C 0.0 0.0 1.2\n")
            f.write("4\n")
            f.write("Test molecule frame 2\n")
            f.write("C 1.0 1.0 1.0\n")
            f.write("C 2.2 1.0 1.0\n")
            f.write("C 1.0 2.2 1.0\n")
            f.write("C 1.0 1.0 2.2\n")

        self.file_var_path = Path(self.tmp_dir.name) / "test_var.xyz"
        with self.file_var_path.open("w") as f:
            f.write("4\n")
            f.write("Test molecule frame 1\n")
            f.write("C 0.0 0.0 0.0\n")
            f.write("C 1.2 0.0 0.0\n")
            f.write("C 0.0 1.2 0.0\n")
            f.write("C 0.0 0.0 1.2\n")
            f.write("3\n")
            f.write("Test molecule frame 2\n")
            f.write("C 1.0 1.0 1.0\n")
            f.write("C 2.2 1.0 1.0\n")
            f.write("C 1.0 2.2 1.0\n")
            f.write("C 1.0 1.0 2.2\n")

        self.file_miss_path = Path(self.tmp_dir.name) / "missing.xyz"

        self.file_inc_path = Path(self.tmp_dir.name) / "incorrect.xyz"
        with self.file_inc_path.open("w") as f:
            f.write("4\n")
            f.write("Test molecule frame 1\n")
            f.write("C 0.0 0.0 0.0\n")
            f.write("C 1.2 0.0 0.0\n")
            f.write("C 0.0 1.2 0.0\n")
            f.write("C 0.0 0.0 1.2\n")
            f.write("4\n")
            f.write("Test molecule frame 2\n")
            f.write("C 1.0 1.0 1.0\n")
            f.write("C 2.2 1.0 1.0\n")
            f.write("C 1.0 2.2 1.0\n")

    def tearDown(self):
        # Clean up the temporary directory and file
        self.tmp_dir.cleanup()

    def test_read_xyz_trajectory_oneframe(self):
        # Test reading the file with the function
        num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(
            self.file_oneframe_path
        )

        # Check the output types and shapes
        self.assertIsInstance(num_atoms, np.ndarray)
        self.assertIsInstance(atom_symbols, np.ndarray)
        self.assertIsInstance(atom_coords, np.ndarray)
        self.assertEqual(num_atoms.shape, (1,))
        self.assertEqual(atom_symbols.shape, (1, 4))
        self.assertEqual(atom_coords.shape, (1, 4, 3))

        # Check the output values
        self.assertEqual(num_atoms[0], 4)
        self.assertListEqual(atom_symbols[0].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[0],
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2]],
            )
        )

    def test_read_xyz_trajectory(self):
        # Test reading the file with the function
        num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(self.file_path)

        # Check the output types and shapes
        self.assertIsInstance(num_atoms, np.ndarray)
        self.assertIsInstance(atom_symbols, np.ndarray)
        self.assertIsInstance(atom_coords, np.ndarray)
        self.assertEqual(num_atoms.shape, (2,))
        self.assertEqual(atom_symbols.shape, (2, 4))
        self.assertEqual(atom_coords.shape, (2, 4, 3))

        # Check the output values for frame 1
        self.assertEqual(num_atoms[0], 4)
        self.assertListEqual(atom_symbols[0].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[0],
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2]],
            )
        )

        # Check the output values for frame 2
        self.assertEqual(num_atoms[1], 4)
        self.assertListEqual(atom_symbols[1].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[1],
                [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [1.0, 2.2, 1.0], [1.0, 1.0, 2.2]],
            )
        )

    def test_read_xyz_trajectory_variable(self):
        # Test reading the file with the function
        with self.assertRaises(SystemExit) as cm:
            num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(
                self.file_var_path
            )
        self.assertEqual(cm.exception.code, 1)

    def test_read_xyz_trajectory_missing(self):
        # Test reading a missing file with the function
        with self.assertRaises(SystemExit) as cm:
            num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(
                self.file_miss_path
            )
        self.assertEqual(cm.exception.code, 2)

    def test_read_xyz_trajectory_incorrect(self):
        # Test reading the file with the function
        with self.assertRaises(SystemExit) as cm:
            num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(
                self.file_inc_path
            )
        self.assertEqual(cm.exception.code, 1)


class TestReadWriteXYZTrajectory(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and file with a constant number of atoms
        self.tmp_dir = tempfile.TemporaryDirectory()

        self.file_path = Path(self.tmp_dir.name) / "test.xyz"
        with self.file_path.open("w") as f:
            f.write("4\n")
            f.write("Test molecule frame 1\n")
            f.write("C 0.0 0.0 0.0\n")
            f.write("C 1.2 0.0 0.0\n")
            f.write("C 0.0 1.2 0.0\n")
            f.write("C 0.0 0.0 1.2\n")
            f.write("4\n")
            f.write("Test molecule frame 2\n")
            f.write("C 1.0 1.0 1.0\n")
            f.write("C 2.2 1.0 1.0\n")
            f.write("C 1.0 2.2 1.0\n")
            f.write("C 1.0 1.0 2.2\n")

    def tearDown(self):
        # Clean up the temporary directory and file
        self.tmp_dir.cleanup()

    def test_read_xyz_trajectory(self):
        # Test reading the file with the function
        num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(self.file_path)

        # Check the output types and shapes
        self.assertIsInstance(num_atoms, np.ndarray)
        self.assertIsInstance(atom_symbols, np.ndarray)
        self.assertIsInstance(atom_coords, np.ndarray)
        self.assertEqual(num_atoms.shape, (2,))
        self.assertEqual(atom_symbols.shape, (2, 4))
        self.assertEqual(atom_coords.shape, (2, 4, 3))

        # Check the output values for frame 1
        self.assertEqual(num_atoms[0], 4)
        self.assertListEqual(atom_symbols[0].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[0],
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2]],
            )
        )

        # Check the output values for frame 2
        self.assertEqual(num_atoms[1], 4)
        self.assertListEqual(atom_symbols[1].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[1],
                [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [1.0, 2.2, 1.0], [1.0, 1.0, 2.2]],
            )
        )

        self.file_new_path = Path(self.tmp_dir.name) / "new.xyz"
        write_xyz_frame_to_file(
            self.file_new_path, 0, num_atoms, atom_coords, atom_symbols
        )
        num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(self.file_new_path)

        # Check the output types and shapes
        self.assertIsInstance(num_atoms, np.ndarray)
        self.assertIsInstance(atom_symbols, np.ndarray)
        self.assertIsInstance(atom_coords, np.ndarray)
        self.assertEqual(num_atoms.shape, (1,))
        self.assertEqual(atom_symbols.shape, (1, 4))
        self.assertEqual(atom_coords.shape, (1, 4, 3))

        # Check the output values for frame 1
        self.assertEqual(num_atoms[0], 4)
        self.assertListEqual(atom_symbols[0].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[0],
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2]],
            )
        )


class TestWriteXYZFrameToFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "test.xyz"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_xyz_frame_to_file(self):
        # Test input data
        frame_idx = 0
        num_atoms = np.array([2, 2, 2])
        atom_coords = np.array(
            [[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]], [[0, 0, 0], [1, 1, 1]]]
        )
        atom_symbols = np.array([["C", "H"], ["C", "H"], ["N", "O"]])
        expected_output = "2\nFrame index: 0\nC 0.000000 0.000000 0.000000\nH 1.000000 1.000000 1.000000\n"
        # Call the function
        write_xyz_frame_to_file(
            self.temp_file, frame_idx, num_atoms, atom_coords, atom_symbols
        )

        # Read the output file
        with open(self.temp_file) as f:
            output = f.read()

        # Check the output
        self.assertEqual(output, expected_output)

    def test_write_xyz_frame_to_file_frame_idx_out_of_range(self):
        # Test input data with frame index out of range
        frame_idx = 3
        num_atoms = np.array([2, 3, 2])
        atom_coords = np.array(
            [[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]], [[0, 0, 0], [1, 1, 1]]]
        )
        atom_symbols = np.array([["C", "H"], ["C", "H"], ["N", "O"]])

        # Call the function and check for the expected error message
        with self.assertRaises(SystemExit) as cm:
            write_xyz_frame_to_file(
                self.temp_file, frame_idx, num_atoms, atom_coords, atom_symbols
            )
        self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
