"""
Created: 2023/01/01
Last modified: 2023/03/27

Test cases for the xyz module.

Class
-----
TestReadXYZTrajectory
    Test case for the read_xyz_trajectory() function.

TestWriteXYZFrameToFile
    Test case for the write_xyz_frame() function.

TestReadWriteXYZTrajectory
    Test case for combined use of read_xyz_trajectory() and write_xyz_frame() functions.
"""
# Standard library modules
import unittest
import tempfile
from pathlib import Path


# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.common.xyz import (
    read_xyz_trajectory,
    write_xyz_frame,
)


class TestReadXYZTrajectory(unittest.TestCase):
    """
    Test case for the read_xyz_trajectory() function.

    Methods
    -------
    test_read_xyz_trajectory_oneframe():
        Test that the function returns the expected arrays for a single frame XYZ file.
    test_read_xyz_trajectory():
        Test that the function returns the expected arrays for a multi-frame XYZ file.
    test_read_xyz_trajectory_variable():
        Test that the function raises a TypeError when the number of atoms is not an integer.
    test_read_xyz_trajectory_missing():
        Test that the function raises a FileNotFoundError when the input file is missing.
    test_read_xyz_trajectory_incorrect():
        Test that the function raises an IndexError when the input file has an incorrect format.
    """

    def setUp(self):
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
        self.tmp_dir.cleanup()

    def test_read_xyz_trajectory_oneframe(self):
        num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(
            self.file_oneframe_path
        )

        self.assertIsInstance(num_atoms, np.ndarray)
        self.assertIsInstance(atom_symbols, np.ndarray)
        self.assertIsInstance(atom_coords, np.ndarray)
        self.assertEqual(num_atoms.shape, (1,))
        self.assertEqual(atom_symbols.shape, (1, 4))
        self.assertEqual(atom_coords.shape, (1, 4, 3))

        self.assertEqual(num_atoms[0], 4)
        self.assertListEqual(atom_symbols[0].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[0],
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2]],
            )
        )

    def test_read_xyz_trajectory(self):
        num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(self.file_path)

        self.assertIsInstance(num_atoms, np.ndarray)
        self.assertIsInstance(atom_symbols, np.ndarray)
        self.assertIsInstance(atom_coords, np.ndarray)
        self.assertEqual(num_atoms.shape, (2,))
        self.assertEqual(atom_symbols.shape, (2, 4))
        self.assertEqual(atom_coords.shape, (2, 4, 3))

        self.assertEqual(num_atoms[0], 4)
        self.assertListEqual(atom_symbols[0].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[0],
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2]],
            )
        )

        self.assertEqual(num_atoms[1], 4)
        self.assertListEqual(atom_symbols[1].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[1],
                [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [1.0, 2.2, 1.0], [1.0, 1.0, 2.2]],
            )
        )

    def test_read_xyz_trajectory_variable(self):
        with self.assertRaises(TypeError) as cm:
            num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(
                self.file_var_path
            )
        error_msg = str(cm.exception)
        expected_error_msg = (
            f"Incorrect file format: number of atoms must be an integer."
        )
        self.assertEqual(error_msg, expected_error_msg)

    def test_read_xyz_trajectory_missing(self):
        with self.assertRaises(FileNotFoundError) as cm:
            num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(
                self.file_miss_path
            )
        error_msg = str(cm.exception)
        expected_error_msg = f"File not found {self.file_miss_path.name} not in {self.file_miss_path.parent}"
        self.assertEqual(error_msg, expected_error_msg)

    def test_read_xyz_trajectory_incorrect(self):
        with self.assertRaises(IndexError) as cm:
            num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(
                self.file_inc_path
            )
        error_msg = str(cm.exception)
        expected_error_msg = f"Incorrect file format: end of file reached prematurely."
        self.assertEqual(error_msg, expected_error_msg)


class TestWriteXYZFrameToFile(unittest.TestCase):
    """
    Test case for the write_xyz_frame() function.

    Methods
    -------
    test_write_xyz_frame():
        Test writing the XYZ coordinates of a specific frame of a trajectory to a file.
    test_write_xyz_frame_frame_idx_out_of_range():
        Test that an IndexError is raised when the frame index is out of range.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "test.xyz"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_xyz_frame(self):
        frame_idx = 0
        num_atoms = np.array([2, 2, 2])
        atom_coords = np.array(
            [[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]], [[0, 0, 0], [1, 1, 1]]]
        )
        atom_symbols = np.array([["C", "H"], ["C", "H"], ["N", "O"]])
        expected_output = "2\nFrame index: 0\nC 0.000000 0.000000 0.000000\nH 1.000000 1.000000 1.000000\n"
        write_xyz_frame(self.temp_file, frame_idx, num_atoms, atom_coords, atom_symbols)

        with open(self.temp_file) as f:
            output = f.read()

        self.assertEqual(output, expected_output)

    def test_write_xyz_frame_frame_idx_out_of_range(self):
        frame_idx = 3
        num_atoms = np.array([2, 3, 2])
        atom_coords = np.array(
            [[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]], [[0, 0, 0], [1, 1, 1]]]
        )
        atom_symbols = np.array([["C", "H"], ["C", "H"], ["N", "O"]])

        with self.assertRaises(IndexError) as cm:
            write_xyz_frame(
                self.temp_file, frame_idx, num_atoms, atom_coords, atom_symbols
            )
        error_msg = str(cm.exception)
        expected_error_msg = f"Frame index out of range: {frame_idx} (number of frames: {num_atoms.size})"
        self.assertEqual(error_msg, expected_error_msg)


class TestReadWriteXYZTrajectory(unittest.TestCase):
    """
    Test case for combined use of read_xyz_trajectory() and write_xyz_frame() functions.

    Methods
    -------
    test_read_write_xyz_trajectory():
        Test reading and writing a trajectory in xyz format
    """

    def setUp(self):
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
        self.tmp_dir.cleanup()

    def test_read_write_xyz_trajectory(self):
        num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(self.file_path)

        self.assertIsInstance(num_atoms, np.ndarray)
        self.assertIsInstance(atom_symbols, np.ndarray)
        self.assertIsInstance(atom_coords, np.ndarray)
        self.assertEqual(num_atoms.shape, (2,))
        self.assertEqual(atom_symbols.shape, (2, 4))
        self.assertEqual(atom_coords.shape, (2, 4, 3))

        self.assertEqual(num_atoms[0], 4)
        self.assertListEqual(atom_symbols[0].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[0],
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2]],
            )
        )

        self.assertEqual(num_atoms[1], 4)
        self.assertListEqual(atom_symbols[1].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[1],
                [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [1.0, 2.2, 1.0], [1.0, 1.0, 2.2]],
            )
        )

        self.file_new_path = Path(self.tmp_dir.name) / "new.xyz"
        write_xyz_frame(self.file_new_path, 0, num_atoms, atom_coords, atom_symbols)
        num_atoms, atom_symbols, atom_coords = read_xyz_trajectory(self.file_new_path)

        self.assertIsInstance(num_atoms, np.ndarray)
        self.assertIsInstance(atom_symbols, np.ndarray)
        self.assertIsInstance(atom_coords, np.ndarray)
        self.assertEqual(num_atoms.shape, (1,))
        self.assertEqual(atom_symbols.shape, (1, 4))
        self.assertEqual(atom_coords.shape, (1, 4, 3))

        self.assertEqual(num_atoms[0], 4)
        self.assertListEqual(atom_symbols[0].tolist(), ["C", "C", "C", "C"])
        self.assertTrue(
            np.allclose(
                atom_coords[0],
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2]],
            )
        )


if __name__ == "__main__":
    unittest.main()
