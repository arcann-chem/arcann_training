"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2026 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2026/02/02

Unit tests for the xyz module.

Classes
-------
TestParseExtendedFormat():
    Test case for the 'parse_extended_format' function.
TestParseXyzTrajectoryFile():
    Test case for the 'parse_xyz_trajectory_file' function.
TestWriteXyzFrame():
    Test case for the 'write_xyz_frame' function.
"""

# Standard library modules
import tempfile
import unittest
from pathlib import Path

# Third-party modules
import numpy as np

# Local imports
from arcann_training.common.xyz import (
    parse_extended_format,
    parse_xyz_trajectory_file,
    write_xyz_frame,
)


class TestParseExtendedFormat(unittest.TestCase):
    """
    Test case for the 'parse_extended_format' function.

    Methods
    -------
    test_parse_all_fields():
        Test parsing a comment line with lattice, properties, pbc, and max_f_std.
    test_parse_missing_fields():
        Test parsing a comment line with no extended info.
    test_parse_boolean_variants():
        Test parsing mixed-case boolean PBC values.
    test_parse_invalid_max_f_std():
        Test rejecting non-numeric max_f_std values.
    """

    def test_parse_all_fields(self):
        """
        Test parsing a comment line with lattice, properties, pbc, and max_f_std.
        """
        comment = (
            'Lattice="1 0 0 0 1 0 0 0 1" '
            'Properties=species:S:1:pos:R:3 pbc="T F true" max_f_std=0.25'
        )
        lattice, properties, pbc, max_f_std = parse_extended_format(comment)
        self.assertEqual(lattice, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertTrue(properties)
        self.assertEqual(pbc, [True, False, True])
        self.assertEqual(max_f_std, 0.25)

    def test_parse_missing_fields(self):
        """
        Test parsing a comment line with no extended info.
        """
        comment = "no extended fields here"
        lattice, properties, pbc, max_f_std = parse_extended_format(comment)
        self.assertIsNone(lattice)
        self.assertFalse(properties)
        self.assertIsNone(pbc)
        self.assertIsNone(max_f_std)

    def test_parse_boolean_variants(self):
        """
        Test parsing mixed-case boolean PBC values.
        """
        comment = 'pbc="t F TRUE"'
        lattice, properties, pbc, max_f_std = parse_extended_format(comment)
        self.assertIsNone(lattice)
        self.assertFalse(properties)
        self.assertEqual(pbc, [True, False, True])
        self.assertIsNone(max_f_std)

    def test_parse_invalid_max_f_std(self):
        """
        Test ignoring non-numeric max_f_std values.
        """
        lattice, properties, pbc, max_f_std = parse_extended_format(
            "max_f_std=not_a_number"
        )
        self.assertIsNone(lattice)
        self.assertFalse(properties)
        self.assertIsNone(pbc)
        self.assertIsNone(max_f_std)


class TestParseXyzTrajectoryFile(unittest.TestCase):
    """
    Test case for the 'parse_xyz_trajectory_file' function.

    Methods
    -------
    test_parse_simple_two_frames():
        Test parsing a two-frame XYZ file with constant atom counts.
    test_non_digit_atom_count():
        Test rejecting non-integer atom counts.
    test_incorrect_line_length():
        Test rejecting atom lines that are not 4 fields.
    test_inconsistent_atom_counts():
        Test rejecting trajectories with changing atom counts.
    test_invalid_pbc_length():
        Test rejecting PBC lines that are not three booleans.
    test_missing_file():
        Test raising when the file does not exist.
    """

    def test_parse_simple_two_frames(self):
        """
        Test parsing a two-frame XYZ file with constant atom counts.
        """
        xyz_text = "\n".join(
            [
                "2",
                'Lattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3',
                "H 0.0 0.0 0.0",
                "O 0.0 0.0 1.0",
                "2",
                "frame 2 comment",
                "H 1.0 0.0 0.0",
                "O 1.0 0.0 1.0",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "traj.xyz"
            xyz_file.write_text(xyz_text)

            (
                atom_counts,
                atomic_symbols,
                atomic_coordinates,
                comments,
                lattice_info,
                pbc_info,
                properties_info,
                max_f_std_info,
            ) = parse_xyz_trajectory_file(xyz_file)

        np.testing.assert_array_equal(atom_counts, np.array([2, 2]))
        self.assertEqual(
            comments,
            [
                'Lattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3',
                "frame 2 comment",
            ],
        )
        self.assertEqual(atomic_symbols.shape, (2, 2))
        np.testing.assert_array_equal(atomic_symbols[0], np.array(["H", "O"]))
        np.testing.assert_array_equal(
            atomic_coordinates[1], np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        )
        self.assertIsNotNone(lattice_info[0])
        self.assertIsNone(lattice_info[1])
        self.assertIsNone(pbc_info[0])
        self.assertIsNone(pbc_info[1])
        self.assertTrue(properties_info[0])
        self.assertIsNone(properties_info[1])
        self.assertIsNone(max_f_std_info[0])

    def test_non_digit_atom_count(self):
        """
        Test rejecting non-integer atom counts.
        """
        xyz_text = "\n".join(
            [
                "two",
                "comment",
                "H 0.0 0.0 0.0",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "bad.xyz"
            xyz_file.write_text(xyz_text)
            with self.assertRaises(TypeError):
                parse_xyz_trajectory_file(xyz_file)

    def test_incorrect_line_length(self):
        """
        Test rejecting atom lines that are not 4 fields.
        """
        xyz_text = "\n".join(
            [
                "1",
                "comment",
                "H 0.0 0.0",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "bad.xyz"
            xyz_file.write_text(xyz_text)
            with self.assertRaises(ValueError):
                parse_xyz_trajectory_file(xyz_file)

    def test_inconsistent_atom_counts(self):
        """
        Test rejecting trajectories with changing atom counts.
        """
        xyz_text = "\n".join(
            [
                "1",
                "frame 1",
                "H 0.0 0.0 0.0",
                "2",
                "frame 2",
                "H 0.0 0.0 0.0",
                "O 0.0 0.0 1.0",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "bad.xyz"
            xyz_file.write_text(xyz_text)
            with self.assertRaises(ValueError):
                parse_xyz_trajectory_file(xyz_file)

    def test_invalid_pbc_length(self):
        """
        Test rejecting PBC lines that are not three booleans.
        """
        xyz_text = "\n".join(
            [
                "1",
                'pbc="T F"',
                "H 0.0 0.0 0.0",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "bad.xyz"
            xyz_file.write_text(xyz_text)
            with self.assertRaises(ValueError):
                parse_xyz_trajectory_file(xyz_file)

    def test_missing_file(self):
        """
        Test raising when the file does not exist.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "missing.xyz"
            with self.assertRaises(FileNotFoundError):
                parse_xyz_trajectory_file(xyz_file)


class TestWriteXyzFrame(unittest.TestCase):
    """
    Test case for the 'write_xyz_frame' function.

    Methods
    -------
    test_write_single_frame_with_lattice():
        Test writing a single frame with lattice info and verify contents.
    test_write_single_frame_without_lattice():
        Test writing a single frame without lattice info and verify comment line.
    test_write_frame_out_of_range():
        Test rejecting invalid frame indices.
    test_write_read_round_trip():
        Test writing a frame then parse it back.
    """

    def test_write_single_frame_with_lattice(self):
        """
        Test writing a single frame with lattice info and verify contents.
        """
        atom_counts = np.array([2])
        atomic_symbols = np.array([["H", "O"]])
        atomic_coordinates = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        cell_info = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
        comments = ["ignored when cell_info is present"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "frame.xyz"
            write_xyz_frame(
                out_file,
                frame_idx=0,
                atom_counts=atom_counts,
                atomic_symbols=atomic_symbols,
                atomic_coordinates=atomic_coordinates,
                cell_info=cell_info,
                comments=comments,
            )
            lines = out_file.read_text().splitlines()

        self.assertEqual(lines[0], "2")
        self.assertTrue(
            lines[1].startswith('Lattice="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0"')
        )
        self.assertIn("Properties=species:S:1:pos:R:3", lines[1])
        self.assertEqual(lines[2], "H 0.000000 0.000000 0.000000")
        self.assertEqual(lines[3], "O 0.000000 0.000000 1.000000")

    def test_write_single_frame_without_lattice(self):
        """
        Test writing a single frame without lattice info and verify comment line.
        """
        atom_counts = np.array([1])
        atomic_symbols = np.array([["He"]])
        atomic_coordinates = np.array([[[1.5, 2.5, 3.5]]])
        cell_info = np.array([])
        comments = ["frame comment"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "frame.xyz"
            write_xyz_frame(
                out_file,
                frame_idx=0,
                atom_counts=atom_counts,
                atomic_symbols=atomic_symbols,
                atomic_coordinates=atomic_coordinates,
                cell_info=cell_info,
                comments=comments,
            )
            lines = out_file.read_text().splitlines()

        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[1], "frame comment")
        self.assertEqual(lines[2], "He 1.500000 2.500000 3.500000")

    def test_write_frame_out_of_range(self):
        """
        Test rejecting invalid frame indices.
        """
        atom_counts = np.array([1])
        atomic_symbols = np.array([["H"]])
        atomic_coordinates = np.array([[[0.0, 0.0, 0.0]]])
        cell_info = np.array([])
        comments = ["comment"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "frame.xyz"
            with self.assertRaises(IndexError):
                write_xyz_frame(
                    out_file,
                    frame_idx=1,
                    atom_counts=atom_counts,
                    atomic_symbols=atomic_symbols,
                    atomic_coordinates=atomic_coordinates,
                    cell_info=cell_info,
                    comments=comments,
                )

    def test_write_read_round_trip(self):
        """
        Test writing a frame then parse it back.
        """
        atom_counts = np.array([2])
        atomic_symbols = np.array([["C", "O"]])
        atomic_coordinates = np.array([[[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]])
        cell_info = np.array([])
        comments = ["round trip"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "frame.xyz"
            write_xyz_frame(
                out_file,
                frame_idx=0,
                atom_counts=atom_counts,
                atomic_symbols=atomic_symbols,
                atomic_coordinates=atomic_coordinates,
                cell_info=cell_info,
                comments=comments,
            )

            (
                atom_counts_out,
                atomic_symbols_out,
                atomic_coordinates_out,
                comments_out,
                lattice_info_out,
                pbc_info_out,
                properties_info_out,
                max_f_std_info_out,
            ) = parse_xyz_trajectory_file(out_file)

        np.testing.assert_array_equal(atom_counts_out, atom_counts)
        np.testing.assert_array_equal(atomic_symbols_out, atomic_symbols)
        np.testing.assert_array_equal(atomic_coordinates_out, atomic_coordinates)
        self.assertEqual(comments_out, ["round trip"])
        self.assertIsNone(lattice_info_out[0])
        self.assertIsNone(pbc_info_out[0])
        self.assertIsNone(properties_info_out[0])
        self.assertIsNone(max_f_std_info_out[0])


if __name__ == "__main__":
    unittest.main()
