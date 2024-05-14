"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/14

Test cases for the list module.

Classes
-------
TestChangeDirectory():
    Test case for the 'change_directory' function.

TestCheckDirectory():
    Test case for the 'check_directory' function.

TestCheckFileExistence():
    Test case for the 'check_file_existence' function.

TestRemoveFile():
    Test case for the 'remove_file' function.

TestRemoveFilesMatchingGlob():
    Test case for the 'remove_files_matching_glob' function.

TestRemoveTree():
    Unit test case for the 'remove_tree' function.

TestRemoveAllSymlink():
    Unit test case for the 'remove_all_symlink' function.
"""

# Standard library modules
import os
import tempfile
import unittest
from pathlib import Path

# Local imports
from arcann_training.common.filesystem import (
    change_directory,
    check_directory,
    check_file_existence,
    remove_file,
    remove_files_matching_glob,
    remove_tree,
    remove_all_symlink,
)


class TestChangeDirectory(unittest.TestCase):
    """
    Test case for the 'change_directory' function.

    Methods
    -------
    test_change_directory_existing_directory():
        Test changing to an existing directory.
    test_change_directory_nonexistent_directory():
        Test raising an error for a nonexistent directory.
    test_change_directory_file_not_directory():
        Test raising an error for a file instead of a directory.
    test_change_directory_error():
        Test raising an error if there is an error in changing the directory.
    test_change_directory_directory_with_space():
        Test changing to a directory with a space in the name.
    """

    def setUp(self):
        self.temp_dirs = [tempfile.TemporaryDirectory() for _ in range(3)]
        self.temp_dir_paths = [Path(_.name) for _ in self.temp_dirs]
        os.chmod(self.temp_dir_paths[2], 0o222)

    def tearDown(self):
        os.chmod(self.temp_dir_paths[2], 0o777)
        for temp_dir in self.temp_dirs:
            temp_dir.cleanup()

    def test_change_directory_existing_directory(self):
        """
        Test changing to an existing directory.
        """
        change_directory(self.temp_dir_paths[0])
        self.assertEqual(
            Path.cwd(),
            self.temp_dir_paths[0],
            msg="Directory not changed to the expected directory.",
        )

    def test_change_directory_nonexistent_directory(self):
        """
        Test raising an error for a nonexistent directory.
        """
        with self.assertRaises(FileNotFoundError, msg="No FileNotFoundError raised."):
            change_directory(Path("nonexistent_directory"))

    def test_change_directory_file_not_directory(self):
        """
        Test raising an error for a file instead of a directory.
        """
        temp_file = self.temp_dir_paths[0] / "temp_file.txt"
        with open(temp_file, "w") as f:
            f.write("This is a temporary file for testing purposes.")
        with self.assertRaises(FileNotFoundError, msg="No FileNotFoundError raised."):
            change_directory(temp_file)

    def test_change_directory_error(self):
        """
        Test raising an error if there is an error in changing the directory.
        """
        with self.assertRaises(OSError, msg="No OSError raised."):
            change_directory(self.temp_dir_paths[2])

    def test_change_directory_directory_with_space(self):
        """
        Test changing to a directory with a space in the name.
        """
        temp_dir_with_space = self.temp_dir_paths[1] / "directory with space"
        Path.mkdir(temp_dir_with_space)
        change_directory(temp_dir_with_space)
        self.assertEqual(
            Path.cwd(),
            temp_dir_with_space,
            msg="Directory not changed to the expected directory.",
        )


class TestCheckDirectory(unittest.TestCase):
    """
    Test case for the 'check_directory' function.

    Methods
    -------
    test_check_directory_existing_directory():
        Test checking an existing directory with 'abort_on_error=True'.
    test_check_directory_nonexistent_directory():
        Test raising a 'FileNotFoundError' for a nonexistent directory with 'abort_on_error=True'.
    test_check_directory_nonexistent_directory_no_abort():
        Test logging a warning for a nonexistent directory with 'abort_on_error=False'.
    test_check_directory_existing_directory_no_abort():
        Test checking an existing directory with 'abort_on_error=False'.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_directory_existing_directory(self):
        """
        Test checking an existing directory with 'abort_on_error=True'.
        """
        check_directory(Path(self.temp_dir.name))
        self.assertTrue(True)

    def test_check_directory_nonexistent_directory(self):
        """
        Test raising a 'FileNotFoundError' for a nonexistent directory with 'abort_on_error=True'.
        """
        with self.assertRaises(FileNotFoundError):
            check_directory(Path("nonexistent_directory"))

    def test_check_directory_nonexistent_directory_no_abort(self):
        """
        Test logging a warning for a nonexistent directory with 'abort_on_error=False'.
        """
        with self.assertLogs(level="WARNING"):
            check_directory(Path("nonexistent_directory"), abort_on_error=False)
        self.assertTrue(True)

    def test_check_directory_existing_directory_no_abort(self):
        """
        Test checking an existing directory with 'abort_on_error=False'.
        """
        check_directory(Path(self.temp_dir.name), abort_on_error=False)
        self.assertTrue(True)


class TestCheckFileExistence(unittest.TestCase):
    """
    Test case for the 'check_file_existence' function.

    Methods
    -------
    test_check_file_existence_existing_file():
        Test checking an existing file with 'expected_existence=True' and 'abort_on_error=True'.
    test_check_file_existence_nonexistent_file():
        Test raising a 'FileNotFoundError' for a nonexistent file with 'abort_on_error=True'.
    test_check_file_existence_nonexistent_file_no_abort():
        Test logging a warning for a nonexistent file with 'abort_on_error=False'.
    test_check_file_existence_existing_file_no_abort():
        Test raising a 'FileExistsError' for an existing file with 'expected_existence=False' and 'abort_on_error=True'.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "temp_file.txt"
        with open(self.temp_file, "w") as f:
            f.write("This is a temporary file for testing purposes.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_file_existence_existing_file(self):
        """
        Test checking an existing file with 'expected_existence=True' and 'abort_on_error=True'.
        """
        check_file_existence(self.temp_file)
        self.assertTrue(True)

    def test_check_file_existence_nonexistent_file(self):
        """
        Test raising a 'FileNotFoundError' for a nonexistent file with 'abort_on_error=True'.
        """
        with self.assertRaises(FileNotFoundError):
            check_file_existence(Path(self.temp_dir.name) / "nonexistent_file.txt")

    def test_check_file_existence_nonexistent_file_no_abort(self):
        """
        Test logging a warning for a nonexistent file with 'abort_on_error=False'.
        """
        with self.assertLogs(level="WARNING"):
            check_file_existence(Path(self.temp_dir.name) / "nonexistent_file.txt", abort_on_error=False)
        self.assertTrue(True)

    def test_check_file_existence_existing_file_no_abort(self):
        """
        Test raising a 'FileExistsError' for an existing file with 'expected_existence=False' and 'abort_on_error=True'.
        """
        with self.assertRaises(FileExistsError):
            check_file_existence(self.temp_file, expected_existence=False, abort_on_error=True)

        self.assertTrue(True)


class TestRemoveFile(unittest.TestCase):
    """
    Test case for the 'remove_file' function.

    Methods
    -------
    test_remove_existing_file():
        Test that the 'remove_file' function successfully removes an existing file.
    test_remove_nonexistent_file():
        Test that the 'remove_file' function does not raise an error when attempting to remove a nonexistent file.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "temp_file.txt"
        with open(self.temp_file, "w") as f:
            f.write("This is a temporary file for testing purposes.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_remove_existing_file(self):
        """
        Test that the 'remove_file' function successfully removes an existing file.
        """
        self.assertTrue(self.temp_file.is_file())
        remove_file(self.temp_file)
        self.assertFalse(self.temp_file.is_file())

    def test_remove_nonexistent_file(self):
        """
        Test that the 'remove_file' function does not raise an error when attempting to remove a nonexistent file.
        """
        self.assertFalse((Path(self.temp_dir.name) / "nonexistent_file.txt").is_file())
        remove_file(Path(self.temp_dir.name) / "nonexistent_file.txt")
        self.assertTrue(True)


class TestRemoveFilesMatchingGlob(unittest.TestCase):
    """
    Test case for the 'remove_files_matching_glob' function.

    Methods
    -------
    test_remove_files_matching_glob()
        Test removing files with a matching glob pattern.
    test_remove_files_matching_glob_nonexistent_directory()
        Test raising an error when the directory does not exist.
    test_remove_files_matching_glob_non_directory()
        Test raising an error when the directory path is not a directory.
    test_remove_files_no_matching_glob()
        Test not removing any files when no files match the glob pattern.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file_1 = Path(self.temp_dir.name) / "temp_file_1.txt"
        with open(self.temp_file_1, "w") as f:
            f.write("This is a temporary file for testing purposes.")
        self.temp_file_2 = Path(self.temp_dir.name) / "temp_file_2.txt"
        with open(self.temp_file_2, "w") as f:
            f.write("This is another temporary file for testing purposes.")
        self.temp_file_3 = Path(self.temp_dir.name) / "not_a_temp_file.npy"
        with open(self.temp_file_3, "w") as f:
            f.write("This file does not match the glob pattern.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_remove_files_matching_glob(self):
        """
        Test removing files with a matching glob pattern.
        """
        self.assertTrue(self.temp_file_1.is_file())
        self.assertTrue(self.temp_file_2.is_file())
        self.assertTrue(self.temp_file_3.is_file())
        remove_files_matching_glob(Path(self.temp_dir.name), "*.txt")
        self.assertFalse(self.temp_file_1.is_file())
        self.assertFalse(self.temp_file_2.is_file())
        self.assertTrue(self.temp_file_3.is_file())

    def test_remove_files_matching_glob_nonexistent_directory(self):
        """
        Test raising an error when the directory does not exist.
        """
        with self.assertRaises(NotADirectoryError):
            remove_files_matching_glob(Path("nonexistent_directory"), "*.toc")

    def test_remove_files_matching_glob_non_directory(self):
        """
        Test raising an error when the directory path is not a directory.
        """
        with self.assertRaises(NotADirectoryError):
            remove_files_matching_glob(self.temp_file_1, "*.txt")

    def test_remove_files_no_matching_glob(self):
        """
        Test not removing any files when no files match the glob pattern.
        """
        self.assertTrue(self.temp_file_1.is_file())
        self.assertTrue(self.temp_file_2.is_file())
        self.assertTrue(self.temp_file_3.is_file())
        remove_files_matching_glob(Path(self.temp_dir.name), "*.npz")
        self.assertTrue(self.temp_file_1.is_file())
        self.assertTrue(self.temp_file_2.is_file())
        self.assertTrue(self.temp_file_3.is_file())


class TestRemoveTree(unittest.TestCase):
    """
    Unit test case for the 'remove_tree' function.

    Methods
    -------
    test_remove_tree():
        Test removing an existing directory tree and its contents.
    test_remove_tree_nonexistent_directory():
        Test removing a nonexistent directory.
    test_remove_tree_file_not_a_directory():
        Test raising an error when the given path is not a directory.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_subdir = Path(self.temp_dir.name) / "subdir"
        self.temp_subsubdir = Path(self.temp_subdir) / "subdir"
        self.temp_subdir.mkdir()
        self.temp_subsubdir.mkdir()
        self.temp_file_1 = Path(self.temp_subdir) / "temp_file_1.txt"
        with open(self.temp_file_1, "w") as f:
            f.write("This is a temporary file for testing purposes.")
        self.temp_file_2 = Path(self.temp_subsubdir) / "temp_file_2.txt"
        with open(self.temp_file_2, "w") as f:
            f.write("This is another temporary file for testing purposes.")
        self.temp_file_3 = Path(self.temp_subsubdir) / "temp_file_3.txt"
        with open(self.temp_file_3, "w") as f:
            f.write("This is a third temporary file for testing purposes.")

    def test_remove_tree(self):
        """
        Test removing an existing directory tree and its contents.
        """
        self.assertTrue(self.temp_file_1.is_file())
        self.assertTrue(self.temp_subdir.is_dir())
        self.assertTrue(self.temp_subsubdir.is_dir())
        self.assertTrue(self.temp_file_2.is_file())
        self.assertTrue(self.temp_file_3.is_file())

        remove_tree(Path(self.temp_subdir))

        self.assertFalse(self.temp_file_1.is_file())
        self.assertFalse(self.temp_subsubdir.is_dir())
        self.assertFalse(self.temp_file_2.is_file())
        self.assertFalse(self.temp_file_3.is_file())

    def test_remove_tree_nonexistent_directory(self):
        """
        Test removing a nonexistent directory.
        """
        with self.assertRaises(FileNotFoundError):
            remove_tree(Path("nonexistent_directory"))

    def test_remove_tree_file_not_a_directory(self):
        """
        Test raising an error when the given path is not a directory.
        """
        with self.assertRaises(NotADirectoryError):
            remove_tree(self.temp_file_1)

    def tearDown(self):
        self.temp_dir.cleanup()


class TestRemoveAllSymlink(unittest.TestCase):
    """
    Unit test case for the 'remove_all_symlink' function.

    Methods
    -------
    test_remove_all_symlink():
        Test removing symbolic links within a directory and its subdirectories.

    test_remove_all_symlink_empty_directory()
        Test removing symbolic links from an empty directory.

    test_remove_all_symlink_no_symlinks()
        Test removing symbolic links from a directory with no symbolic links.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

        (self.temp_dir_path / "file1").symlink_to("existing_file")
        (self.temp_dir_path / "dir1").mkdir()
        (self.temp_dir_path / "dir1" / "file2").symlink_to("existing_file")
        (self.temp_dir_path / "dir2").symlink_to("dir1")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_remove_all_symlink(self):
        """
        Test removing symbolic links within a directory and its subdirectories.
        """
        self.assertTrue((self.temp_dir_path / "file1").is_symlink())
        self.assertTrue((self.temp_dir_path / "dir1" / "file2").is_symlink())
        self.assertTrue((self.temp_dir_path / "dir2").is_symlink())

        remove_all_symlink(self.temp_dir_path)

        self.assertFalse((self.temp_dir_path / "file1").is_symlink())
        self.assertFalse((self.temp_dir_path / "dir1" / "file2").is_symlink())
        self.assertFalse((self.temp_dir_path / "dir2").is_symlink())

    def test_remove_all_symlink_empty_directory(self):
        """
        Test removing symbolic links from an empty directory.
        """
        empty_dir = self.temp_dir_path / "empty_dir"
        empty_dir.mkdir()
        remove_all_symlink(empty_dir)

        # Ensure the empty directory still exists
        self.assertTrue(empty_dir.exists())

    def test_remove_all_symlink_no_symlinks(self):
        """
        Test removing symbolic links from a directory with no symbolic links.
        """
        no_symlinks_dir = self.temp_dir_path / "no_symlinks_dir"
        no_symlinks_dir.mkdir()
        file_path = no_symlinks_dir / "file.txt"
        file_path.write_text("Test content")
        remove_all_symlink(no_symlinks_dir)

        self.assertTrue(file_path.is_file())
        self.assertTrue(no_symlinks_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
