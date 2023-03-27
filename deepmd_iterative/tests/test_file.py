# Standard library modules
import logging
import tempfile
import unittest
from pathlib import Path

# Local imports
from deepmd_iterative.common.file import (
    change_directory,
    check_directory,
    check_file_existence,
    file_to_list_of_strings,
    remove_file,
    remove_files_matching_glob,
    remove_tree,
    write_list_of_strings_to_file,
)


class TestChangeDirectory(unittest.TestCase):
    """
    Test case for the `change_directory` function.

    Methods
    -------
    test_change_directory_existing_directory()
        Test changing to an existing directory.
    test_change_directory_nonexistent_directory()
        Test raising an error for a nonexistent directory.
    test_change_directory_file_not_directory()
        Test raising an error for a file instead of a directory.
    test_change_directory_error()
        Test raising an error if there is an error in changing the directory.
    test_change_directory_directory_with_space()
        Test changing to a directory with a space in the name.
    """

    def setUp(self):
        self.temp_dirs = [tempfile.TemporaryDirectory() for _ in range(2)]

    def tearDown(self):
        for temp_dir in self.temp_dirs:
            temp_dir.cleanup()

    def test_change_directory_existing_directory(self):
        """Test that `change_directory` changes to an existing directory."""
        temp_dir = Path(self.temp_dirs[0].name)
        change_directory(temp_dir)
        self.assertEqual(Path.cwd(), temp_dir, msg="Directory not changed to the expected directory.")

    def test_change_directory_nonexistent_directory(self):
        """Test that `change_directory` raises a FileNotFoundError for a nonexistent directory."""
        with self.assertRaises(FileNotFoundError, msg="No exception raised."):
            change_directory(Path("nonexistent_directory"))

    def test_change_directory_file_not_directory(self):
        """Test that `change_directory` raises an Exception for a file instead of a directory."""
        temp_file = Path(self.temp_dirs[0].name) / "temp_file.txt"
        with open(temp_file, "w") as f:
            f.write("This is a temporary file for testing purposes.")
        with self.assertRaises(Exception, msg="No exception raised."):
            change_directory(temp_file)

    def test_change_directory_error(self):
        """Test that `change_directory` raises an Exception if there is an error in changing the directory."""
        with self.assertRaises(Exception, msg="No exception raised."):
            change_directory(Path("/"))

    def test_change_directory_directory_with_space(self):
        """Test that `change_directory` can change to a directory with a space in the name."""
        temp_dir_with_space = Path(self.temp_dirs[1].name) / "directory with space"
        Path.mkdir(temp_dir_with_space)
        change_directory(temp_dir_with_space)
        self.assertEqual(Path.cwd(), temp_dir_with_space, msg="Directory not changed to the expected directory.")


class TestCheckDirectory(unittest.TestCase):
    """Test case for `check_directory` function."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_directory_existing_directory(self):
        """Test checking an existing directory."""
        check_directory(Path(self.temp_dir.name))
        self.assertTrue(True)

    def test_check_directory_nonexistent_directory(self):
        """Test raising an error for a nonexistent directory with abort."""
        with self.assertRaises(FileNotFoundError):
            check_directory(Path("nonexistent_directory"))

    def test_check_directory_nonexistent_directory_no_abort(self):
        """Test logging a warning for a nonexistent directory without abort."""
        with self.assertLogs(logging.WARNING):
            check_directory(Path("nonexistent_directory"), abort_on_error=False)
        self.assertTrue(True)

    def test_check_directory_existing_directory_no_abort(self):
        """Test checking an existing directory without abort."""
        check_directory(Path(self.temp_dir.name), abort_on_error=False)
        self.assertTrue(True)


class TestCheckFileExistence(unittest.TestCase):
    """Test case for `check_file_existence` function."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "temp_file.txt"
        with open(self.temp_file, "w") as f:
            f.write("This is a temporary file for testing purposes.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_file_existence_existing_file(self):
        """Test checking an existing file."""
        check_file_existence(self.temp_file)
        self.assertTrue(True)

    def test_check_file_existence_nonexistent_file(self):
        """Test raising an error for a nonexistent file with abort."""
        with self.assertRaises(FileNotFoundError):
            check_file_existence(Path(self.temp_dir.name) / "nonexistent_file.txt")

    def test_check_file_existence_nonexistent_file_no_abort(self):
        """Test logging a warning for a nonexistent file without abort."""
        with self.assertLogs(logging.WARNING):
            check_file_existence(
                Path(self.temp_dir.name) / "nonexistent_file.txt", abort_on_error=False
            )
        self.assertTrue(True)

    def test_check_file_existence_existing_file_no_abort(self):
        """Test raising an error for an existing file with abort."""
        # Call check_file_existence on an existing file without aborting
        with self.assertRaises(FileExistsError):
            check_file_existence(
                self.temp_file, expected_existence=False, abort_on_error=True
            )

        # Ensure that the function does not abort the program
        self.assertTrue(True)


    """Test case for `file_to_list_of_strings` function."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.temp_file.write("Line 1\nLine 2\nLine 3\n")
        self.temp_file.close()
        self.file_path = Path(self.temp_file.name)

    def tearDown(self):
        self.file_path.unlink()

    def test_file_to_strings(self):
        """Test converting a file to a list of strings."""
        strings = file_to_list_of_strings(self.file_path)
        self.assertIsInstance(strings, list)
        self.assertIsInstance(strings[0], str)
        self.assertEqual(strings, ["Line 1", "Line 2", "Line 3"])

    def test_file_to_strings_file_not_found(self):
        """Test raising an error for a nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            file_to_list_of_strings(Path("/path/to/nonexistent/file.txt"))

        # Ensure that the function does not continue after raising an error
        self.assertTrue(True)


class TestRemoveFile(unittest.TestCase):
    """Test case for `remove_file` function."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "temp_file.txt"
        with open(self.temp_file, "w") as f:
            f.write("This is a temporary file for testing purposes.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_remove_existing_file(self):
        """Test removing an existing file."""
        self.assertTrue(self.temp_file.is_file())
        remove_file(self.temp_file)
        self.assertFalse(self.temp_file.is_file())

    def test_remove_nonexistent_file(self):
        """Test removing a nonexistent file."""
        self.assertFalse((Path(self.temp_dir.name) / "nonexistent_file.txt").is_file())
        remove_file(Path(self.temp_dir.name) / "nonexistent_file.txt")
        self.assertTrue(True)



class TestRemoveFilesMatchingGlob(unittest.TestCase):
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
        # Ensure that all files exist before calling remove_files_matching_glob
        self.assertTrue(self.temp_file_1.is_file())
        self.assertTrue(self.temp_file_2.is_file())
        self.assertTrue(self.temp_file_3.is_file())

        # Call remove_files_matching_glob with a glob pattern that matches two files
        remove_files_matching_glob(Path(self.temp_dir.name), "*.txt")

        # Ensure that only the two matching files were removed
        self.assertFalse(self.temp_file_1.is_file())
        self.assertFalse(self.temp_file_2.is_file())
        self.assertTrue(self.temp_file_3.is_file())

    def test_remove_files_matching_glob_nonexistent_directory(self):
        # Call remove_files_matching_glob with a nonexistent directory
        with self.assertRaises(SystemExit):
            remove_files_matching_glob(Path("nonexistent_directory"), "*.toc")

    def test_remove_files_matching_glob_non_directory(self):
        # Call remove_files_matching_glob with a file instead of a directory
        with self.assertRaises(SystemExit):
            remove_files_matching_glob(self.temp_file_1, "*.txt")

    def test_remove_files_no_matching_glob(self):
        # Ensure that all files exist before calling remove_files_matching_glob
        self.assertTrue(self.temp_file_1.is_file())
        self.assertTrue(self.temp_file_2.is_file())
        self.assertTrue(self.temp_file_3.is_file())

        # Call remove_files_matching_glob with a glob pattern that matches two files
        remove_files_matching_glob(Path(self.temp_dir.name), "*.npz")

        # Ensure that all files exist before calling remove_files_matching_glob
        self.assertTrue(self.temp_file_1.is_file())
        self.assertTrue(self.temp_file_2.is_file())
        self.assertTrue(self.temp_file_3.is_file())


class TestRemoveTree(unittest.TestCase):
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

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_remove_tree(self):
        # Ensure that all files and directories exist before calling remove_tree
        self.assertTrue(self.temp_file_1.is_file())
        self.assertTrue(self.temp_subdir.is_dir())
        self.assertTrue(self.temp_subsubdir.is_dir())
        self.assertTrue(self.temp_file_2.is_file())
        self.assertTrue(self.temp_file_3.is_file())

        # Call remove_tree on the temporary directory
        remove_tree(Path(self.temp_subdir))

        # Ensure that all files and directories have been removed
        self.assertFalse(self.temp_file_1.is_file())
        self.assertFalse(self.temp_subsubdir.is_dir())
        self.assertFalse(self.temp_file_2.is_file())
        self.assertFalse(self.temp_file_3.is_file())

    def test_remove_tree_nonexistent_directory(self):
        # Call remove_tree with a nonexistent directory
        with self.assertRaises(FileNotFoundError):
            remove_tree(Path("nonexistent_directory"))

    def test_remove_tree_file_not_directory(self):
        # Call remove_tree with a file instead of a directory
        with self.assertRaises(NotADirectoryError):
            remove_tree(self.temp_file_1)





if __name__ == "__main__":
    unittest.main()
