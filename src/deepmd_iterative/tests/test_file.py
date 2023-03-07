from pathlib import Path

# Unittest imports
import unittest
import tempfile

# deepmd_iterative imports
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
    def setUp(self):
        self.temp_dir_1 = tempfile.TemporaryDirectory()
        self.temp_dir_2 = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir_1.cleanup()
        self.temp_dir_2.cleanup()

    def test_change_directory(self):
        # Call change_directory with one of the temporary directories
        change_directory(Path(self.temp_dir_1.name))

        # Ensure that the current working directory has been changed
        self.assertEqual(Path.cwd(), Path(self.temp_dir_1.name))

    def test_change_directory_nonexistent_directory(self):
        # Call change_directory with a nonexistent directory
        with self.assertRaises(SystemExit):
            change_directory(Path("nonexistent_directory"))

    def test_change_directory_file_not_directory(self):
        # Call change_directory with a file instead of a directory
        temp_file = Path(self.temp_dir_1.name) / "temp_file.txt"
        with open(temp_file, "w") as f:
            f.write("This is a temporary file for testing purposes.")
        with self.assertRaises(SystemExit):
            change_directory(temp_file)


class TestCheckDirectory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_directory_existing_directory(self):
        # Call check_directory on an existing directory
        check_directory(Path(self.temp_dir.name))

        # Ensure that the function does not abort the program
        self.assertTrue(True)

    def test_check_directory_nonexistent_directory(self):
        # Call check_directory on a nonexistent directory
        with self.assertRaises(SystemExit):
            check_directory(Path("nonexistent_directory"))

    def test_check_directory_nonexistent_directory_no_abort(self):
        # Call check_directory on a nonexistent directory without aborting
        check_directory(Path("nonexistent_directory"), abort_on_error=False)

        # Ensure that the function does not abort the program
        self.assertTrue(True)

    def test_check_directory_existing_directory_no_abort(self):
        # Call check_directory on an existing directory without aborting
        check_directory(Path(self.temp_dir.name), abort_on_error=False)

        # Ensure that the function does not abort the program
        self.assertTrue(True)


class TestCheckFileExistence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "temp_file.txt"
        with open(self.temp_file, "w") as f:
            f.write("This is a temporary file for testing purposes.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_file_existence_existing_file(self):
        # Call check_file_existence on an existing file
        check_file_existence(self.temp_file)

        # Ensure that the function does not abort the program
        self.assertTrue(True)

    def test_check_file_existence_nonexistent_file(self):
        # Call check_file_existence on a nonexistent file
        with self.assertRaises(SystemExit):
            check_file_existence(Path(self.temp_dir.name) / "nonexistent_file.txt")

    def test_check_file_existence_nonexistent_file_no_abort(self):
        # Call check_file_existence on a nonexistent file without aborting
        check_file_existence(
            Path(self.temp_dir.name) / "nonexistent_file.txt", abort_on_error=False
        )

        # Ensure that the function does not abort the program
        self.assertTrue(True)

    def test_check_file_existence_existing_file_no_abort(self):
        # Call check_file_existence on an existing file without aborting
        check_file_existence(
            self.temp_file, expected_existence=False, abort_on_error=False
        )

        # Ensure that the function does not abort the program
        self.assertTrue(True)


class TestFileToStrings(unittest.TestCase):
    def setUp(self):
        # Create a temporary file and write some lines to it
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.temp_file.write("Line 1\nLine 2\nLine 3\n")
        self.temp_file.close()
        self.file_path = Path(self.temp_file.name)

    def tearDown(self):
        # Remove the temporary file
        self.file_path.unlink()

    def test_file_to_strings(self):
        # Test that the function returns a list of strings
        strings = file_to_list_of_strings(self.file_path)
        self.assertIsInstance(strings, list)
        self.assertIsInstance(strings[0], str)

        # Test that the strings match the lines in the file
        self.assertEqual(strings, ["Line 1", "Line 2", "Line 3"])

    def test_file_to_strings_file_not_found(self):
        # Test that the function raises a ValueError if the file is not found
        with self.assertRaises(SystemExit):
            file_to_list_of_strings(Path("/path/to/nonexistent/file.txt"))


class TestRemoveFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "temp_file.txt"
        with open(self.temp_file, "w") as f:
            f.write("This is a temporary file for testing purposes.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_remove_existing_file(self):
        # Ensure that the file exists before calling remove_file
        self.assertTrue(self.temp_file.is_file())

        # Call remove_file and ensure that the file no longer exists
        remove_file(self.temp_file)
        self.assertFalse(self.temp_file.is_file())

    def test_remove_nonexistent_file(self):
        # Ensure that the file does not exist before calling remove_file
        self.assertFalse((Path(self.temp_dir.name) / "nonexistent_file.txt").is_file())

        # Call remove_file and ensure that it does not raise an exception
        remove_file(Path(self.temp_dir.name) / "nonexistent_file.txt")


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


class TestWriteListOfStringsToFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for testing
        self.temp_file = Path(tempfile.mkstemp()[1])

    def tearDown(self):
        # Remove the temporary file after testing
        self.temp_file.unlink()

    def test_writes_to_file(self):
        # Define test data
        expected_output = ["foo", "bar", "baz"]
        input_file = self.temp_file
        # Call the function under test
        write_list_of_strings_to_file(input_file, expected_output)
        # Check that the file was written correctly
        with input_file.open("r") as f:
            lines = f.readlines()
        self.assertEqual(lines, [f"{s}\n" for s in expected_output])


if __name__ == "__main__":
    unittest.main()
