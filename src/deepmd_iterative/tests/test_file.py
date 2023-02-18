import unittest
import tempfile
from pathlib import Path
from deepmd_iterative.common.file import (
    file_to_list_of_strings,
    write_list_of_strings_to_file,
)


class TestFileToStrings(unittest.TestCase):
    def setUp(self):
        # create a temporary file and write some lines to it
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.temp_file.write("Line 1\nLine 2\nLine 3\n")
        self.temp_file.close()
        self.file_path = Path(self.temp_file.name)

    def tearDown(self):
        # remove the temporary file
        self.file_path.unlink()

    def test_file_to_strings(self):
        # test that the function returns a list of strings
        strings = file_to_list_of_strings(self.file_path)
        self.assertIsInstance(strings, list)
        self.assertIsInstance(strings[0], str)

        # test that the strings match the lines in the file
        self.assertEqual(strings, ["Line 1", "Line 2", "Line 3"])

    def test_file_to_strings_file_not_found(self):
        # test that the function raises a ValueError if the file is not found
        with self.assertRaises(SystemExit):
            file_to_list_of_strings(Path("/path/to/nonexistent/file.txt"))


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
