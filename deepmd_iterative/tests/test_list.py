"""
Created: 2023/01/01
Last modified: 2023/03/27

Test cases for the list module.

Class
-----
TestExcludeSubstringFromStringList
    Test case for the exclude_substring_from_string_list() function.

TestReplaceSubstringInStringList
    Test case for the replace_substring_in_string_list() function.

TestStringListToTextfile
    Test case for the string_list_to_textfile() function.

TestTextfileToStringList
    Test case for the textfile_to_string_list() function.
"""
# Standard library modules
import tempfile
import unittest
from pathlib import Path

# Local imports
from deepmd_iterative.common.list import (
    exclude_substring_from_string_list,
    replace_substring_in_string_list,
    string_list_to_textfile,
    textfile_to_string_list,
)


class TestExcludeSubstringFromStringList(unittest.TestCase):
    """
    Test case for the exclude_substring_from_string_list() function.

    Methods
    -------
    test_exclude_substring_from_string_list():
        Test the exclude_substring_from_string_list() function with valid input.
    test_exclude_substring_from_string_list_empty_list():
        Test the exclude_substring_from_string_list() function with an empty input list.
    test_exclude_substring_from_string_list_invalid_input():
        Test the exclude_substring_from_string_list() function with invalid input types.
    """

    def setUp(self):
        self.input_list = [
            "quantum mechanics",
            "chemical kinetics",
            "thermodynamics",
            "quantum chemistry",
            "chemical equilibrium",
        ]
        self.substring = "quantum"

    def tearDown(self):
        pass

    def test_exclude_substring_from_string_list(self):
        expected_output = [
            "chemical kinetics",
            "thermodynamics",
            "chemical equilibrium",
        ]
        output = exclude_substring_from_string_list(self.input_list, self.substring)
        self.assertEqual(output, expected_output)

    def test_exclude_substring_from_string_list_empty_list(self):
        self.input_list = []
        with self.assertRaises(ValueError):
            exclude_substring_from_string_list(self.input_list, self.substring)

    def test_exclude_substring_from_string_list_invalid_input(self):
        self.input_list = "not a list"
        with self.assertRaises(TypeError):
            exclude_substring_from_string_list(self.input_list, self.substring)

        self.input_list = [
            "quantum mechanics",
            "chemical kinetics",
            "thermodynamics",
            "quantum chemistry",
            "chemical equilibrium",
        ]
        self.substring = 10
        with self.assertRaises(TypeError):
            exclude_substring_from_string_list(self.input_list, self.substring)


class TestReplaceSubstringInStringList(unittest.TestCase):
    """
    Test case for the replace_substring_in_string_list() function.

    Methods
    -------
    test_replace_substring_in_string_list():
        Test the function with a list of strings and check that it replaces the specified substring correctly.
    test_replace_substring_in_string_list_invalid_input():
        Test the function with an invalid input and check that it raises a TypeError.
    test_replace_substring_in_string_list_empty_substring():
        Test the function with an empty substring and check that it raises a ValueError.
    test_replace_substring_in_list_with_temp_file():
        Test the function with a file object and check that it replaces the specified substring correctly.
    """

    def setUp(self):
        self.input_list = [
            "quantum mechanics",
            "chemical kinetics",
            "thermodynamics",
            "quantum chemistry",
            "chemical equilibrium",
        ]
        self.substring_in = "quantum"
        self.substring_out = "classical"
        self.tmp_file = tempfile.NamedTemporaryFile()

    def tearDown(self):
        self.tmp_file.close()

    def test_replace_substring_in_string_list(self):
        expected_output = [
            "classical mechanics",
            "chemical kinetics",
            "thermodynamics",
            "classical chemistry",
            "chemical equilibrium",
        ]
        output = replace_substring_in_string_list(
            self.input_list, self.substring_in, self.substring_out
        )
        self.assertEqual(output, expected_output)

    def test_replace_substring_in_string_list_invalid_input(self):
        input_list = "not a list"
        with self.assertRaises(TypeError):
            replace_substring_in_string_list(
                input_list, self.substring_in, self.substring_out
            )

    def test_replace_substring_in_string_list_empty_substring(self):
        substring_in = ""
        with self.assertRaises(ValueError):
            replace_substring_in_string_list(
                self.input_list, substring_in, self.substring_out
            )

        substring_out = ""
        with self.assertRaises(ValueError):
            replace_substring_in_string_list(
                self.input_list, self.substring_in, substring_out
            )

    def test_replace_substring_in_list_with_temp_file(self):
        with open(self.tmp_file.name, "w") as f:
            f.write("\n".join(self.input_list))

        with open(self.tmp_file.name, "r") as f:
            output = replace_substring_in_string_list(
                f.readlines(), self.substring_in, self.substring_out
            )

        expected_output = [
            "classical mechanics",
            "chemical kinetics",
            "thermodynamics",
            "classical chemistry",
            "chemical equilibrium",
        ]
        self.assertEqual(output, expected_output)


class TestStringListToTextfile(unittest.TestCase):
    """
    Test case for the string_list_to_textfile() function.

    Methods
    -------
    test_string_list_to_textfile_writes_to_file():
        Test the function writing a list of strings to a text file.
    test_string_list_to_textfile_with_empty_list():
        Test the function raising a `ValueError` for an empty `string_list`.
    test_string_list_to_textfile_with_one_string():
        Test the function writing a list with one string to a text file.
    test_string_list_to_textfile_appends_to_file():
        Test the function appending a list of strings to an existing file.
    """

    def setUp(self):
        self.temp_file = Path(tempfile.mkstemp()[1])

    def tearDown(self):
        self.temp_file.unlink()

    def test_string_list_to_textfile_writes_to_file(self):
        expected_output = ["foo", "bar", "baz"]
        print(expected_output)
        input_file = self.temp_file
        string_list_to_textfile(input_file, expected_output)
        with input_file.open("r") as f:
            lines = f.readlines()
        print(expected_output)
        expected_lines = [f"{s}\n" for s in expected_output]
        self.assertEqual(
            lines, expected_lines, "The file does not contain the expected contents"
        )

    def test_string_list_to_textfile_with_empty_list(self):
        input_file = self.temp_file
        with self.assertRaises(ValueError):
            string_list_to_textfile(input_file, [])

    def test_string_list_to_textfile_with_one_string(self):
        expected_output = ["foo"]
        input_file = self.temp_file
        string_list_to_textfile(input_file, expected_output)
        with input_file.open("r") as f:
            lines = f.readlines()
        expected_lines = [f"{s}\n" for s in expected_output]
        self.assertEqual(
            lines, expected_lines, "The file does not contain the expected contents"
        )

    def test_string_list_to_textfile_appends_to_file(self):
        existing_content = ["existing", "content"]
        input_file = self.temp_file
        with input_file.open("w") as f:
            f.write("\n".join(existing_content))
        new_content = ["foo", "bar", "baz"]
        string_list_to_textfile(input_file, new_content)
        with input_file.open("r") as f:
            lines = f.readlines()
        expected_lines = [f"{s}\n" for s in new_content]
        self.assertEqual(
            lines, expected_lines, "The file does not contain the expected contents"
        )


class TestTextfileToStringList(unittest.TestCase):
    """
    Test case for the textfile_to_string_list() function.

    Methods
    test_textfile_to_string_list_with_existing_file():
        Test the function reading a file with multiple lines to a list of strings.
    test_textfile_to_string_list_with_empty_file():
        Test the function reading an empty file to an empty list.
    test_textfile_to_string_list_with_one_line_file():
        Test the function reading a file with one line to a list containing that line.
    test_textfile_to_string_list_with_nonexistent_file():
        Test the function to raise a `FileNotFoundError` for a nonexistent file.
    """

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.temp_file.write("Line 1\nLine 2\nLine 3\n")
        self.temp_file.close()
        self.file_path = Path(self.temp_file.name)

    def tearDown(self):
        self.file_path.unlink()

    def test_textfile_to_string_list_with_existing_file(self):
        strings = textfile_to_string_list(self.file_path)
        self.assertIsInstance(strings, list)
        self.assertIsInstance(strings[0], str)
        self.assertEqual(strings, ["Line 1", "Line 2", "Line 3"])

    def test_textfile_to_string_list_with_empty_file(self):
        empty_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        empty_file.close()
        empty_file_path = Path(empty_file.name)
        strings = textfile_to_string_list(empty_file_path)
        self.assertEqual(strings, [])
        empty_file_path.unlink()

    def test_textfile_to_string_list_with_one_line_file(self):
        one_line_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        one_line_file.write("Line 1")
        one_line_file.close()
        one_line_file_path = Path(one_line_file.name)
        strings = textfile_to_string_list(one_line_file_path)
        self.assertEqual(strings, ["Line 1"])
        one_line_file_path.unlink()

    def test_textfile_to_string_list_with_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            textfile_to_string_list(Path("/path/to/nonexistent/file.txt"))


if __name__ == "__main__":
    unittest.main()
