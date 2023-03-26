"""
Author: Rolf David
Created: 2023/01/01
Last modified: 2023/03/26
"""
# Standard library modules
import tempfile
import unittest

# Local imports
from deepmd_iterative.common.list import (
    remove_strings_containing_substring_in_list_of_strings,
    replace_substring_in_list_of_strings,
)


class TestRemoveStringsContainingSubstringInList(unittest.TestCase):
    """
    Test case for the remove_strings_containing_substring_in_list_of_strings() function.

    Methods
    -------
    test_remove_strings_containing_substring_in_list_of_strings():
        Test the remove_strings_containing_substring_in_list_of_strings() function with valid input.
    test_remove_strings_containing_substring_in_list_of_strings_empty_list():
        Test the remove_strings_containing_substring_in_list_of_strings() function with an empty input list.
    test_remove_strings_containing_substring_in_list_of_strings_invalid_input():
        Test the remove_strings_containing_substring_in_list_of_strings() function with invalid input types.
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

    def test_remove_strings_containing_substring_in_list_of_strings(self):
        expected_output = [
            "chemical kinetics",
            "thermodynamics",
            "chemical equilibrium",
        ]
        output = remove_strings_containing_substring_in_list_of_strings(
            self.input_list, self.substring
        )
        self.assertEqual(output, expected_output)

    def test_remove_strings_containing_substring_in_list_of_strings_empty_list(self):
        self.input_list = []
        with self.assertRaises(ValueError):
            remove_strings_containing_substring_in_list_of_strings(
                self.input_list, self.substring
            )

    def test_remove_strings_containing_substring_in_list_of_strings_invalid_input(self):
        self.input_list = "not a list"
        with self.assertRaises(TypeError):
            remove_strings_containing_substring_in_list_of_strings(
                self.input_list, self.substring
            )

        self.input_list = [
            "quantum mechanics",
            "chemical kinetics",
            "thermodynamics",
            "quantum chemistry",
            "chemical equilibrium",
        ]
        self.substring = 10
        with self.assertRaises(TypeError):
            remove_strings_containing_substring_in_list_of_strings(
                self.input_list, self.substring
            )


class TestReplaceSubstringInList(unittest.TestCase):
    """
    Test case for the replace_substring_in_list_of_strings() function.

    Methods
    -------
    test_replace_substring_in_list_of_strings():
        Test the function with a list of strings and check that it replaces the specified substring correctly.
    test_replace_substring_in_list_of_strings_invalid_input():
        Test the function with an invalid input and check that it raises a TypeError.
    test_replace_substring_in_list_of_strings_empty_substring():
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

    def test_replace_substring_in_list_of_strings(self):
        expected_output = [
            "classical mechanics",
            "chemical kinetics",
            "thermodynamics",
            "classical chemistry",
            "chemical equilibrium",
        ]
        output = replace_substring_in_list_of_strings(
            self.input_list, self.substring_in, self.substring_out
        )
        self.assertEqual(output, expected_output)

    def test_replace_substring_in_list_of_strings_invalid_input(self):
        input_list = "not a list"
        with self.assertRaises(TypeError):
            replace_substring_in_list_of_strings(
                input_list, self.substring_in, self.substring_out
            )

    def test_replace_substring_in_list_of_strings_empty_substring(self):
        substring_in = ""
        with self.assertRaises(ValueError):
            replace_substring_in_list_of_strings(
                self.input_list, substring_in, self.substring_out
            )

        substring_out = ""
        with self.assertRaises(ValueError):
            replace_substring_in_list_of_strings(
                self.input_list, self.substring_in, substring_out
            )

    def test_replace_substring_in_list_with_temp_file(self):
        with open(self.tmp_file.name, "w") as f:
            f.write("\n".join(self.input_list))

        with open(self.tmp_file.name, "r") as f:
            output = replace_substring_in_list_of_strings(
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


if __name__ == "__main__":
    unittest.main()
