from pathlib import Path
import unittest
import tempfile

# deepmd_iterative imports
from deepmd_iterative.common.list import (
    replace_substring_in_list_of_strings,
    remove_strings_containing_substring_in_list_of_strings,
)


class TestReplaceSubstringInList(unittest.TestCase):
    def setUp(self):
        # Create a list of input strings to use in the tests
        self.input_list = ["hello world", "foo bar", "baz qux"]

        # Create a temporary file to use in the tests
        self.tmp_file = tempfile.NamedTemporaryFile()

    def tearDown(self):
        # Delete the temporary file
        self.tmp_file.close()

    def test_replace_substring_in_list(self):
        # Call the function with some test inputs
        output_list = replace_substring_in_list_of_strings(self.input_list, "o", "O")

        # Check that the output list has the correct length
        self.assertEqual(len(output_list), len(self.input_list))

        # Check that each string in the output list has the specified substring replaced
        self.assertEqual(output_list[0], "hellO wOrld")
        self.assertEqual(output_list[1], "fOO bar")
        self.assertEqual(output_list[2], "baz qux")

    def test_replace_substring_in_list_with_empty_input_list(self):
        # Call the function with an empty input list
        output_list = replace_substring_in_list_of_strings([], "o", "O")

        # Check that the output list is also empty
        self.assertEqual(len(output_list), 0)

    def test_replace_substring_in_list_with_temp_file(self):
        # Write the input list to the temporary file
        with open(self.tmp_file.name, "w") as f:
            f.write("\n".join(self.input_list))

        # Call the function with the contents of the temporary file
        with open(self.tmp_file.name, "r") as f:
            output_list = replace_substring_in_list_of_strings(f.readlines(), "o", "O")

        # Check that the output list has the correct length
        self.assertEqual(len(output_list), len(self.input_list))

        # Check that each string in the output list has the specified substring replaced
        self.assertEqual(output_list[0], "hellO wOrld")
        self.assertEqual(output_list[1], "fOO bar")
        self.assertEqual(output_list[2], "baz qux")


class TestRemoveStringsContainingSubstringInList(unittest.TestCase):
    def setUp(self):
        self.input_list = ["hello", "world", "hello world", "goodbye"]
        self.substring = "hello"

    def tearDown(self):
        pass

    def test_remove_strings_containing_substring(self):
        output_list = remove_strings_containing_substring_in_list_of_strings(
            self.input_list, self.substring
        )
        self.assertListEqual(output_list, ["world", "goodbye"])
