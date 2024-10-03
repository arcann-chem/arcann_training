"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15

This module contains unit tests for the 'utils' module in the 'initialization' package.

Classes
-------
TestGenerateMainJson
    Test suite for the 'generate_main_json' function.
TestCheckPropertiesFile
    Test suite for the 'check_properties_file' function.
"""

# Standard library modules
import tempfile
import unittest
from pathlib import Path

# Local imports
from arcann_training.initialization.utils import (
    generate_main_json,
    check_properties_file,
)


class TestGenerateMainJson(unittest.TestCase):
    """
    Test suite for the 'generate_main_json' function.

    Methods
    -------
    test_set_main_config_with_valid_input():
        Tests correct JSON generation and merging with complete and valid input.
    test_set_main_config_with_minimal_input():
        Tests JSON generation and merging using minimal input, relying on default settings.
    test_set_main_config_with_invalid_type_input():
        Tests response to input with incorrect data types, expecting a TypeError.
    test_set_main_config_with_invalid_element_type_in_list():
        Tests handling of invalid element types within list structures, expecting a TypeError.
    """

    def setUp(self):
        self.default_json = {
            "systems_auto": [""],
            "nnp_count": 3,
        }

    def test_generate_main_json_with_valid_input(self):
        """
        Tests correct JSON generation and merging with complete and valid input.
        """
        input_json = {
            "systems_auto": ["subsys1", "subsys2"],
            "nnp_count": 5,
        }

        expected_config_json = {
            "systems_auto": {"subsys1": {"index": 0}, "subsys2": {"index": 1}},
            "nnp_count": 5,
            "current_iteration": 0,
        }
        expected_merged_input_json = {
            "systems_auto": ["subsys1", "subsys2"],
            "nnp_count": 5,
        }
        expected_padded_curr_iter = "000"

        config_json, merged_input_json, padded_curr_iter = generate_main_json(
            input_json, self.default_json
        )

        self.assertDictEqual(config_json, expected_config_json)
        self.assertDictEqual(merged_input_json, expected_merged_input_json)
        self.assertEqual(padded_curr_iter, expected_padded_curr_iter)

    def test_generate_main_json_with_minimal_input(self):
        """
        Tests JSON generation and merging using minimal input, relying on default settings.
        """
        input_json = {
            "systems_auto": ["subsys1", "subsys2"],
        }

        expected_config_json = {
            "systems_auto": {"subsys1": {"index": 0}, "subsys2": {"index": 1}},
            "nnp_count": 3,
            "current_iteration": 0,
        }
        expected_merged_input_json = {
            "systems_auto": ["subsys1", "subsys2"],
            "nnp_count": 3,
        }
        expected_padded_curr_iter = "000"

        config_json, merged_input_json, padded_curr_iter = generate_main_json(
            input_json, self.default_json
        )

        self.assertDictEqual(config_json, expected_config_json)
        self.assertDictEqual(merged_input_json, expected_merged_input_json)
        self.assertEqual(padded_curr_iter, expected_padded_curr_iter)

    def test_generate_main_json_with_invalid_type_input(self):
        """
        Tests response to input with incorrect data types, expecting a TypeError.
        """
        input_json = {
            "systems_auto": ["subsys1", 2],
            "nnp_count": "not a number",
        }

        with self.assertRaises(TypeError):
            generate_main_json(input_json, self.default_json)

    def test_generate_main_json_with_invalid_element_type_in_list(self):
        """
        Tests handling of invalid element types within list structures, expecting a TypeError.
        """
        input_json = {
            "systems_auto": ["subsys1", 2],
            "nnp_count": 2,
        }

        with self.assertRaises(TypeError):
            generate_main_json(input_json, self.default_json)


class TestCheckPropertiesFile(unittest.TestCase):
    """
    Test suite for the 'check_properties_file' function.

    Methods
    -------
    test_correct_file():
        Tests parsing of a correctly formatted properties file.
    test_file_not_found():
        Tests the function's response to a non-existent file, expecting a FileNotFoundError.
    test_missing_type_section():
        Tests handling of a file missing the 'type' section, expecting a ValueError.
    test_missing_masses_section():
        Tests handling of a file missing the 'masses' section, expecting a ValueError.
    test_incorrect_order_sections():
        Tests file parsing when sections are in the incorrect order, expecting a ValueError.
    test_incorrect_data_types_type():
        Tests handling of incorrect data types in the 'type' section, expecting a ValueError.
    test_incorrect_data_types_mass():
        Tests handling of incorrect data types in the 'masses' section, expecting a ValueError.
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def create_temp_file(self, content):
        temp_file = Path(self.test_dir.name) / "temp_properties_file.txt"
        with open(temp_file, "w") as file:
            file.write(content)
        return temp_file

    def test_correct_file(self):
        """
        Tests parsing of a correctly formatted properties file.
        """
        content = """type
H 1
He 2
masses
H 1.007
He 4.002
"""
        temp_file = self.create_temp_file(content)
        result = check_properties_file(temp_file)
        self.assertEqual(
            result,
            {1: {"symbol": "H", "mass": 1.007}, 2: {"symbol": "He", "mass": 4.002}},
        )

    def test_file_not_found(self):
        """
        Tests the function's response to a non-existent file, expecting a FileNotFoundError.
        """
        non_existent_file = Path(self.test_dir.name) / "non_existent_file.txt"
        with self.assertRaises(FileNotFoundError):
            check_properties_file(non_existent_file)

    def test_missing_type_section(self):
        """
        Tests handling of a file missing the 'type' section, expecting a ValueError.
        """
        content = """masses
H 1.007
He 4.002
"""
        temp_file = self.create_temp_file(content)
        with self.assertRaises(ValueError):
            check_properties_file(temp_file)

    def test_missing_masses_section(self):
        """
        Tests handling of a file missing the 'masses' section, expecting a ValueError.
        """
        content = """type
H 1
He 2
"""
        temp_file = self.create_temp_file(content)
        with self.assertRaises(ValueError):
            check_properties_file(temp_file)

    def test_incorrect_order_sections(self):
        """
        Tests file parsing when sections are in the incorrect order, expecting a ValueError.
        """
        content = """masses
H 1.007
He 4.002
type
H 1
He 2
"""
        temp_file = self.create_temp_file(content)
        with self.assertRaises(ValueError):
            check_properties_file(temp_file)

    def test_incorrect_data_types_type(self):
        """
        Tests handling of incorrect data types in the 'type' section, expecting a ValueError.
        """
        content = """type
H one
He two
masses
H 1.007
He four.002
"""
        temp_file = self.create_temp_file(content)
        with self.assertRaises(ValueError):
            check_properties_file(temp_file)

    def test_incorrect_data_types_mass(self):
        """
        Tests handling of incorrect data types in the 'masses' section, expecting a ValueError.
        """
        content = """type
H 1
He 2
masses
H 1.007
He four.002
"""
        temp_file = self.create_temp_file(content)
        with self.assertRaises(ValueError):
            check_properties_file(temp_file)


if __name__ == "__main__":
    unittest.main()
