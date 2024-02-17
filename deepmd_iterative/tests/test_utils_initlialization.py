"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/02/17

Test cases for the utils/initialization module.

Classes
-------
TestGenerateMainJson
    Test cases for the 'generate_main_json' function.
"""

# Standard library modules
import unittest

# Local imports
from deepmd_iterative.initialization.utils import generate_main_json


class TestGenerateMainJson(unittest.TestCase):
    """
    Test cases for the 'generate_main_json' function.

    Methods
    -------
    test_set_main_config_with_valid_input():
        Test if the function correctly generates main and merged input JSON with valid input.

    test_set_main_config_with_minimal_input():
        Test if the function correctly generates main and merged input JSON with minimal input.

    test_set_main_config_with_invalid_type_input():
        Test if the function correctly raises TypeError for invalid input types.

    test_set_main_config_with_missing_mandatory_input():
        Test if the function correctly raises ValueError for missing mandatory input.

    test_set_main_config_with_invalid_element_type_in_list():
        Test if the function correctly raises TypeError for invalid element types in a list.
    """

    def setUp(self):
        self.default_json = {
            "systems_auto": [""],
            "nnp_count": 3,
            "exploration_type": "lammps",
        }

    def test_generate_main_json_with_valid_input(self):
        """
        Test if the function correctly generates main and merged input JSON with valid input.
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
        Test if the function correctly generates main and merged input JSON with minimal input.
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
        Test if the function correctly raises TypeError for invalid input types.
        """
        input_json = {
            "systems_auto": ["subsys1", 2],
            "nnp_count": "not a number",
        }

        with self.assertRaises(TypeError):
            generate_main_json(input_json, self.default_json)

    def test_generate_main_json_with_missing_mandatory_input(self):
        """
        Test if the function correctly raises ValueError for missing mandatory input.
        """
        input_json = {
            "nnp_count": 4,
        }

        with self.assertRaises(ValueError):
            generate_main_json(input_json, self.default_json)

    def test_generate_main_json_with_invalid_element_type_in_list(self):
        """
        Test if the function correctly raises TypeError for invalid element types in a list.
        """
        input_json = {
            "systems_auto": ["subsys1", 2],
            "nnp_count": 2,
        }

        with self.assertRaises(TypeError):
            generate_main_json(input_json, self.default_json)


if __name__ == "__main__":
    unittest.main()
