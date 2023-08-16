"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/16
"""
# Standard library modules
import unittest

# Local imports
from deepmd_iterative.common.json_parameters import set_main_config


class TestSetConfigJson(unittest.TestCase):
    def setUp(self):
        self.default_json = {
            "system": "",
            "subsys_nr": [""],
            "nnp_count": 3,
            "exploration_type": "lammps",
        }

    def test_set_main_config_with_valid_input(self):
        input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", "subsys2"],
            "nnp_count": 5,
            "exploration_type": "lammps",
        }

        expected_config_json = {
            "system": "My system",
            "subsys_nr": {"subsys1": {}, "subsys2": {}},
            "nnp_count": 5,
            "exploration_type": "lammps",
            "current_iteration": 0,
        }
        expected_new_input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", "subsys2"],
            "nnp_count": 5,
            "exploration_type": "lammps",
        }
        expected_padded_curr_iter = "000"
        config_json, new_input_json, padded_curr_iter = set_main_config(
            input_json, self.default_json
        )
        self.assertDictEqual(config_json, expected_config_json)
        self.assertDictEqual(new_input_json, expected_new_input_json)
        self.assertEqual(padded_curr_iter, expected_padded_curr_iter)

    def test_set_main_config_with_minimal_input(self):
        input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", "subsys2"],
        }

        expected_config_json = {
            "system": "My system",
            "subsys_nr": {"subsys1": {}, "subsys2": {}},
            "nnp_count": 3,
            "exploration_type": "lammps",
            "current_iteration": 0,
        }
        expected_new_input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", "subsys2"],
            "nnp_count": 3,
            "exploration_type": "lammps",
        }
        expected_padded_curr_iter = "000"
        config_json, new_input_json, padded_curr_iter = set_main_config(
            input_json, self.default_json
        )
        self.assertDictEqual(config_json, expected_config_json)
        self.assertDictEqual(new_input_json, expected_new_input_json)
        self.assertEqual(padded_curr_iter, expected_padded_curr_iter)

    def test_set_main_config_with_invalid_input(self):
        input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", 2],
            "nnp_count": "not a number",
            "exploration_type": "invalid",
        }
        with self.assertRaises(TypeError):
            set_main_config(input_json, self.default_json)

    def test_set_main_config_with_missing_input(self):
        input_json = {
            "system": "My system",
            "nnp_count": 4,
            "exploration_type": "lammps",
        }
        with self.assertRaises(ValueError):
            set_main_config(input_json, self.default_json)

    def test_set_main_config_with_invalid_input2(self):
        input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", 2],
            "nnp_count": 2,
        }
        with self.assertRaises(TypeError):
            set_main_config(input_json, self.default_json)


if __name__ == "__main__":
    unittest.main()
