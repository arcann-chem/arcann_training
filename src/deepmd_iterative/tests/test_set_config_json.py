import unittest

from deepmd_iterative.common.json_parameters import set_config_json


class TestSetConfigJson(unittest.TestCase):
    def setUp(self):
        self.default_json = {
            "system": "",
            "subsys_nr": [""],
            "nb_nnp": 3,
            "exploration_type": "lammps"
         }

    def test_set_config_json_with_valid_input(self):
        input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", "subsys2"],
            "nb_nnp": 5,
            "exploration_type": "lammps"
        }

        expected_config_json = {
            "system": "My system",
            "subsys_nr": {
                "subsys1": {},
                "subsys2": {}
            },
            "nb_nnp": 5,
            "exploration_type": "lammps",
            "current_iteration": 0
        }
        expected_new_input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", "subsys2"],
            "nb_nnp": 5,
            "exploration_type": "lammps"
        }
        expected_padded_curr_iter = "000"
        config_json, new_input_json, padded_curr_iter = set_config_json(input_json, self.default_json)
        self.assertDictEqual(config_json, expected_config_json)
        self.assertDictEqual(new_input_json, expected_new_input_json)
        self.assertEqual(padded_curr_iter, expected_padded_curr_iter)

    def test_set_config_json_with_minimal_input(self):
        input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", "subsys2"],
        }

        expected_config_json = {
            "system": "My system",
            "subsys_nr": {
                "subsys1": {},
                "subsys2": {}
            },
            "nb_nnp": 3,
            "exploration_type": "lammps",
            "current_iteration": 0
        }
        expected_new_input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", "subsys2"],
            "nb_nnp": 3,
            "exploration_type": "lammps"
        }
        expected_padded_curr_iter = "000"
        config_json, new_input_json, padded_curr_iter = set_config_json(input_json, self.default_json)
        self.assertDictEqual(config_json, expected_config_json)
        self.assertDictEqual(new_input_json, expected_new_input_json)
        self.assertEqual(padded_curr_iter, expected_padded_curr_iter)

    def test_set_config_json_with_invalid_input(self):
        input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", 2],
            "nb_nnp": "not a number",
            "exploration_type": "invalid"
        }
        with self.assertRaises(SystemExit):
            set_config_json(input_json, self.default_json)
            
    def test_set_config_json_with_missing_input(self):
        input_json = {
            "system": "My system",
            "nb_nnp": 4,
            "exploration_type": "lammps"
        }
        with self.assertRaises(SystemExit):
            set_config_json(input_json, self.default_json)
            
    def test_set_config_json_with_invalid_input2(self):
        input_json = {
            "system": "My system",
            "subsys_nr": ["subsys1", 2],
            "nb_nnp": 2,
        }
        with self.assertRaises(SystemExit):
            set_config_json(input_json, self.default_json)

if __name__ == '__main__':
    unittest.main()

