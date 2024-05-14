"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/14

Test cases for the json module.

Classes
-------
TestAddKeyValueToDict():
    Test case for the 'add_key_value_to_dict' function.

TestGetKeyInDict():
    Test case for the 'get_key_in_dict' function.

TestBackupAndOverwriteJsonFile():
    Test case for the 'backup_and_overwrite_json_file' function.

TestLoadDefaultJsonFile():
    Test case for the 'load_default_json_file' function.

TestLoadJsonFile():
    Test case for the 'load_json_file' function.

TestWriteJsonFile():
    Test case for the 'write_json_file' function.
"""

# Standard library modules
import json
import os
import tempfile
import unittest
from pathlib import Path

# Local imports
from arcann_training.common.json import (
    add_key_value_to_dict,
    get_key_in_dict,
    backup_and_overwrite_json_file,
    load_default_json_file,
    load_json_file,
    write_json_file,
)


class TestAddKeyValueToDict(unittest.TestCase):
    """
    Test case for the 'add_key_value_to_dict' function.

    Methods
    -------
    test_add_to_empty_dict():
        Test that the function adds a key-value pair to an empty dictionary.
    test_add_new_key_to_dict():
        Test that the function adds a key-value pair to a non-empty dictionary.
    test_update_existing_key_in_dict():
        Test that the function updates the value of an existing key in the dictionary.
    test_add_integer_value_to_dict():
        Test that the function adds an integer value to the dictionary.
    test_add_dict_value_to_dict():
        Test that the function adds a dictionary value to the dictionary.
    test_add_list_value_to_dict():
        Test that the function adds a list value to the dictionary.
    test_add_nested_dict_to_dict():
        Test that the function adds a nested dictionary to the dictionary.
    test_input_types():
        Test that the function raises TypeError/ValueError when given invalid input types.
    """

    def test_add_to_empty_dict(self):
        """
        Test that the function adds a key-value pair to an empty dictionary.
        """
        d = {}
        add_key_value_to_dict(d, "key1", "value1")
        self.assertEqual(d, {"key1": {"value": "value1"}})

    def test_add_new_key_to_dict(self):
        """
        Test that the function adds a key-value pair to a non-empty dictionary.
        """
        d = {"key1": {"value": "value1"}}
        add_key_value_to_dict(d, "key2", "value2")
        self.assertEqual(d, {"key1": {"value": "value1"}, "key2": {"value": "value2"}})

    def test_update_existing_key_in_dict(self):
        """
        Test that the function updates the value of an existing key in the dictionary.
        """
        d = {"key1": {"value": "value1"}}
        add_key_value_to_dict(d, "key1", "new_value1")
        self.assertEqual(d, {"key1": {"value": "new_value1"}})

    def test_add_integer_value_to_dict(self):
        """
        Test that the function adds an integer value to the dictionary.
        """
        d = {}
        add_key_value_to_dict(d, "key1", 123)
        self.assertEqual(d, {"key1": {"value": 123}})

    def test_add_dict_value_to_dict(self):
        """
        Test that the function adds a dictionary value to the dictionary.
        """
        d = {"key1": {"value": "value1"}}
        value = {"key2": "value2"}
        add_key_value_to_dict(d, "key2", value)
        self.assertEqual(d, {"key1": {"value": "value1"}, "key2": {"value": value}})

    def test_add_list_value_to_dict(self):
        """
        Test that the function adds a list value to the dictionary.
        """
        d = {"key1": {"value": "value1"}}
        value = ["item1", "item2"]
        add_key_value_to_dict(d, "key2", value)
        self.assertEqual(d, {"key1": {"value": "value1"}, "key2": {"value": value}})

    def test_add_nested_dict_to_dict(self):
        """
        Test that the function adds a nested dictionary to the dictionary.
        """
        d = {"key1": {"value": "value1"}}
        nested_dict = {"key2": {"key3": "value3"}}
        add_key_value_to_dict(d, "key2", nested_dict)
        self.assertEqual(d, {"key1": {"value": "value1"}, "key2": {"value": nested_dict}})

    def test_input_types(self):
        """
        Test that the function raises TypeError/ValueError when given invalid input types.
        """
        d = {}
        with self.assertRaises(TypeError):
            add_key_value_to_dict(None, "key1", "value1")
        with self.assertRaises(TypeError):
            add_key_value_to_dict(d, 123, "value1")
        with self.assertRaises(ValueError):
            add_key_value_to_dict(d, "", "value1")
        with self.assertRaises(TypeError):
            add_key_value_to_dict(d, "key1", None)


class TestGetKeyInDict(unittest.TestCase):
    """
    Test case for the 'get_key_in_dict' function.

    Methods
    -------
    test_get_existing_key_from_input_json():
        Test getting an existing key from the input JSON.
    test_get_nonexistent_key_from_input_json():
        Test getting a nonexistent key from the input JSON.
    test_get_existing_key_from_previous_json():
        Test getting an existing key from the previous JSON.
    test_get_existing_key_from_default_json():
        Test getting an existing key from the default JSON.
    test_key_not_present_in_any_json():
        Test getting a key not present in any JSON.
    test_wrong_type_from_input_json():
        Test getting a key with wrong type from the input JSON.
    test_wrong_type_from_previous_json():
        Test getting a key with wrong type from the previous JSON.
    """

    def setUp(self):
        self.input_json = {"key1": 42}
        self.input_json_incomplete = {}
        self.previous_json = {"key1": 5, "key3": "bob"}
        self.previous_json_incomplete = {}
        self.default_json = {"key1": 0, "key2": "", "key3": False}

    def test_get_existing_key_from_input_json(self):
        """
        Test getting an existing key from the input JSON.
        """
        result = get_key_in_dict("key1", self.input_json, self.previous_json, self.default_json)
        self.assertEqual(result, 42)

    def test_get_nonexistent_key_from_input_json(self):
        """
        Test getting a nonexistent key from the input JSON.
        """
        result = get_key_in_dict(
            "key1",
            self.input_json_incomplete,
            self.previous_json_incomplete,
            self.default_json,
        )
        self.assertEqual(result, 0)  # Should return the default value from default_json

    def test_get_existing_key_from_previous_json(self):
        """
        Test getting an existing key from the previous JSON.
        """
        result = get_key_in_dict("key1", self.input_json_incomplete, self.previous_json, self.default_json)
        self.assertEqual(result, 5)

    def test_get_existing_key_from_default_json(self):
        """
        Test getting an existing key from the default JSON.
        """
        result = get_key_in_dict("key2", self.input_json, self.previous_json, self.default_json)
        self.assertEqual(result, "")

    def test_key_not_present_in_any_json(self):
        """
        Test getting a key not present in any JSON.
        """
        with self.assertRaises(KeyError):
            get_key_in_dict(
                "nonexistent_key",
                self.input_json,
                self.previous_json,
                self.default_json,
            )

    def test_wrong_type_from_input_json(self):
        """
        Test getting a key with wrong type from the input JSON.
        """
        with self.assertRaises(TypeError):
            get_key_in_dict(
                "key1",
                {"key1": "invalid_type_value"},
                self.previous_json,
                self.default_json,
            )

    def test_wrong_type_from_previous_json(self):
        """
        Test getting a key with wrong type from the previous JSON.
        """
        with self.assertRaises(TypeError):
            get_key_in_dict("key3", self.input_json, self.previous_json, self.default_json)


class TestBackupAndOverwriteJsonFile(unittest.TestCase):
    """
    Test case for the 'backup_and_overwrite_json_file' function.

    Methods
    -------
    test_backup_and_overwrite_json_file():
        Test the function with a path to an existing file, checking that a backup file is created and the file is overwritten with new data.
    test_backup_and_overwrite_json_file_symlink():
        Test the function with a path to an existing symlink, checking that the symlink is converted to a file and the data is overwritten.
    test_backup_and_overwrite_json_file_invalid_input():
        Test the function with invalid input (not a Path object), checking that a TypeError is raised.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "test.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_backup_and_overwrite_json_file(self):
        """
        Test the function with a path to an existing file, checking that a backup file is created and the file is overwritten with new data.
        """
        initial_data = {"a": 1, "b": 2}
        with self.file_path.open("w") as f:
            json.dump(initial_data, f)
        new_data = {"c": 3, "d": 4}
        backup_and_overwrite_json_file(new_data, self.file_path)
        self.assertTrue(self.file_path.with_suffix(".json").is_file())
        with self.file_path.with_suffix(".json").open("r") as f:
            written_data = json.load(f)
        self.assertDictEqual(written_data, new_data)
        self.assertTrue(self.file_path.with_suffix(".json.bak").is_file())
        with self.file_path.with_suffix(".json.bak").open("r") as f:
            backup_data = json.load(f)
        self.assertDictEqual(backup_data, initial_data)

    def test_backup_and_overwrite_json_file_symlink(self):
        """
        Test the function with a path to an existing symlink, checking that the symlink is converted to a file and the data is overwritten.
        """
        initial_data = {"a": 1, "b": 2}
        symlink_path = Path(self.temp_dir.name) / "test_symlink.json"
        with self.file_path.open("w") as f:
            json.dump(initial_data, f)
        os.symlink(self.file_path, symlink_path)
        new_data = {"c": 3, "d": 4}
        backup_and_overwrite_json_file(new_data, symlink_path)
        self.assertFalse(symlink_path.is_symlink())
        with symlink_path.open("r") as f:
            written_data = json.load(f)
        self.assertDictEqual(written_data, new_data)

    def test_backup_and_overwrite_json_file_invalid_input(self):
        """
        Test the function with invalid input (not a Path object), checking that a TypeError is raised.
        """
        initial_data = {"a": 1, "b": 2}
        with self.assertRaises(TypeError) as cm:
            backup_and_overwrite_json_file(initial_data, "invalid_path.json")
        error_msg = str(cm.exception)
        expected_error_msg = f"'invalid_path.json' must be a '{type(Path('.'))}'."
        self.assertEqual(error_msg, expected_error_msg)


class TestLoadDefaultJsonFile(unittest.TestCase):
    """
    Test case for the 'load_default_json_file' function.

    Methods
    -------
    test_load_default_json_file():
        Test the function when the default JSON file exists and contains valid data.
    test_load_default_json_file_empty_file():
        Test the function when the default JSON file is empty and verify that an empty dictionary is returned.
    test_load_default_json_file_file_not_found():
        Test the function when the default JSON file is not found.
    test_load_default_json_file_invalid_input():
        Test the function with invalid input (not a Path object) when loading a JSON file, and verify that a TypeError is raised.
    """

    def setUp(self):
        self.file_content = {"key1": "value1", "key2": 10}
        self.file_content_empty = ""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.tmp_dir.name)
        self.temp_file = self.temp_dir_path / "test.json"
        with self.temp_file.open(mode="w") as f:
            json.dump(self.file_content, f)
        self.temp_empty_file = self.temp_dir_path / "test_empty.json"
        with self.temp_empty_file.open(mode="w") as f:
            f.write(self.file_content_empty)
        self.temp_fake_file = self.temp_dir_path / "test_fake.json"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_default_json_file(self):
        """
        Test the function when the default JSON file exists and contains valid data.
        """
        output = load_default_json_file(self.temp_file)
        self.assertDictEqual(output, self.file_content)

    def test_load_default_json_file_empty_file(self):
        """
        Test the function when the default JSON file is empty and verify that an empty dictionary is returned.
        """
        output = load_default_json_file(self.temp_empty_file)
        self.assertDictEqual(output, {})

    def test_load_default_json_file_file_not_found(self):
        """
        Test the function when the default JSON file is not found.
        """
        output = load_default_json_file(self.temp_fake_file)
        self.assertDictEqual(output, {})
        self.assertLogs(level="WARNING")

    def test_load_default_json_file_invalid_input(self):
        """
        Test the function with invalid input (not a Path object) when loading a JSON file, and verify that a TypeError is raised.
        """
        with self.assertRaises(TypeError) as cm:
            load_default_json_file("invalid_path.json")
        error_msg = str(cm.exception)
        expected_error_msg = f"'invalid_path.json' must be a '{type(Path('.'))}'."
        self.assertEqual(error_msg, expected_error_msg)


class TestLoadJsonFile(unittest.TestCase):
    """
    Test case for the 'load_json_file' function.

    Methods
    -------
    test_load_existing_json_file():
        Test the function with an existing JSON file and ensure the loaded data is correct.
    test_load_nonexistent_json_file_with_abort_on_error():
        Test the function when the JSON file is not found with 'abort_on_error=True', and check that a FileNotFoundError is raised.
    test_load_nonexistent_json_file_without_abort_on_error():
        Test the function when the JSON file is not found with 'abort_on_error=False', and verify that an empty dictionary is returned.
    test_load_empty_json_file():
        Test the function when loading an empty JSON file and verify that an empty dictionary is returned.
    test_load_json_file_invalid_input():
        Test the function with invalid input (not a Path object) when loading a JSON file, and check that a TypeError is raised.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.json_data = {"key1": "value1", "key2": "value2"}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_existing_json_file(self):
        """
        Test the function with an existing JSON file and ensure the loaded data is correct.
        """
        file_path = Path(self.temp_dir.name) / "test.json"
        write_json_file(self.json_data, file_path)

        loaded_data = load_json_file(file_path)
        self.assertDictEqual(loaded_data, self.json_data)

    def test_load_nonexistent_json_file_with_abort_on_error(self):
        """
        Test the function when the JSON file is not found with 'abort_on_error=True', and check that a FileNotFoundError is raised.
        """
        file_path = Path(self.temp_dir.name) / "nonexistent.json"
        with self.assertRaises(FileNotFoundError) as cm:
            load_json_file(file_path)
        error_msg = str(cm.exception)
        expected_error_msg = f"File '{file_path.name}' not found in '{file_path.parent}'."
        self.assertEqual(error_msg, expected_error_msg)

    def test_load_nonexistent_json_file_without_abort_on_error(self):
        """
        Test the function when the JSON file is not found with 'abort_on_error=False', and verify that an empty dictionary is returned.
        """
        file_path = Path(self.temp_dir.name) / "nonexistent.json"
        loaded_data = load_json_file(file_path, abort_on_error=False)
        self.assertDictEqual(loaded_data, {})

    def test_load_empty_json_file(self):
        """
        Test the function when loading an empty JSON file and verify that an empty dictionary is returned.
        """
        file_path = Path(self.temp_dir.name) / "test.json"
        with file_path.open("w", encoding="UTF-8") as json_file:
            json_file.write("")
        loaded_data = load_json_file(file_path, False)
        self.assertDictEqual(loaded_data, {})

    def test_load_json_file_invalid_input(self):
        """
        Test the function with invalid input (not a Path object) when loading a JSON file, and check that a TypeError is raised.
        """
        with self.assertRaises(TypeError) as cm:
            load_json_file("invalid_path.json")
        error_msg = str(cm.exception)
        expected_error_msg = f"'invalid_path.json' must be a '{type(Path('.'))}'."
        self.assertEqual(error_msg, expected_error_msg)


class TestWriteJsonFile(unittest.TestCase):
    """
    Test case for the 'write_json_file' function.

    Methods
    -------
    test_write_json_file():
        Test the function with valid arguments and checks that it writes the JSON file correctly.
    test_write_json_file_with_enable_logging():
        Test the function with valid arguments and checks that it writes the JSON file correctly and enables logging.
    test_write_json_file_invalid_input:
        Test the function with with invalid input (not a Path object) and check that a TypeError is raised.
    test_write_json_file_with_ioerror():
        Test the function with invalid file permissions and checks that it raises an IOError exception and does not write the JSON file.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "test.json"
        self.json_data = {"key1": "value1", "key2": "value2"}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_json_file(self):
        """
        Test the function with valid arguments and check that it writes the JSON file correctly.
        """
        write_json_file(self.json_data, self.file_path)
        self.assertTrue(self.file_path.is_file())
        with self.file_path.open("r", encoding="UTF-8") as f:
            written_data = json.load(f)
        self.assertDictEqual(written_data, self.json_data)

    def test_write_json_file_with_enable_logging(self):
        """
        Test the function with valid arguments and checks that it writes the JSON file correctly and enables logging.
        """
        write_json_file(self.json_data, self.file_path, enable_logging=True)
        self.assertTrue(self.file_path.is_file())
        with self.file_path.open("r", encoding="UTF-8") as f:
            written_data = json.load(f)
        self.assertDictEqual(written_data, self.json_data)

    def test_write_json_file_invalid_input(self):
        """
        Test the function with with invalid input (not a Path object) and check that a TypeError is raised.
        """
        with self.assertRaises(TypeError) as cm:
            write_json_file(self.json_data, "invalid_path.json")
        error_msg = str(cm.exception)
        expected_error_msg = f"'invalid_path.json' must be a '{type(Path('.'))}'."
        self.assertEqual(error_msg, expected_error_msg)

    def test_write_json_file_with_ioerror(self):
        """
        Test the function with invalid file permissions and checks that it raises an IOError exception and does not write the JSON file.
        """
        os.chmod(self.temp_dir.name, 0o500)
        with self.assertRaises(Exception):
            write_json_file(self.json_data, self.file_path)
        self.assertFalse(self.file_path.is_file())


if __name__ == "__main__":
    unittest.main()
