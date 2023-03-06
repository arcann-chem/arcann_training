from pathlib import Path
import unittest
import tempfile
import os
import json

# deepmd_iterative imports
from deepmd_iterative.common.json import (
    load_json_file,
    load_default_json_file,
    write_json_file,
    backup_and_overwrite_json_file,
    add_key_value_to_dict,
)


class TestLoadJsonFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.json_data = {"key1": "value1", "key2": "value2"}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_existing_json_file(self):
        # Create a test JSON file
        file_path = Path(self.temp_dir.name) / "test.json"
        write_json_file(self.json_data, file_path)

        # Test loading the file
        loaded_data = load_json_file(file_path)
        self.assertEqual(loaded_data, self.json_data)

    def test_load_nonexistent_json_file_with_abort_on_error(self):
        # Create a test JSON file path
        file_path = Path(self.temp_dir.name) / "nonexistent.json"

        # Test loading the file with abort_on_error=True
        with self.assertRaises(SystemExit) as cm:
            load_json_file(file_path)
        self.assertEqual(cm.exception.code, 2)

    def test_load_nonexistent_json_file_without_abort_on_error(self):
        # Create a test JSON file path
        file_path = Path(self.temp_dir.name) / "nonexistent.json"

        # Test loading the file with abort_on_error=False
        loaded_data = load_json_file(file_path, abort_on_error=False)
        self.assertEqual(loaded_data, {})

    def test_load_empty_json_file(self):
        # Create a test JSON file
        file_path = Path(self.temp_dir.name) / "test.json"
        with file_path.open("w", encoding="UTF-8") as json_file:
            json_file.write("")

        # Test loading the file
        loaded_data = load_json_file(file_path, False)
        self.assertEqual(loaded_data, {})


class TestLoadDefaultJsonFile(unittest.TestCase):
    def test_load_existing_json_file(self):
        # Create a temporary directory and file
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.json"
            json_data = {"foo": "bar"}
            with file_path.open("w", encoding="UTF-8") as json_file:
                json.dump(json_data, json_file)

            # Test loading the file
            loaded_data = load_default_json_file(file_path)

            self.assertEqual(loaded_data, json_data)

    def test_load_empty_json_file(self):
        # Create a temporary directory and file
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.json"
            with file_path.open("w", encoding="UTF-8") as json_file:
                json_file.write("")

            # Test loading the file
            loaded_data = load_default_json_file(file_path)

            self.assertEqual(loaded_data, {})

    def test_load_nonexistent_json_file(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.json"

            # Test loading the file
            loaded_data = load_default_json_file(file_path)

            self.assertEqual(loaded_data, {})
            self.assertLogs(level="WARNING")


class TestWriteJsonFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "test.json"
        self.json_data = {"key1": "value1", "key2": "value2"}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_json_file(self):

        # Write the test data to a JSON file
        write_json_file(self.json_data, self.file_path)

        # Verify that the file was written and its contents match the test data
        self.assertTrue(self.file_path.is_file())
        with self.file_path.open("r", encoding="UTF-8") as f:
            written_data = json.load(f)
        self.assertEqual(written_data, self.json_data)

    def test_write_json_file_with_enable_logging(self):
        # Define test data to be written to the JSON file

        # Write the test data to a JSON file with logging enabled
        write_json_file(self.json_data, self.file_path, enable_logging=True)

        # Verify that the file was written and its contents match the test data
        self.assertTrue(self.file_path.is_file())
        with self.file_path.open("r", encoding="UTF-8") as f:
            written_data = json.load(f)
        self.assertEqual(written_data, self.json_data)

    def test_write_json_file_with_ioerror(self):
        # Define test data to be written to the JSON file
        # Remove write permission from the temporary directory
        os.chmod(self.temp_dir.name, 0o500)

        # Verify that an IOError is raised when attempting to write to the file
        with self.assertRaises(SystemExit) as cm:
            write_json_file(self.json_data, self.file_path)
        self.assertEqual(cm.exception.code, 2)

        # Verify that the file was not written
        self.assertFalse(self.file_path.is_file())


class TestBackupAndOverwriteJsonFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "test.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_backup_and_write(self):
        # Create an initial file
        initial_data = {"a": 1, "b": 2}
        with self.file_path.open("w") as f:
            json.dump(initial_data, f)

        # Write new data to the file, creating a backup of the old data
        new_data = {"c": 3, "d": 4}
        backup_and_overwrite_json_file(new_data, self.file_path)

        # Verify that the original file was backed up and the new data was written
        self.assertTrue(self.file_path.with_suffix(".json").is_file())
        with self.file_path.with_suffix(".json").open("r") as f:
            written_data = json.load(f)
        self.assertEqual(written_data, new_data)

        # Verify that the original file was backed up and the new data was written
        self.assertTrue(self.file_path.with_suffix(".json.bak").is_file())
        with self.file_path.with_suffix(".json.bak").open("r") as f:
            backup_data = json.load(f)
        self.assertEqual(backup_data, initial_data)

    def test_symbolic_link_removal_and_write(self):

        # Create a symbolic link to a file
        initial_data = {"a": 1, "b": 2}
        symlink_path = Path(self.temp_dir.name) / "test_symlink.json"
        with self.file_path.open("w") as f:
            json.dump(initial_data, f)
        os.symlink(self.file_path, symlink_path)

        # Write new data to the linked file, removing the symbolic link
        new_data = {"c": 3, "d": 4}
        backup_and_overwrite_json_file(new_data, symlink_path)

        # Verify that the symbolic link was removed and the new data was written
        self.assertFalse(symlink_path.is_symlink())
        with symlink_path.open("r") as f:
            written_data = json.load(f)
        self.assertEqual(written_data, new_data)


class TestAddKeyValueToDict(unittest.TestCase):
    def test_add_to_empty_dict(self):
        d = {}
        add_key_value_to_dict(d, "key1", "value1")
        self.assertEqual(d, {"key1": {"value": "value1"}})

    def test_add_new_key_to_dict(self):
        d = {"key1": {"value": "value1"}}
        add_key_value_to_dict(d, "key2", "value2")
        self.assertEqual(d, {"key1": {"value": "value1"}, "key2": {"value": "value2"}})

    def test_update_existing_key_in_dict(self):
        d = {"key1": {"value": "value1"}}
        add_key_value_to_dict(d, "key1", "new_value1")
        self.assertEqual(d, {"key1": {"value": "new_value1"}})

    def test_add_integer_value_to_dict(self):
        d = {}
        add_key_value_to_dict(d, "key1", 123)
        self.assertEqual(d, {"key1": {"value": 123}})

    def test_add_dict_value_to_dict(self):
        d = {"key1": {"value": "value1"}}
        value = {"key2": "value2"}
        add_key_value_to_dict(d, "key2", value)
        self.assertEqual(d, {"key1": {"value": "value1"}, "key2": {"value": value}})

    def test_add_list_value_to_dict(self):
        d = {"key1": {"value": "value1"}}
        value = ["item1", "item2"]
        add_key_value_to_dict(d, "key2", value)
        self.assertEqual(d, {"key1": {"value": "value1"}, "key2": {"value": value}})

    def test_add_nested_dict_to_dict(self):
        d = {"key1": {"value": "value1"}}
        nested_dict = {"key2": {"key3": "value3"}}
        add_key_value_to_dict(d, "key2", nested_dict)
        self.assertEqual(
            d, {"key1": {"value": "value1"}, "key2": {"value": nested_dict}}
        )

if __name__ == '__main__':
    unittest.main()