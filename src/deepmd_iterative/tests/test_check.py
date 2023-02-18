import unittest
import tempfile
import os
from pathlib import Path

from deepmd_iterative.common.check import validate_step_folder


class TestValidateStepFolder(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.step_name = "step1"
        os.chdir(self.temp_dir.name)
        os.mkdir(self.step_name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_step_folder(self):
        os.chdir(self.temp_dir.name)
        os.chdir(self.step_name)
        validate_step_folder(self.step_name)

    def test_validate_step_folder_raises_error(self):
        os.chdir(self.temp_dir.name)
        os.chdir(self.step_name)
        with self.assertRaises(SystemExit) as cm:
            validate_step_folder("step2")
        self.assertEqual(cm.exception.code, 1)
