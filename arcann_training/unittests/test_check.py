"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15

Test case for the check module.

Classes
-------
TestCheckAtomsk():
    Test case for the 'check_atomsk' function.

TestCheckVMD():
    Test case for the 'check_vmd' function.

TestValidateStepFolder():
    Test case for the 'validate_step_folder' function.
"""

# Standard library modules
import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Local imports
from arcann_training.common.check import (
    check_atomsk,
    check_vmd,
    validate_step_folder,
)


class TestCheckAtomsk(unittest.TestCase):
    """
    Test case for the 'check_atomsk' function.

    Methods
    -------
    test_system_path():
        Test that 'check_atomsk' finds atomsk in the system path and returns the full path.

    test_atomsk_path():
        Test that 'check_atomsk' finds atomsk at a specified path and returns the full path.

    test_invalid_path():
        Test that 'check_atomsk' logs a warning for an invalid path.

    test_invalid_env_var():
        Test that 'check_atomsk' ignores an invalid ATOMSK_PATH environment variable.

    test_env_var():
        Test that 'check_atomsk' finds atomsk at an environment variable-specified path and returns the full path.
    """

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        vmd_file = Path(self.tempdir) / "atomsk"
        vmd_file.touch()
        vmd_file.chmod(0o755)
        path_separator = ";" if os.name == "nt" else ":"
        os.environ["PATH"] = f"{self.tempdir}{path_separator}{os.environ['PATH']}"

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    @patch("shutil.which")
    def test_system_path(self, mock_which):
        """
        Test that 'check_atomsk' finds atomsk in the system path and returns the full path.
        """
        mock_which.return_value = "/usr/bin/atomsk"
        atomsk_bin = check_atomsk()
        self.assertEqual(atomsk_bin, str(Path("/usr/bin/atomsk").resolve()))

    def test_atomsk_path(self):
        """
        Test that 'check_atomsk finds atomsk at a specified path and returns the full path.
        """
        atomsk_path = Path(self.tempdir) / "atomsk"
        atomsk_bin = check_atomsk(str(atomsk_path))
        self.assertEqual(atomsk_bin, str(atomsk_path.resolve()))

    def test_invalid_path(self):
        """
        Test that 'check_atomsk' logs a warning for an invalid path.
        """
        invalid_path = "/invalid/path/to/atomsk"
        with self.assertLogs(level=logging.WARNING):
            atomsk_bin = check_atomsk(invalid_path)
            self.assertEqual(atomsk_bin, str(shutil.which("atomsk")))

    def test_invalid_env_var(self):
        """
        Test that 'check_atomsk' ignores an invalid ATOMSK_PATH environment variable.
        """
        os.environ["ATOMSK_PATH"] = "/invalid/path/to/atomsk"
        atomsk_bin = check_atomsk()
        self.assertEqual(atomsk_bin, str(shutil.which("atomsk")))

    def test_env_var(self):
        """
        Test that 'check_atomsk' finds vmd at an environment variable-specified path and returns the full path.
        """
        atomsk_path = Path(self.tempdir) / "atomsk"
        os.environ["ATOMSK_PATH"] = str(atomsk_path)
        atomsk_bin = check_atomsk()
        self.assertEqual(atomsk_bin, str(atomsk_path.resolve()))


class TestCheckVMD(unittest.TestCase):
    """
    Test case for the 'check_vmd' function.

    Methods
    -------
    test_system_path():
        Test that 'check_vmd' finds vmd in the system path and returns the full path.

    test_vmd_path():
        Test that 'check_vmd' finds vmd at a specified path and returns the full path.

    test_invalid_path():
        Test that 'check_vmd' logs a warning for an invalid path.

    test_invalid_env_var():
        Test that 'check_vmd' ignores an invalid VMD_PATH environment variable.

    test_env_var():
        Test that 'check_vmd' finds vmd at an environment variable-specified path and returns the full path.
    """

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        vmd_file = Path(self.tempdir) / "vmd"
        vmd_file.touch()
        vmd_file.chmod(0o755)
        path_separator = ";" if os.name == "nt" else ":"
        os.environ["PATH"] = f"{self.tempdir}{path_separator}{os.environ['PATH']}"

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    @patch("shutil.which")
    def test_system_path(self, mock_which):
        """
        Test that 'check_vmd' finds vmd in the system path and returns the full path.
        """
        mock_which.return_value = "/usr/bin/vmd"
        vmd_bin = check_vmd()
        self.assertEqual(vmd_bin, str(Path("/usr/bin/vmd").resolve()))

    def test_vmd_path(self):
        """
        Test that 'check_vmd' finds vmd at a specified path and returns the full path.
        """
        vmd_path = Path(self.tempdir) / "vmd"
        vmd_bin = check_vmd(str(vmd_path))
        self.assertEqual(vmd_bin, str(vmd_path.resolve()))

    def test_invalid_path(self):
        """
        Test that 'check_vmd' logs a warning for an invalid path.
        """
        invalid_path = "/invalid/path/to/vmd"
        with self.assertLogs(level=logging.WARNING):
            vmd_bin = check_vmd(invalid_path)
            self.assertEqual(vmd_bin, str(shutil.which("vmd")))

    def test_invalid_env_var(self):
        """
        Test that 'check_vmd' ignores an invalid VMD_PATH environment variable.
        """
        os.environ["VMD_PATH"] = "/invalid/path/to/vmd"
        vmd_bin = check_vmd()
        self.assertEqual(vmd_bin, str(shutil.which("vmd")))

    def test_env_var(self):
        """
        Test that 'check_vmd' finds vmd at an environment variable-specified path and returns the full path.
        """
        vmd_path = Path(self.tempdir) / "vmd"
        os.environ["VMD_PATH"] = str(vmd_path)
        vmd_bin = check_vmd()
        self.assertEqual(vmd_bin, str(vmd_path.resolve()))


class TestValidateStepFolder(unittest.TestCase):
    """
    Test case for 'validate_step_folder' function.

    Methods
    -------
    test_validate_step_folder():
        Test that 'validate_step_folder' returns None when the current directory name matches the expected directory for the step.
    test_validate_step_folder_raises_error():
        Test that 'validate_step_folder' raises a ValueError when the current directory name does not contain the step name.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.step_name = "step1"
        self.step_folder = Path(self.temp_dir.name) / self.step_name
        self.step_folder.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_step_folder(self):
        """
        Test that 'validate_step_folder' returns None when the current directory name matches the expected directory for the step.
        """
        os.chdir(self.step_folder)
        with self.step_folder:
            self.assertIsNone(validate_step_folder(self.step_name))

    def test_validate_step_folder_raises_error(self):
        """
        Test that 'validate_step_folder' raises a ValueError when the current directory name does not contain the step name.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            with self.assertRaises(ValueError):
                validate_step_folder(self.step_name)


if __name__ == "__main__":
    unittest.main()
