from pathlib import Path
import logging
import os
import shutil

# Unittest imports
import unittest
import tempfile
from unittest.mock import patch

# deepmd_iterative imports
from deepmd_iterative.common.check import (
    check_atomsk,
    check_vmd,
    validate_step_folder,
)


class TestCheckAtomsk(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tempdir = tempfile.mkdtemp()
        # Create a temporary "atomsk" file in the directory
        atomsk_file = Path(self.tempdir) / "atomsk"
        atomsk_file.touch()
        atomsk_file.chmod(0o755)
        # Add the temporary directory to the PATH environment variable
        os.environ["PATH"] = f"{self.tempdir}:{os.environ['PATH']}"

    def tearDown(self):
        # Remove the temporary directory and its contents
        shutil.rmtree(self.tempdir)

    @patch("subprocess.check_output")
    def test_system_path(self, mock_check_output):
        # Test that the function finds atomsk in the system path and returns the full path
        mock_check_output.return_value = b"/usr/bin/atomsk\n"
        atomsk_bin = check_atomsk()
        self.assertEqual(atomsk_bin, str(Path("/usr/bin/atomsk").resolve()))

    def test_atomsk_path(self):
        # Test that the function finds atomsk at a specified path and returns the full path
        atomsk_path = Path(self.tempdir) / "atomsk"
        atomsk_bin = check_atomsk(str(atomsk_path))
        self.assertEqual(atomsk_bin, str(atomsk_path.resolve()))

    def test_invalid_path(self):
        # Test that the function logs a warning for an invalid path
        invalid_path = "/invalid/path/to/atomsk"
        with self.assertLogs(level=logging.WARNING):
            atomsk_bin = check_atomsk(invalid_path)
            self.assertEqual(atomsk_bin, str(Path(shutil.which("atomsk")).resolve()))

    def test_invalid_env_var(self):
        # Test that the function ignores an invalid ATMSK_PATH environment variable
        os.environ["ATMSK_PATH"] = "/invalid/path/to/atomsk"
        atomsk_bin = check_atomsk()
        self.assertEqual(atomsk_bin, str(Path(shutil.which("atomsk")).resolve()))

    def test_env_var(self):
        # Test that the function finds atomsk at an environment variable-specified path and returns the full path
        atomsk_path = Path(self.tempdir) / "atomsk"
        os.environ["ATMSK_PATH"] = str(atomsk_path)
        atomsk_bin = check_atomsk()
        self.assertEqual(atomsk_bin, str(atomsk_path.resolve()))


class TestCheckVMD(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tempdir = tempfile.mkdtemp()
        # Create a temporary "vmd" file in the directory
        vmd_file = Path(self.tempdir) / "vmd"
        vmd_file.touch()
        vmd_file.chmod(0o755)
        # Add the temporary directory to the PATH environment variable
        os.environ["PATH"] = f"{self.tempdir}:{os.environ['PATH']}"

    def tearDown(self):
        # Remove the temporary directory and its contents
        shutil.rmtree(self.tempdir)

    @patch("subprocess.check_output")
    def test_system_path(self, mock_check_output):
        # Test that the function finds vmd in the system path and returns the full path
        mock_check_output.return_value = b"/usr/bin/vmd\n"
        vmd_bin = check_vmd()
        self.assertEqual(vmd_bin, str(Path("/usr/bin/vmd").resolve()))

    def test_vmd_path(self):
        # Test that the function finds vmd at a specified path and returns the full path
        vmd_path = Path(self.tempdir) / "vmd"
        vmd_bin = check_vmd(str(vmd_path))
        self.assertEqual(vmd_bin, str(vmd_path.resolve()))

    def test_invalid_path(self):
        # Test that the function logs a warning for an invalid path
        invalid_path = "/invalid/path/to/vmd"
        with self.assertLogs(level=logging.WARNING):
            vmd_bin = check_vmd(invalid_path)
            self.assertEqual(vmd_bin, str(Path(shutil.which("vmd")).resolve()))

    def test_invalid_env_var(self):
        # Test that the function ignores an invalid VMD_PATH environment variable
        os.environ["VMD_PATH"] = "/invalid/path/to/vmd"
        vmd_bin = check_vmd()
        self.assertEqual(vmd_bin, str(Path(shutil.which("vmd")).resolve()))

    def test_env_var(self):
        # Test that the function finds vmd at an environment variable-specified path and returns the full path
        vmd_path = Path(self.tempdir) / "vmd"
        os.environ["VMD_PATH"] = str(vmd_path)
        vmd_bin = check_vmd()
        self.assertEqual(vmd_bin, str(vmd_path.resolve()))


class TestValidateStepFolder(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.step_name = "step1"
        self.step_folder = Path(self.temp_dir.name) / self.step_name
        self.step_folder.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_step_folder(self):
        with self.step_folder:
            validate_step_folder(self.step_name)

    def test_validate_step_folder_raises_error(self):
        with self.assertRaises(SystemExit) as cm:
            with self.step_folder:
                validate_step_folder("step2")
        self.assertEqual(cm.exception.code, 1)


class TestValidateStepFolder(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.step_name = "step1"
        os.chdir(self.temp_dir.name)
        Path(self.step_name).mkdir()

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


if __name__ == "__main__":
    unittest.main()
