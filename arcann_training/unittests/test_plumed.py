"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2026 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2023/09/04
Last modified: 2026/02/02

Unit tests for the plumed module.

Classes
-------
TestPlumedAnalysis():
    Test case for the 'analyze_plumed_file_for_movres' function.
"""

# Standard library modules
import tempfile
import unittest
from pathlib import Path

# Local imports
from arcann_training.common.plumed import analyze_plumed_file_for_movres
from arcann_training.common.list import textfile_to_string_list


class TestPlumedAnalysis(unittest.TestCase):
    """
    Test case for the 'analyze_plumed_file_for_movres' function.

    Methods
    -------
    test_movres_present_with_step():
        Test checking the function correctly identifies MOVINGRESTRAINT with STEP value.
    test_movres_present_without_step():
        Test checking the function raises ValueError for MOVINGRESTRAINT without STEP value.
    test_movres_not_present():
        Test checking the function correctly identifies absence of MOVINGRESTRAINT.
    """

    def test_movres_present_with_step(self):
        """
        Test checking the function correctly identifies MOVINGRESTRAINT with STEP value.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            plumed_file = Path(tmpdir) / "plumed.dat"

            plumed_lines = "\n".join(
                [
                    "Some lines before",
                    "MOVINGRESTRAINT ...",
                    "STEP1 = 100",
                    "STEP2 = 200",
                    "More lines after",
                ]
            )

            plumed_file.write_text(plumed_lines + "\n")
            plumed_content = textfile_to_string_list(plumed_file)

            result = analyze_plumed_file_for_movres(plumed_content)
            self.assertEqual(result, (True, 200))

    def test_movres_present_without_step(self):
        """
        Test checking the function raises ValueError for MOVINGRESTRAINT without STEP value.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            plumed_file = Path(tmpdir) / "plumed.dat"

            plumed_lines = "\n".join(
                ["Some lines before", "MOVINGRESTRAINT ...", "More lines after"]
            )

            plumed_file.write_text(plumed_lines + "\n")
            plumed_content = textfile_to_string_list(plumed_file)

            with self.assertRaises(ValueError):
                analyze_plumed_file_for_movres(plumed_content)

    def test_movres_not_present(self):
        """
        Test checking the function correctly identifies absence of MOVINGRESTRAINT.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            plumed_file = Path(tmpdir) / "plumed.dat"

            plumed_lines = "\n".join(
                ["Some lines before", "Some other lines", "More lines after"]
            )

            plumed_file.write_text(plumed_lines + "\n")
            plumed_content = textfile_to_string_list(plumed_file)

            result = analyze_plumed_file_for_movres(plumed_content)
            self.assertEqual(result, (False, False))


if __name__ == "__main__":
    unittest.main()
