"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2023/09/04
Last modified: 2023/09/04

Test cases for the plumed module.

Classes
-------
TestPlumedAnalysis
    Test cases for the 'analyze_plumed_file_for_movres' function.
"""

# Standard library modules
import unittest

# Local imports
from deepmd_iterative.common.plumed import analyze_plumed_file_for_movres


class TestPlumedAnalysis(unittest.TestCase):
    """
    Test cases for the 'analyze_plumed_file_for_movres' function.

    Methods
    -------
    test_movres_present_with_step():
        Test if the function correctly identifies MOVINGRESTRAINT with STEP value.

    test_movres_present_without_step():
        Test if the function raises ValueError for MOVINGRESTRAINT without STEP value.

    test_movres_not_present():
        Test if the function correctly identifies absence of MOVINGRESTRAINT.
    """

    def test_movres_present_with_step(self):
        """
        Test if the function correctly identifies MOVINGRESTRAINT with STEP value.
        """
        plumed_lines = [
            "Some lines before",
            "MOVINGRESTRAINT ...",
            "STEP1 = 100",
            "STEP2 = 200",
            "More lines after",
        ]
        result = analyze_plumed_file_for_movres(plumed_lines)
        self.assertEqual(result, (True, 200))

    def test_movres_present_without_step(self):
        """
        Test if the function raises ValueError for MOVINGRESTRAINT without STEP value.
        """
        plumed_lines = ["Some lines before", "MOVINGRESTRAINT ...", "More lines after"]
        with self.assertRaises(ValueError):
            analyze_plumed_file_for_movres(plumed_lines)

    def test_movres_not_present(self):
        """
        Test if the function correctly identifies absence of MOVINGRESTRAINT.
        """
        plumed_lines = ["Some lines before", "Some other lines", "More lines after"]
        result = analyze_plumed_file_for_movres(plumed_lines)
        self.assertEqual(result, (False, False))


if __name__ == "__main__":
    unittest.main()
