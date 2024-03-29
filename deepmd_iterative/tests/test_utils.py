"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/03/28

Test case for the utils module.

Classes
-------
TestConvertSecondsToHhMmSs():
    Test case for the 'convert_seconds_to_hh_mm_ss' function.

TestCatchErrorsDecorator():
    Test case for the 'catch_errors_decorator' function.
"""

# Standard library modules
import unittest

# Local imports
from deepmd_iterative.common.utils import (
    convert_seconds_to_hh_mm_ss,
    catch_errors_decorator,
    natural_sort_key,
)


class TestConvertSecondsToHhMmSs(unittest.TestCase):
    """
    Test case for the 'convert_seconds_to_hh_mm_ss' function.

    Methods
    -------
    test_convert_seconds_to_hh_mm_ss():
        Test the conversion of time durations in seconds to the HH:MM:SS format.
    """

    def test_convert_seconds_to_hh_mm_ss(self):
        """Test conversion of various time durations in seconds to the HH:MM:SS format."""
        test_cases = [
            (0, "0:00:00"),
            (1, "0:00:01"),
            (60, "0:01:00"),
            (3600, "1:00:00"),
            (3661, "1:01:01"),
            (86400, "24:00:00"),
        ]
        for seconds, expected_output in test_cases:
            with self.subTest(seconds=seconds, expected_output=expected_output):
                self.assertEqual(convert_seconds_to_hh_mm_ss(seconds), expected_output)


class TestCatchErrorsDecorator(unittest.TestCase):
    """
    Test case for the 'catch_errors_decorator' function.

    Methods
    -------
    test_no_exception():
        Test the decorator behavior when the decorated function runs without exceptions.

    test_exception_raised():
        Test the decorator behavior when the decorated function raises an exception.
    """

    def test_no_exception(self):
        """
        Test the decorator behavior when the decorated function runs without exceptions.
        """

        @catch_errors_decorator
        def func_no_exception():
            return 42

        result = func_no_exception()
        self.assertEqual(result, 42)

    def test_exception_raised(self):
        """Test the decorator behavior when the decorated function raises an exception."""

        @catch_errors_decorator
        def func_with_exception():
            raise ValueError("Test exception")

        with self.assertRaises(ValueError):
            func_with_exception()


class TestNaturalSortKey(unittest.TestCase):
    def test_with_numbers(self):
        self.assertEqual(natural_sort_key("abc123def"), ["abc", 123, "def"])

    def test_without_numbers(self):
        self.assertEqual(natural_sort_key("abcdef"), ["abcdef"])

    def test_empty_string(self):
        self.assertEqual(natural_sort_key(""), [])

    def test_mixed_case_string(self):
        self.assertEqual(natural_sort_key("AbC123DeF"), ["abc", 123, "def"])

    def test_string_with_multiple_numbers(self):
        self.assertEqual(natural_sort_key("abc12def34"), ["abc", 12, "def", 34])

    def test_non_string_input(self):
        with self.assertRaises(TypeError):
            natural_sort_key(123)


if __name__ == "__main__":
    unittest.main()
