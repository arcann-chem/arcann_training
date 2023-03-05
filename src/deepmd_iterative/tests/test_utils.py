import unittest

# deepmd_iterative imports
from deepmd_iterative.common.utils import convert_seconds_to_hh_mm_ss


class TestConvertSecondsToHhMmSs(unittest.TestCase):
    def test_convert_seconds_to_hh_mm_ss(self):
        # Test conversion of various time durations in seconds to the HH:MM:SS format
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
