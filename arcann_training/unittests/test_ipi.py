"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/14

Test cases for the xml module.

Classes
-------
TestGetTemperatureFromIpiXml():
    Test case for the 'get_temperature_from_ipi_xml' function.
"""

# Standard library modules
import unittest
import tempfile
from pathlib import Path

# Local imports
from arcann_training.common.ipi import (
    get_temperature_from_ipi_xml,
)


class TestGetTemperatureFromIpiXml(unittest.TestCase):
    """
    Test case for the 'get_temperature_from_ipi_xml' function.

    Methods
    -------
    test_get_temperature_from_ipi_xml():
        Test whether the function returns the correct temperature from a valid i-PI input file.
    test_get_temperature_from_ipi_xml_no_temperature():
        Test whether the function raises a ValueError if the temperature is not found in the input file.
    test_get_temperature_from_ipi_xml_invalid_file():
        Test whether the function raises an exception when attempting to read an invalid or non-existent file.
    test_get_temperature_from_ipi_xml_parse_error():
        Test whether the function raises an exception when encountering a parse error in the input file.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.example_input_file = Path(self.temp_dir.name) / "input.xml"
        with self.example_input_file.open("w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write("<input>\n")
            f.write("  <simulation>\n")
            f.write("    <temperature>300.0</temperature>\n")
            f.write("  </simulation>\n")
            f.write("</input>\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_temperature_from_ipi_xml(self):
        """
        Test whether the function returns the correct temperature from a valid i-PI input file.
        """
        temperature = get_temperature_from_ipi_xml(self.example_input_file)
        self.assertEqual(temperature, 300.0)

    def test_get_temperature_from_ipi_xml_no_temperature(self):
        """
        Test whether the function raises a ValueError if the temperature is not found in the input file.
        """
        no_temp_input_file = Path(self.temp_dir.name) / "no_temp_input.xml"
        with no_temp_input_file.open("w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write("<input>\n")
            f.write("  <simulation>\n")
            f.write("  </simulation>\n")
            f.write("</input>\n")
        with self.assertRaises(ValueError):
            get_temperature_from_ipi_xml(no_temp_input_file)

    def test_get_temperature_from_ipi_xml_invalid_file(self):
        """
        Test whether the function raises an exception when attempting to read an invalid or non-existent file.
        """
        with self.assertRaises(Exception):
            get_temperature_from_ipi_xml(Path("nonexistent.xml"))

    def test_get_temperature_from_ipi_xml_parse_error(self):
        """
        Test whether the function raises an exception when encountering a parse error in the input file.
        """
        invalid_xml_input_file = Path(self.temp_dir.name) / "invalid_xml_input.xml"
        with invalid_xml_input_file.open("w") as f:
            f.write("<input>\n")
            f.write("  <simulation>\n")
            f.write("    <temperature>300.0</temperature>\n")
            f.write("  </simulation>\n")
            f.write("<input>\n")
        with self.assertRaises(Exception):
            get_temperature_from_ipi_xml(invalid_xml_input_file)


if __name__ == "__main__":
    unittest.main()
