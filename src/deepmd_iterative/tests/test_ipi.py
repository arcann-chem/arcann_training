from pathlib import Path
import xml.etree.ElementTree as ET

# Unittest imports
import unittest
import tempfile

# deepmd_iterative imports
from deepmd_iterative.common.ipi import (
    get_temperature_from_ipi_xml,
)


class TestGetTemperatureFromIpiXml(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        # Create an example IPI input file for testing
        self.example_input_file = Path(self.temp_dir.name) / "input.xml"
        with self.example_input_file.open("w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write("<input>\n")
            f.write("  <simulation>\n")
            f.write("    <temperature>300.0</temperature>\n")
            f.write("  </simulation>\n")
            f.write("</input>\n")

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_get_temperature_from_ipi_xml(self):
        # Call the function on the example input file
        temperature = get_temperature_from_ipi_xml(self.example_input_file)
        # Check that the returned temperature is correct
        self.assertEqual(temperature, 300.0)

    def test_get_temperature_from_ipi_xml_no_temperature(self):
        # Create an example input file with no temperature tag
        no_temp_input_file = Path(self.temp_dir.name) / "no_temp_input.xml"
        with no_temp_input_file.open("w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write("<input>\n")
            f.write("  <simulation>\n")
            f.write("  </simulation>\n")
            f.write("</input>\n")
        # Call the function on the example input file with no temperature tag
        with self.assertRaises(SystemExit):
            get_temperature_from_ipi_xml(no_temp_input_file)

    def test_get_temperature_from_ipi_xml_invalid_file(self):
        # Call the function on a nonexistent file
        with self.assertRaises(SystemExit):
            get_temperature_from_ipi_xml(Path("nonexistent.xml"))

    def test_get_temperature_from_ipi_xml_parse_error(self):
        # Create an example input file with invalid XML
        invalid_xml_input_file = Path(self.temp_dir.name) / "invalid_xml_input.xml"
        with invalid_xml_input_file.open("w") as f:
            f.write("<input>\n")
            f.write("  <simulation>\n")
            f.write("    <temperature>300.0</temperature>\n")
            f.write("  </simulation>\n")
            f.write("<input>\n")
        # Call the function on the example input file with invalid XML
        with self.assertRaises(SystemExit):
            get_temperature_from_ipi_xml(invalid_xml_input_file)


if __name__ == "__main__":
    unittest.main()
