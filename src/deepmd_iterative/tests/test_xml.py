from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Unittest
import unittest
import tempfile

# deepmd_iterative imports
from deepmd_iterative.common.xml import (
    convert_list_of_strings_to_xml,
    convert_xml_to_list_of_strings,
    parse_xml_file,
    write_xml,
)


class TestConvertListOfStringsToXml(unittest.TestCase):
    def setUp(self):
        self.xml_string = (
            "<root>\n  <child1>value1</child1>\n  <child2>value2</child2>\n</root>"
        )
        self.xml_tree = ET.ElementTree(ET.fromstring(self.xml_string))
        self.expected_lines = [
            "<root>",
            "  <child1>value1</child1>",
            "  <child2>value2</child2>",
            "</root>",
        ]
        self.expected_xml_string = (
            b"<root><child1>value1</child1><child2>value2</child2></root>"
        )

    def tearDown(self):
        pass

    def test_convert_list_of_strings_to_xml(self):
        lines = convert_xml_to_list_of_strings(self.xml_tree)
        tree = convert_list_of_strings_to_xml(lines)
        self.assertIsInstance(tree, ET.ElementTree)
        self.assertEqual(ET.tostring(tree.getroot()), self.expected_xml_string)


class TestConvertXmlToListOfStrings(unittest.TestCase):
    def setUp(self):
        self.xml_string = (
            "<root>\n  <child1>value1</child1>\n  <child2>value2</child2>\n</root>"
        )
        self.xml_tree = ET.ElementTree(ET.fromstring(self.xml_string))
        self.expected_lines_no_spaces = [
            "<root>",
            "<child1>value1</child1>",
            "<child2>value2</child2>",
            "</root>",
        ]

    def tearDown(self):
        pass

    def test_convert_xml_to_list_of_strings(self):
        lines = convert_xml_to_list_of_strings(self.xml_tree)
        self.assertListEqual(lines, self.expected_lines_no_spaces)


class TestParseXmlFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Remove the temporary directory and its contents
        self.temp_dir.cleanup()

    def test_file_not_found(self):
        # Try to parse a file that doesn't exist
        xml_file_path = Path(self.temp_dir.name) / "nonexistent.xml"
        with self.assertRaises(SystemExit) as context:
            parse_xml_file(xml_file_path)
        # Check that a FileNotFoundError was raised
        self.assertEqual(context.exception.code, 2)

    def test_parse_error(self):
        # Create a test XML string with a syntax error
        malformed_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <root>
              <element>value</element>
              <missing_end_tag>
                <nested>value</nested>
            </root>
            """

        # Write the malformed XML string to a temporary file
        xml_file_path = Path(self.temp_dir.name) / "malformed.xml"
        with xml_file_path.open("w", encoding="UTF-8") as f:
            f.write(malformed_xml)

        # Try to parse the malformed XML file
        with self.assertRaises(SystemExit) as context:
            parse_xml_file(xml_file_path)
        # Check that an ET.ParseError was raised
        self.assertEqual(context.exception.code, 1)

    def test_valid_file(self):
        # Create a test XML string with a valid structure
        valid_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <root>
              <element>value</element>
              <nested>
                <subelement>value</subelement>
              </nested>
            </root>
            """

        # Write the valid XML string to a temporary file
        xml_file_path = Path(self.temp_dir.name) / "valid.xml"
        with xml_file_path.open("w", encoding="UTF-8") as f:
            f.write(valid_xml)

        # Parse the valid XML file using parse_xml_file
        xml_tree = parse_xml_file(xml_file_path)

        # Check that the parsed XML tree has the expected structure
        root = xml_tree.getroot()
        self.assertEqual(root.tag, "root")
        self.assertEqual(len(root), 2)
        self.assertEqual(root[0].tag, "element")
        self.assertEqual(root[1].tag, "nested")
        self.assertEqual(len(root[1]), 1)
        self.assertEqual(root[1][0].tag, "subelement")


class TestWriteXml(unittest.TestCase):
    def setUp(self):
        # Create a temporary file and a sample XML tree to write to it.
        self.tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.xml_tree = ET.ElementTree(
            ET.fromstring("<root><child1>value1</child1><child2>value2</child2></root>")
        )
        self.expected_xml_string = minidom.parseString(
            ET.tostring(self.xml_tree.getroot())
        ).toprettyxml(indent=" ")
        self.tmp_file_path = Path(self.tmp_file.name)

    def tearDown(self):
        # Remove the temporary file.
        Path.unlink(self.tmp_file_path)

    def test_write_xml(self):
        # Call the write_xml() function and assert that the file was written correctly.
        write_xml(self.xml_tree, self.tmp_file_path)
        with self.tmp_file_path.open("r") as f:
            file_contents = f.read()
        self.assertEqual(file_contents, self.expected_xml_string)


if __name__ == "__main__":
    unittest.main()
