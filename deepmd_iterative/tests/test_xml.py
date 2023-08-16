"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2023/08/16

Test cases for the xml module.

Class
-----
TestStringListToXml
    Test case for the string_list_to_xml() function.

TestXmlToStringList
    Test case for the xml_to_string_list() function.

TestReadXmlFile
    Test case for the read_xml_file() function.

TestWriteXmlFile
    Test case for the write_xml_file() function.
"""
# Standard library modules
import unittest
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

# Local imports
from deepmd_iterative.common.xml import (
    string_list_to_xml,
    xml_to_string_list,
    read_xml_file,
    write_xml_file,
)


class TestStringListToXml(unittest.TestCase):
    """
    Test case for the string_list_to_xml() function.

    Methods
    -------
    test_string_list_to_xml():
        Test that the function correctly converts a list of strings to a XML tree.
    """

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

    def test_string_list_to_xml(self):
        lines = xml_to_string_list(self.xml_tree)
        tree = string_list_to_xml(lines)
        self.assertIsInstance(tree, ET.ElementTree)
        self.assertEqual(ET.tostring(tree.getroot()), self.expected_xml_string)


class TestXmlToStringList(unittest.TestCase):
    """
    Test case for the xml_to_string_list() function.

    Methods
    -------
    test_xml_to_string_list():
        Test that the function correctly converts a XML tree to a list of strings.
    """

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

    def test_xml_to_string_list(self):
        lines = xml_to_string_list(self.xml_tree)
        self.assertListEqual(lines, self.expected_lines_no_spaces)


class TestReadXmlFile(unittest.TestCase):
    """
    Test case for the read_xml_file() function.

    Methods
    -------
    test_file_not_found():
        Test that a FileNotFoundError is raised when trying to parse a non-existent file.
    test_parse_error():
        Test that an ET.ParseError is raised when trying to parse a file with a syntax error.
    test_valid_file():
        Test that a valid XML file is parsed correctly and has the expected structure.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_not_found(self):
        xml_file_path = Path(self.temp_dir.name) / "nonexistent.xml"

        with self.assertRaises(FileNotFoundError) as cm:
            read_xml_file(xml_file_path)
        error_msg = str(cm.exception)
        expected_error_msg = (
            f"File not found {xml_file_path.name} not in {xml_file_path.parent}"
        )
        self.assertEqual(error_msg, expected_error_msg)

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
        with self.assertRaises(ET.ParseError) as cm:
            read_xml_file(xml_file_path)
        error_msg = str(cm.exception)
        expected_error_msg = f"Failed to parse XML file: {xml_file_path.name}"
        self.assertEqual(error_msg, expected_error_msg)

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

        # Parse the valid XML file using read_xml_file
        xml_tree = read_xml_file(xml_file_path)

        # Check that the parsed XML tree has the expected structure
        root = xml_tree.getroot()
        self.assertEqual(root.tag, "root")
        self.assertEqual(len(root), 2)
        self.assertEqual(root[0].tag, "element")
        self.assertEqual(root[1].tag, "nested")
        self.assertEqual(len(root[1]), 1)
        self.assertEqual(root[1][0].tag, "subelement")


class TestWriteXmlFile(unittest.TestCase):
    """
    Test case for the write_xml_file() function.

    Methods
    -------
    test_write_xml_file():
        Test that a valid XML file is writtend correctly and has the expected structure.
    """

    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.xml_tree = ET.ElementTree(
            ET.fromstring("<root><child1>value1</child1><child2>value2</child2></root>")
        )
        self.expected_xml_string = minidom.parseString(
            ET.tostring(self.xml_tree.getroot())
        ).toprettyxml(indent=" ")
        self.tmp_file_path = Path(self.tmp_file.name)

    def tearDown(self):
        Path.unlink(self.tmp_file_path)

    def test_write_xml_file(self):
        write_xml_file(self.xml_tree, self.tmp_file_path)
        with self.tmp_file_path.open("r") as f:
            file_contents = f.read()
        self.assertEqual(file_contents, self.expected_xml_string)


if __name__ == "__main__":
    unittest.main()
