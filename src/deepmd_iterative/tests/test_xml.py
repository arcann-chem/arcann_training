from pathlib import Path
import unittest
import tempfile
import xml.etree.ElementTree as ET
from xml.dom import minidom

# deepmd_iterative imports
from deepmd_iterative.common.xml import (
    parse_xml_file,
    convert_xml_to_list_of_strings,
    convert_list_of_strings_to_xml,
    write_xml,
)


class TestParseXMLFile(unittest.TestCase):
    def test_file_not_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            xml_file_path = Path(temp_dir) / "nonexistent.xml"
            with self.assertRaises(SystemExit) as cm:
                parse_xml_file(xml_file_path)
            self.assertEqual(cm.exception.code, 2)

    def test_parse_error(self):
        malformed_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <root>
              <element>value</element>
              <missing_end_tag>
                <nested>value</nested>
            </root>
            """

        with tempfile.TemporaryDirectory() as temp_dir:
            xml_file_path = Path(temp_dir) / "malformed.xml"
            with xml_file_path.open("w", encoding="UTF-8") as f:
                f.write(malformed_xml)
            with self.assertRaises(SystemExit) as cm:
                parse_xml_file(xml_file_path)
            self.assertEqual(cm.exception.code, 1)

    def test_valid_file(self):
        valid_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <root>
              <element>value</element>
              <nested>
                <subelement>value</subelement>
              </nested>
            </root>
            """
        with tempfile.TemporaryDirectory() as temp_dir:
            xml_file_path = Path(temp_dir) / "valid.xml"
            with xml_file_path.open("w", encoding="UTF-8") as f:
                f.write(valid_xml)
            xml_tree = parse_xml_file(xml_file_path)
            self.assertIsInstance(xml_tree, ET.ElementTree)
            self.assertEqual(xml_tree.getroot().tag, "root")


class TestXmlToStrings(unittest.TestCase):
    def setUp(self):
        self.xml_string = """<root>
                                <child1>value1</child1>
                                <child2>value2</child2>
                                <child3>value3</child3>
                            </root>"""
        self.xml_tree = ET.ElementTree(ET.fromstring(self.xml_string))

    def test_returns_list(self):
        result = convert_xml_to_list_of_strings(self.xml_tree)
        self.assertIsInstance(result, list)

    def test_returns_correct_number_of_lines(self):
        result = convert_xml_to_list_of_strings(self.xml_tree)
        self.assertEqual(len(result), 5)

    def test_each_line_is_single_node(self):
        result = convert_xml_to_list_of_strings(self.xml_tree)
        self.assertEqual(result[0], "<root>")
        self.assertEqual(result[1], "<child1>value1</child1>")
        self.assertEqual(result[2], "<child2>value2</child2>")
        self.assertEqual(result[3], "<child3>value3</child3>")


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
