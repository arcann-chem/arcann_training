import logging
import sys
import xml.etree.ElementTree as ET


#Unittested
def get_temperature_from_ipi_xml(input_file: ET.ElementTree):
    """
    Extract the temperature value from an XML file and return it as a float.

    Args:
        input_file: A string representing the file path of the input XML file.

    Returns:
        A float representing the temperature value in the XML file.

    """
    try:
        tree = ET.parse(input_file)
    except (IOError,ET.ParseError) as e:
        error_msg=f"Error reading input file {input_file}: {e}"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)
    root = tree.getroot()

    temperature = None
    for child in root.iter():
        if "temperature" in child.tag:
            try:
                temperature = float(child.text)
            except ValueError as e:
                error_msg=f"Error parsing temperature value in {input_file}: {e}"
                logging.error(f"{error_msg}\nAborting...")
                sys.exit(1)

    if temperature is None:
        error_msg=f"Temperature value not found in {input_file}"
        logging.error(f"{error_msg}\nAborting...")
        sys.exit(1)

    return temperature
