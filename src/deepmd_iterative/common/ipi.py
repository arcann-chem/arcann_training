import xml.etree.ElementTree as ET

def get_temperature_from_ipi_xml(input_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    temperature = None
    for child in root.iter():
        if 'temperature' in child.tag:
            temperature = float(child.text)

    return temperature