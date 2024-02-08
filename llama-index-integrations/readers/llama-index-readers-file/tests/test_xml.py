import xml.etree.ElementTree as ET

import pytest
from llama_index.readers.file.xml import XMLReader

# Sample XML data for testing
SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<data>
    <item type="fruit">
        <name>Apple</name>
        <color>Red</color>
        <price>1.20</price>
    </item>
    <item type="vegetable">
        <name>Carrot</name>
        <color>Orange</color>
        <price>0.50</price>
    </item>
    <item type="fruit">
        <name>Banana</name>
        <color>Yellow</color>
        <price>0.30</price>
    </item>
    <company>
        <name>Fresh Produce Ltd.</name>
        <address>
            <street>123 Green Lane</street>
            <city>Garden City</city>
            <state>Harvest</state>
            <zip>54321</zip>
        </address>
    </company>
</data>"""


# Fixture to create a temporary XML file
@pytest.fixture()
def xml_file(tmp_path):
    file = tmp_path / "test.xml"
    with open(file, "w") as f:
        f.write(SAMPLE_XML)
    return file


def test_xml_reader_init():
    reader = XMLReader(tree_level_split=2)
    assert reader.tree_level_split == 2


def test_parse_xml_to_document():
    reader = XMLReader(1)
    root = ET.fromstring(SAMPLE_XML)
    documents = reader._parse_xmlelt_to_document(root)
    assert "Fresh Produce Ltd." in documents[-1].text
    assert "fruit" in documents[0].text


def test_load_data_xml(xml_file):
    reader = XMLReader()

    documents = reader.load_data(xml_file)
    assert len(documents) == 1
    assert "Apple" in documents[0].text
    assert "Garden City" in documents[0].text
