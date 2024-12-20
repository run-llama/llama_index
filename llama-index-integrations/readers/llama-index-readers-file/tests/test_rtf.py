import pytest
from striprtf.striprtf import rtf_to_text

from llama_index.readers.file.rtf import RTFReader

# Sample XML data for testing
SAMPLE_RTF = """{\\rtf
    Hello!\\par
    This is a rtf file {\\b bolded}.\\par
}"""


# Fixture to create a temporary XML file
@pytest.fixture()
def rtf_file(tmp_path):
    file = tmp_path / "test.rtf"
    with open(file, "w") as f:
        f.write(SAMPLE_RTF)
    return file


def test_load_data_rtf(rtf_file):
    reader = RTFReader()
    text = rtf_to_text(SAMPLE_RTF).strip()
    documents = reader.load_data(rtf_file)
    assert len(documents) == 1
    assert text == documents[0].text
