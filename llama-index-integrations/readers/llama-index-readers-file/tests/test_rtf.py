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


def test_rtf_reader_honours_explicit_encoding(tmp_path):
    """
    RTFReader must accept an encoding and use it, as PagedCSVReader does.

    Without one, open() falls back to the platform default, which is cp1252 on
    Windows and utf-8 on Linux, so the same file reads differently per platform.
    """
    file = tmp_path / "test_cp1252.rtf"
    file.write_bytes(r"{\rtf Café naïve\par}".encode("cp1252"))

    documents = RTFReader(encoding="cp1252").load_data(file)

    assert len(documents) == 1
    assert "Café" in documents[0].text


def test_rtf_reader_defaults_to_utf8(tmp_path):
    """
    Parity guard: the default stays utf-8 rather than the platform default.
    """
    file = tmp_path / "test_utf8.rtf"
    file.write_text(r"{\rtf Café naïve\par}", encoding="utf-8")

    documents = RTFReader().load_data(file)

    assert len(documents) == 1
    assert "Café" in documents[0].text
