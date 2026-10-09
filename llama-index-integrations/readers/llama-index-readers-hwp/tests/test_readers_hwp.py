import io
import struct
import zlib

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.readers.hwp import HWPReader


def test_class():
    names_of_base_classes = [b.__name__ for b in HWPReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


class FakeHWPFile:
    def __init__(self, section_data: bytes) -> None:
        self.section_data = section_data

    def openstream(self, section: str) -> io.BytesIO:
        if section == "FileHeader":
            header = bytearray(37)
            header[36] = 1
            return io.BytesIO(header)
        return io.BytesIO(self.section_data)


def raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(data) + compressor.flush()


def hwp_text_record(text: str) -> bytes:
    encoded_text = text.encode("utf-16")
    header = 67 | (len(encoded_text) << 20)
    return struct.pack("<I", header) + encoded_text


def test_compressed_section_is_decompressed_with_limit():
    reader = HWPReader()
    load_file = FakeHWPFile(raw_deflate(hwp_text_record("hello")))

    assert reader.get_text_from_section(load_file, "BodyText/Section0") == "hello\n"


def test_compressed_section_rejects_decompression_bomb():
    reader = HWPReader()
    reader.MAX_DECOMPRESSED_SECTION_SIZE = 16
    load_file = FakeHWPFile(raw_deflate(hwp_text_record("too large")))

    with pytest.raises(ValueError, match="Decompressed HWP section exceeds"):
        reader.get_text_from_section(load_file, "BodyText/Section0")


def test_compressed_section_rejects_truncated_stream():
    reader = HWPReader()
    load_file = FakeHWPFile(raw_deflate(hwp_text_record("hello"))[:-1])

    with pytest.raises(zlib.error, match="ended before the stream was complete"):
        reader.get_text_from_section(load_file, "BodyText/Section0")
