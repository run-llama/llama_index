import io
import struct
import zlib

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import (
    DocxReader,
    EpubReader,
    FlatReader,
    HWPReader,
    ImageCaptionReader,
    ImageReader,
    ImageVisionLLMReader,
    IPYNBReader,
    MarkdownReader,
    MboxReader,
    PandasCSVReader,
    PDFReader,
    PptxReader,
    VideoAudioReader,
    XMLReader,
    ImageTabularChartReader,
)


class FakeHWPFile:
    def __init__(self, section_data: bytes, compressed: bool = True) -> None:
        self.section_data = section_data
        self.compressed = compressed

    def openstream(self, section: str) -> io.BytesIO:
        if section == "FileHeader":
            header = bytearray(37)
            header[36] = int(self.compressed)
            return io.BytesIO(header)
        return io.BytesIO(self.section_data)


def raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(data) + compressor.flush()


def hwp_record(record_type: int, payload: bytes) -> bytes:
    length = len(payload)
    header = record_type | (min(length, 0xFFF) << 20)
    record = struct.pack("<I", header)
    if length >= 0xFFF:
        record += struct.pack("<I", length)
    return record + payload


def hwp_text_record(text: str) -> bytes:
    return hwp_record(67, text.encode("utf-16"))


def test_classes():
    names_of_base_classes = [b.__name__ for b in DocxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in HWPReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PDFReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in EpubReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in FlatReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageCaptionReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageVisionLLMReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in IPYNBReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MarkdownReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MboxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PptxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PandasCSVReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in VideoAudioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in XMLReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageTabularChartReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_hwp_compressed_section_is_decompressed_with_limit():
    reader = HWPReader()
    load_file = FakeHWPFile(raw_deflate(hwp_text_record("hello")))

    assert reader.get_text_from_section(load_file, "BodyText/Section0") == "hello\n"


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("text", ["한" * 2046, "한" * 2047], ids=["inline", "extended"])
def test_hwp_section_reads_text_records(compressed: bool, text: str) -> None:
    reader = HWPReader()
    section = hwp_text_record(text) + hwp_text_record("following paragraph")
    load_file = FakeHWPFile(
        raw_deflate(section) if compressed else section, compressed=compressed
    )

    assert (
        reader.get_text_from_section(load_file, "BodyText/Section0")
        == f"{text}\nfollowing paragraph\n"
    )


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("payload_size", [0xFFE, 0xFFF, 0x1000])
def test_hwp_section_skips_nontext_record(compressed: bool, payload_size: int) -> None:
    reader = HWPReader()
    section = hwp_record(0x200, b"\x01" * payload_size) + hwp_text_record(
        "following paragraph"
    )
    load_file = FakeHWPFile(
        raw_deflate(section) if compressed else section, compressed=compressed
    )

    assert (
        reader.get_text_from_section(load_file, "BodyText/Section0")
        == "following paragraph\n"
    )


def test_hwp_compressed_section_rejects_decompression_bomb():
    reader = HWPReader()
    reader.MAX_DECOMPRESSED_SECTION_SIZE = 16
    load_file = FakeHWPFile(raw_deflate(hwp_text_record("too large")))

    with pytest.raises(ValueError, match="Decompressed HWP section exceeds"):
        reader.get_text_from_section(load_file, "BodyText/Section0")


def test_hwp_compressed_section_rejects_truncated_stream():
    reader = HWPReader()
    load_file = FakeHWPFile(raw_deflate(hwp_text_record("hello"))[:-1])

    with pytest.raises(zlib.error, match="ended before the stream was complete"):
        reader.get_text_from_section(load_file, "BodyText/Section0")
