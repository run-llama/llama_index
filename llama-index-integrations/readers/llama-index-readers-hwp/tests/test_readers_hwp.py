import io
import zlib

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.readers.hwp import HWPReader


def test_class():
    names_of_base_classes = [b.__name__ for b in HWPReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def _raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(data) + compressor.flush()


class _FakeHWPFile:
    def __init__(self, section_data: bytes) -> None:
        header = bytearray(37)
        header[36] = 1
        self._streams = {
            "FileHeader": bytes(header),
            "BodyText/Section0": section_data,
        }

    def openstream(self, name: str) -> io.BytesIO:
        return io.BytesIO(self._streams[name])


def test_decompress_section_preserves_small_raw_deflate() -> None:
    reader = HWPReader()
    data = b"small section payload"

    assert reader._decompress_section(_raw_deflate(data)) == data


def test_get_text_from_section_rejects_oversized_decompressed_data() -> None:
    reader = HWPReader()
    reader.MAX_DECOMPRESSED_SECTION_BYTES = 8
    compressed_section = _raw_deflate(b"A" * 32)

    with pytest.raises(ValueError, match="Decompressed HWP section exceeds"):
        reader.get_text_from_section(
            _FakeHWPFile(compressed_section), "BodyText/Section0"
        )
