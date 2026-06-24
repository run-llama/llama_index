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


def test_decompress_section_allows_data_within_limit():
    reader = HWPReader(max_decompressed_section_size=32)
    data = b"hwp section text"

    assert reader._decompress_section(_raw_deflate(data)) == data


def test_decompress_section_rejects_data_over_limit():
    reader = HWPReader(max_decompressed_section_size=8)

    with pytest.raises(ValueError, match="exceeds 8 bytes"):
        reader._decompress_section(_raw_deflate(b"a" * 64))
