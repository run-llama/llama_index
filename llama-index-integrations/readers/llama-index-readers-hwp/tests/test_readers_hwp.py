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


def test_decompress_section_round_trips_raw_deflate():
    payload = b"hello hwp" * 128
    reader = HWPReader(max_decompressed_size=len(payload))

    assert reader._decompress_section(_raw_deflate(payload)) == payload


def test_decompress_section_rejects_oversized_output():
    payload = b"x" * 4096
    reader = HWPReader(max_decompressed_size=len(payload) - 1)

    with pytest.raises(ValueError, match="exceeds configured maximum size"):
        reader._decompress_section(_raw_deflate(payload))


def test_max_decompressed_size_must_be_positive():
    with pytest.raises(ValueError, match="greater than 0"):
        HWPReader(max_decompressed_size=0)
