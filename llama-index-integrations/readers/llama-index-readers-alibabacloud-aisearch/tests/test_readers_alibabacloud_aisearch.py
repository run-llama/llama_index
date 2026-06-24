from io import BytesIO

from llama_index.readers.alibabacloud_aisearch import (
    AlibabaCloudAISearchDocumentReader,
    AlibabaCloudAISearchImageReader,
)
from llama_index.core.readers.base import BasePydanticReader
from llama_index.readers.alibabacloud_aisearch.base import _read_file_as_base64


def test_class():
    names_of_base_classes = [
        b.__name__ for b in AlibabaCloudAISearchDocumentReader.__mro__
    ]
    assert BasePydanticReader.__name__ in names_of_base_classes
    names_of_base_classes = [
        b.__name__ for b in AlibabaCloudAISearchImageReader.__mro__
    ]
    assert BasePydanticReader.__name__ in names_of_base_classes


def test_read_file_as_base64_closes_file(monkeypatch):
    class ClosingBytesIO(BytesIO):
        was_closed = False

        def close(self):
            self.was_closed = True
            super().close()

    file_obj = ClosingBytesIO(b"hello")

    def fake_open(file_path, mode):
        assert file_path == "doc.txt"
        assert mode == "rb"
        return file_obj

    monkeypatch.setattr("builtins.open", fake_open)

    assert _read_file_as_base64("doc.txt") == "aGVsbG8="
    assert file_obj.was_closed
