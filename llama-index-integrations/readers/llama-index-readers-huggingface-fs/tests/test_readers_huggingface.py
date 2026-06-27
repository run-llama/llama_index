from io import BytesIO
from llama_index.core.readers.base import BaseReader
from llama_index.readers.huggingface_fs import HuggingFaceFSReader


def test_class():
    names_of_base_classes = [b.__name__ for b in HuggingFaceFSReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_load_dicts_closes_gzip_file(monkeypatch):
    class FakeFileSystem:
        def read_bytes(self, path):
            return b"compressed"

    class ClosingBytesIO(BytesIO):
        was_closed = False

        def close(self):
            self.was_closed = True
            super().close()

    gzip_file = ClosingBytesIO(b'{"text": "hello"}\n')

    def fake_gzip_open(*args, **kwargs):
        return gzip_file

    monkeypatch.setattr("gzip.open", fake_gzip_open)

    reader = HuggingFaceFSReader.__new__(HuggingFaceFSReader)
    reader.fs = FakeFileSystem()

    assert reader.load_dicts("repo/data.jsonl.gz") == [{"text": "hello"}]
    assert gzip_file.was_closed
