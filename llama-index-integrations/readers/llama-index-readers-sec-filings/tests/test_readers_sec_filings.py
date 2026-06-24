from io import BytesIO

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.sec_filings import SECFilingsLoader


def test_class():
    names_of_base_classes = [b.__name__ for b in SECFilingsLoader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_ungz_file_closes_gzip_reader(monkeypatch):
    pytest.importorskip("fastapi")
    section = pytest.importorskip(
        "llama_index.readers.sec_filings.prepline_sec_filings.api.section"
    )

    class ClosingBytesIO(BytesIO):
        was_closed = False

        def close(self):
            self.was_closed = True
            super().close()

    source = ClosingBytesIO(b"plain text")
    monkey_gzip = ClosingBytesIO(b"plain text")

    class Upload:
        filename = "filing.txt.gz"
        content_type = "application/gzip"
        file = source

    class GzipOpen:
        def __call__(self, file):
            assert file is source
            return monkey_gzip

    monkeypatch.setattr(section.gzip, "open", GzipOpen())
    result = section.ungz_file(Upload())

    assert result.file.read() == b"plain text"
    assert monkey_gzip.was_closed
