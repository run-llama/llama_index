import os
import tempfile
from types import SimpleNamespace

from llama_index.core.readers.base import BaseReader
from llama_index.readers.azstorage_blob import AzStorageBlobReader

test_container_name = "test_container"


def test_class():
    names_of_base_classes = [b.__name__ for b in AzStorageBlobReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_serialize():
    reader = AzStorageBlobReader(
        container_name=test_container_name,
    )

    schema = reader.schema()
    assert schema is not None
    assert len(schema) > 0
    assert "container_name" in schema["properties"]

    json_data = reader.json(exclude_unset=True)

    new_reader = AzStorageBlobReader.parse_raw(json_data)
    assert new_reader.container_name == reader.container_name


def test_download_files_preserves_distinct_blob_names(monkeypatch):
    class MockDownloadStream:
        def __init__(self, text):
            self.text = text

        def readinto(self, fp):
            fp.write(self.text.encode("utf-8"))

    class MockBlobClient:
        def __init__(self, blob):
            self.blob = blob

        def download_blob(self):
            if self.blob.name == "contracts/2025-report.txt":
                return MockDownloadStream("Quarterly report")
            return MockDownloadStream("Annual report")

        def get_blob_properties(self):
            return {
                "creation_time": None,
                "last_modified": None,
                "last_accessed_on": None,
            }

    class MockContainerClient:
        def list_blobs(self, name_starts_with=None, include=None):
            return [
                SimpleNamespace(name="contracts/2025-report.txt"),
                SimpleNamespace(name="contracts-2025/report.txt"),
            ]

        def get_blob_client(self, blob):
            return MockBlobClient(blob)

    monkeypatch.setattr(
        AzStorageBlobReader,
        "_get_container_client",
        lambda self: MockContainerClient(),
    )

    reader = AzStorageBlobReader(container_name="test")

    with tempfile.TemporaryDirectory() as temp_dir:
        metadata = reader._download_files_and_extract_metadata(temp_dir)

        assert os.path.exists(os.path.join(temp_dir, "contracts", "2025-report.txt"))

        assert os.path.exists(os.path.join(temp_dir, "contracts-2025", "report.txt"))

        assert len(metadata) == 2
