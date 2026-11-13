from types import SimpleNamespace

from llama_index.core.readers.base import BaseReader
from llama_index.readers.azstorage_blob import AzStorageBlobReader

test_container_name = "test_container"


class FakeBlobClient:
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def download_blob(self):
        return SimpleNamespace(readinto=lambda file: file.write(self.content))

    def get_blob_properties(self):
        return {"name": self.name}


class FakeContainerClient:
    def __init__(self, blobs):
        self.blobs = blobs

    def list_blobs(self, *args, **kwargs):
        return [SimpleNamespace(name=name) for name in self.blobs]

    def get_blob_client(self, blob):
        return FakeBlobClient(blob.name, self.blobs[blob.name])


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

    json = reader.json(exclude_unset=True)

    new_reader = AzStorageBlobReader.parse_raw(json)
    assert new_reader.container_name == reader.container_name


def test_load_data_keeps_blobs_whose_names_flatten_to_the_same_file_name(monkeypatch):
    # Listed in the lexicographic order Azure returns. All three flatten to
    # contracts-2025-report*.txt, and the first is already named like a renamed copy.
    blobs = {
        "contracts-2025-report_1.txt": b"existing copy",
        "contracts-2025/report.txt": b"second report",
        "contracts/2025-report.txt": b"first report",
    }
    monkeypatch.setattr(
        AzStorageBlobReader,
        "_get_container_client",
        lambda self: FakeContainerClient(blobs),
    )

    documents = AzStorageBlobReader(container_name=test_container_name).load_data()

    assert sorted((doc.metadata["name"], doc.text) for doc in documents) == [
        ("contracts-2025-report_1.txt", "existing copy"),
        ("contracts-2025/report.txt", "second report"),
        ("contracts/2025-report.txt", "first report"),
    ]
