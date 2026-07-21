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

    json = reader.json(exclude_unset=True)

    new_reader = AzStorageBlobReader.parse_raw(json)
    assert new_reader.container_name == reader.container_name


def test_load_data_preserves_blobs_with_colliding_sanitized_names(mocker):
    blob_contents = {
        "contracts/2025-report.txt": b"first report",
        "contracts-2025/report.txt": b"second report",
    }

    class BlobStream:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def readinto(self, file) -> None:
            file.write(self.content)

    class BlobClient:
        def __init__(self, name: str) -> None:
            self.name = name

        def download_blob(self) -> BlobStream:
            return BlobStream(blob_contents[self.name])

        def get_blob_properties(self):
            return {"name": self.name, "metadata": {"source": self.name}}

    container_client = mocker.Mock()
    container_client.list_blobs.return_value = [
        SimpleNamespace(name=name) for name in blob_contents
    ]
    container_client.get_blob_client.side_effect = lambda obj: BlobClient(obj.name)

    reader = AzStorageBlobReader(container_name=test_container_name)
    mocker.patch.object(reader, "_get_container_client", return_value=container_client)

    documents = reader.load_data()

    assert len(documents) == 2
    documents_by_name = {document.metadata["name"]: document for document in documents}
    assert set(documents_by_name) == set(blob_contents)
    for name, content in blob_contents.items():
        document = documents_by_name[name]
        assert document.get_content() == content.decode()
        assert document.metadata["metadata"] == {"source": name}
