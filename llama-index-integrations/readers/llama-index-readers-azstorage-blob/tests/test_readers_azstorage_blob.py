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
