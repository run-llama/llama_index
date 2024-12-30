from llama_index.core.readers.base import BasePydanticReader
from llama_index.readers.gcs import GCSReader

test_bucket = "test"
test_service_account_key = {"test-key": "test-value"}


def test_class():
    names_of_base_classes = [b.__name__ for b in GCSReader.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes


def test_serialize():
    reader = GCSReader(
        bucket=test_bucket,
        service_account_key=test_service_account_key,
    )

    schema = reader.schema()
    assert schema is not None
    assert len(schema) > 0
    assert "bucket" in schema["properties"]

    json = reader.json(exclude_unset=True)

    new_reader = GCSReader.parse_raw(json)
    assert new_reader.bucket == reader.bucket
    assert new_reader.service_account_key == reader.service_account_key
