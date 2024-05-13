from llama_index.core.readers.base import BaseReader
from llama_index.readers.microsoft_sharepoint import SharePointReader

test_client_id = "test_client_id"
test_client_secret = "test_client_secret"
test_tenant_id = "test_tenant_id"


def test_class():
    names_of_base_classes = [b.__name__ for b in SharePointReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_serialize():
    reader = SharePointReader(
        client_id=test_client_id,
        client_secret=test_client_secret,
        tenant_id=test_tenant_id,
    )

    schema = reader.schema()
    assert schema is not None
    assert len(schema) > 0
    assert "client_id" in schema["properties"]
    assert "client_secret" in schema["properties"]
    assert "tenant_id" in schema["properties"]

    json = reader.json(exclude_unset=True)

    new_reader = SharePointReader.parse_raw(json)
    assert new_reader.client_id == reader.client_id
    assert new_reader.client_secret == reader.client_secret
    assert new_reader.tenant_id == reader.tenant_id
