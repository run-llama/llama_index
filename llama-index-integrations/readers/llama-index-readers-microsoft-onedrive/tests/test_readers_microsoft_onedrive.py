import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.microsoft_onedrive import OneDriveReader

test_client_id = "test_client_id"
test_tenant_id = "test_tenant_id"


def test_class():
    names_of_base_classes = [b.__name__ for b in OneDriveReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_serialize():
    reader = OneDriveReader(
        client_id=test_client_id,
        tenant_id=test_tenant_id,
        required_exts=[".txt", ".csv"],
    )

    schema = reader.schema()
    assert schema is not None
    assert len(schema) > 0
    assert "client_id" in schema["properties"]
    assert "tenant_id" in schema["properties"]
    assert "required_exts" in schema["properties"]

    json = reader.json(exclude_unset=True)

    new_reader = OneDriveReader.parse_raw(json)
    assert new_reader.client_id == reader.client_id
    assert new_reader.tenant_id == reader.tenant_id
    assert new_reader.required_exts == reader.required_exts


@pytest.fixture()
def real_onedrive_reader():
    raise pytest.skip("Fill in redacted values to run this test")
    return OneDriveReader(
        userprincipalname="REDACTED",
        folder_path="REDACTED",
        client_id="REDACTED",
        client_secret="REDACTED",
        tenant_id="REDACTED",
    )


def test_mixins(real_onedrive_reader: OneDriveReader):
    docs = real_onedrive_reader.load_data()
    assert len(docs) > 0
    resources = real_onedrive_reader.list_resources()
    assert len(resources) == len(docs)
    resource = resources[0]
    resource_info = real_onedrive_reader.get_resource_info(resource)
    assert resource_info is not None
    assert resource_info["file_path"] == resource
    assert resource_info["file_name"] in resource
    assert resource_info["file_size"] > 0

    file_content = real_onedrive_reader.read_file_content(resource)
    assert file_content is not None
    assert len(file_content) == resource_info["file_size"]
