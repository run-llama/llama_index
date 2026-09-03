import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.microsoft_onedrive import OneDriveReader
from llama_index.readers.microsoft_onedrive.base import DEFAULT_REQUEST_TIMEOUT

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


def test_onedrive_requests_use_default_timeout(monkeypatch, tmp_path):
    calls = []

    class MockResponse:
        status_code = 200
        content = b"file contents"

        def json(self):
            return {"value": []}

    def mock_get(*args, **kwargs):
        calls.append((args, kwargs))
        return MockResponse()

    monkeypatch.setattr(
        "llama_index.readers.microsoft_onedrive.base.requests.get", mock_get
    )

    reader = OneDriveReader(client_id=test_client_id, tenant_id=test_tenant_id)

    reader._get_items_in_drive_with_maxretries("access-token")
    reader._download_file_by_url(
        {
            "@microsoft.graph.downloadUrl": "https://download.example/file",
            "name": "a.txt",
        },
        str(tmp_path),
    )
    reader._get_permissions_info({"id": "file-id"}, "user@example.com", "access-token")

    assert len(calls) == 3
    assert all(
        call_kwargs["timeout"] == DEFAULT_REQUEST_TIMEOUT for _, call_kwargs in calls
    )


def test_onedrive_requests_use_custom_timeout(monkeypatch, tmp_path):
    calls = []

    class MockResponse:
        status_code = 200
        content = b"file contents"

        def json(self):
            return {"value": []}

    def mock_get(*args, **kwargs):
        calls.append((args, kwargs))
        return MockResponse()

    monkeypatch.setattr(
        "llama_index.readers.microsoft_onedrive.base.requests.get", mock_get
    )

    request_timeout = (1.0, 5.0)
    reader = OneDriveReader(
        client_id=test_client_id,
        tenant_id=test_tenant_id,
        request_timeout=request_timeout,
    )

    reader._get_items_in_drive_with_maxretries("access-token")
    reader._download_file_by_url(
        {
            "@microsoft.graph.downloadUrl": "https://download.example/file",
            "name": "a.txt",
        },
        str(tmp_path),
    )
    reader._get_permissions_info({"id": "file-id"}, "user@example.com", "access-token")

    assert len(calls) == 3
    assert all(call_kwargs["timeout"] == request_timeout for _, call_kwargs in calls)


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
