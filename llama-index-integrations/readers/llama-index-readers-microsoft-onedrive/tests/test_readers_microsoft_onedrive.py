from types import SimpleNamespace

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.microsoft_onedrive import OneDriveReader
from llama_index.readers.microsoft_onedrive.base import _OneDriveResourcePayload

test_client_id = "test_client_id"
test_tenant_id = "test_tenant_id"


def _report_item(year):
    return {
        "id": f"item-{year}",
        "name": "report.txt",
        "parentReference": {"path": f"/drive/root:/Contracts/{year}"},
        "@microsoft.graph.downloadUrl": f"https://example.com/{year}",
    }


@pytest.fixture()
def fake_downloads(monkeypatch):
    monkeypatch.setattr(
        "requests.get",
        lambda url: SimpleNamespace(content=f"report from {url}".encode()),
    )


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


def test_same_named_files_from_different_folders_are_all_loaded(
    tmp_path, fake_downloads
):
    reader = OneDriveReader(client_id=test_client_id, tenant_id=test_tenant_id)
    items = [_report_item(2025), _report_item(2026)]
    payloads = [
        _OneDriveResourcePayload(
            resource_info=reader._extract_metadata_for_file(item),
            downloaded_file_path=reader._download_file_by_url(item, str(tmp_path)),
        )
        for item in items
    ]

    documents = reader._load_documents_with_metadata(payloads, str(tmp_path))

    assert sorted((doc.metadata["file_path"], doc.text) for doc in documents) == [
        ("/Contracts/2025/report.txt", "report from https://example.com/2025"),
        ("/Contracts/2026/report.txt", "report from https://example.com/2026"),
    ]


def test_an_item_downloaded_twice_is_saved_to_the_same_file(tmp_path, fake_downloads):
    reader = OneDriveReader(client_id=test_client_id, tenant_id=test_tenant_id)
    item = _report_item(2025)

    first = reader._download_file_by_url(item, str(tmp_path))
    second = reader._download_file_by_url(item, str(tmp_path))

    assert first == second
    assert first.endswith(".txt")
    assert len(list(tmp_path.iterdir())) == 1
