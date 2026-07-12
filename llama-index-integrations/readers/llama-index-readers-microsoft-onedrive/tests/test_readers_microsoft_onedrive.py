from pathlib import Path
from unittest import mock

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.microsoft_onedrive import OneDriveReader
from llama_index.readers.microsoft_onedrive.base import _OneDriveResourcePayload

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


def _drive_item(item_id: str, name: str, url: str) -> dict:
    return {"id": item_id, "name": name, "@microsoft.graph.downloadUrl": url}


def _mock_requests_get(url):
    return mock.Mock(content=f"content of {url}".encode())


def test_download_same_named_files_do_not_collide(tmp_path):
    reader = OneDriveReader(client_id=test_client_id, tenant_id=test_tenant_id)
    item_a = _drive_item("A1", "report.txt", "url_a")
    item_b = _drive_item("B2", "report.txt", "url_b")

    with mock.patch(
        "llama_index.readers.microsoft_onedrive.base.requests.get",
        side_effect=_mock_requests_get,
    ):
        path_a = reader._download_file_by_url(item_a, str(tmp_path))
        path_b = reader._download_file_by_url(item_b, str(tmp_path))

    assert path_a == str(tmp_path / "A1" / "report.txt")
    assert path_b == str(tmp_path / "B2" / "report.txt")
    assert Path(path_a).read_text() == "content of url_a"
    assert Path(path_b).read_text() == "content of url_b"


@pytest.mark.parametrize(
    "item_id",
    ["..", ".", "", "x/../A1", "x\\..\\A1", "/A1", "C:A1", "A1.", "A1 "],
)
def test_unsafe_drive_item_id_raises(tmp_path, item_id):
    reader = OneDriveReader(client_id=test_client_id, tenant_id=test_tenant_id)
    item = _drive_item(item_id, "report.txt", "url_a")

    with mock.patch(
        "llama_index.readers.microsoft_onedrive.base.requests.get",
        side_effect=_mock_requests_get,
    ):
        with pytest.raises(ValueError, match="Unsafe drive item id"):
            reader._download_file_by_url(item, str(tmp_path))


def test_load_data_recursive_false_still_reads_downloaded_files():
    reader = OneDriveReader(client_id=test_client_id, tenant_id=test_tenant_id)
    item = _drive_item("A1", "report.txt", "url_a")

    def fake_download(**kwargs):
        with mock.patch(
            "llama_index.readers.microsoft_onedrive.base.requests.get",
            side_effect=_mock_requests_get,
        ):
            path = reader._download_file_by_url(item, kwargs["temp_dir"])
        return [
            _OneDriveResourcePayload(
                resource_info={"item_id": "A1"}, downloaded_file_path=path
            )
        ]

    with mock.patch.object(
        reader, "_get_downloaded_files_metadata", side_effect=fake_download
    ):
        documents = reader.load_data(recursive=False)

    assert len(documents) == 1
    assert documents[0].text == "content of url_a"


def test_load_documents_with_metadata_same_named_files(tmp_path):
    reader = OneDriveReader(client_id=test_client_id, tenant_id=test_tenant_id)
    item_a = _drive_item("A1", "report.txt", "url_a")
    item_b = _drive_item("B2", "report.txt", "url_b")

    with mock.patch(
        "llama_index.readers.microsoft_onedrive.base.requests.get",
        side_effect=_mock_requests_get,
    ):
        payloads = [
            _OneDriveResourcePayload(
                resource_info={"item_id": item["id"]},
                downloaded_file_path=reader._download_file_by_url(item, str(tmp_path)),
            )
            for item in (item_a, item_b)
        ]

    documents = reader._load_documents_with_metadata(payloads, str(tmp_path))

    assert len(documents) == 2
    assert {doc.text for doc in documents} == {"content of url_a", "content of url_b"}
    assert {doc.metadata["item_id"] for doc in documents} == {"A1", "B2"}


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
