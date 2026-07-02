import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
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


def _reader() -> OneDriveReader:
    return OneDriveReader(client_id=test_client_id, tenant_id=test_tenant_id)


def test_alist_resources_offloads_to_thread():
    reader = _reader()
    with (
        patch.object(
            OneDriveReader, "list_resources", return_value=["a.txt"]
        ) as mock_sync,
        patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy,
    ):
        result = asyncio.run(reader.alist_resources())

    # The async resource API must run the blocking sync call off the event loop.
    spy.assert_called_once()
    mock_sync.assert_called_once()
    assert result == ["a.txt"]


def test_aget_resource_info_offloads_to_thread():
    reader = _reader()
    info = {"file_path": "a.txt"}
    with (
        patch.object(
            OneDriveReader, "get_resource_info", return_value=info
        ) as mock_sync,
        patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy,
    ):
        result = asyncio.run(reader.aget_resource_info("a.txt"))

    spy.assert_called_once()
    mock_sync.assert_called_once()
    assert result == info


def test_aload_resource_offloads_to_thread():
    reader = _reader()
    docs = [Document(text="hi")]
    with (
        patch.object(OneDriveReader, "load_resource", return_value=docs) as mock_sync,
        patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy,
    ):
        result = asyncio.run(reader.aload_resource("a.txt"))

    spy.assert_called_once()
    mock_sync.assert_called_once()
    assert result == docs


def test_aread_file_content_offloads_to_thread():
    reader = _reader()
    with (
        patch.object(
            OneDriveReader, "read_file_content", return_value=b"data"
        ) as mock_sync,
        patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy,
    ):
        result = asyncio.run(reader.aread_file_content(Path("a.txt")))

    spy.assert_called_once()
    mock_sync.assert_called_once()
    assert result == b"data"


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
