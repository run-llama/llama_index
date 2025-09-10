import os
import pytest
import tempfile

from llama_index.core.readers.base import BaseReader
from llama_index.readers.microsoft_sharepoint import SharePointReader
from llama_index.readers.microsoft_sharepoint.base import SharePointType
from llama_index.readers.microsoft_sharepoint.event import (
    FileType,
    PageDataFetchStartedEvent,
    PageDataFetchCompletedEvent,
    PageSkippedEvent,
    PageFailedEvent,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.schema import Document

from unittest.mock import patch, MagicMock
from pathlib import Path


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

    # Test basic attributes instead of schema (due to callable fields)
    assert reader.client_id == test_client_id
    assert reader.client_secret == test_client_secret
    assert reader.tenant_id == test_tenant_id
    
    # Test that the reader can be created with basic serialization
    json_data = reader.model_dump_json(exclude_unset=True, exclude={'process_document_callback', 'process_attachment_callback'})
    assert json_data is not None
    
    # Test that a new reader can be created with the same basic attributes
    new_reader = SharePointReader(
        client_id=reader.client_id,
        client_secret=reader.client_secret, 
        tenant_id=reader.tenant_id
    )
    assert new_reader.client_id == reader.client_id
    assert new_reader.client_secret == reader.client_secret
    assert new_reader.tenant_id == reader.tenant_id


@pytest.fixture()
def sharepoint_reader():
    sharepoint_reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        drive_name="dummy_drive_name",
    )

    sharepoint_reader._drive_id_endpoint = (
        "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives"
    )
    sharepoint_reader._authorization_headers = {"Authorization": "Bearer dummy_token"}

    return sharepoint_reader


def mock_send_get_with_retry(url):
    mock_response = MagicMock()
    mock_response.status_code = 200

    if url == "https://graph.microsoft.com/v1.0/sites":
        # Mock response for site information endpoint
        mock_response.json.return_value = {
            "value": [{"id": "dummy_site_id", "name": "dummy_site_name"}]
        }
    elif url == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives":
        # Mock response for drive information endpoint
        mock_response.json.return_value = {
            "value": [{"id": "dummy_drive_id", "name": "dummy_drive_name"}]
        }
    elif (
        url
        == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id/root:/dummy_folder_path"
    ):
        # Mock response for folder information endpoint
        mock_response.json.return_value = {"id": "dummy_folder_id"}
    elif (
        url
        == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id/items/dummy_folder_id/children"
    ):
        # Mock response for listing folder contents
        mock_response.json.return_value = {
            "value": [
                {"id": "file1_id", "name": "file1.txt", "file": {}},
                {"id": "file2_id", "name": "file2.txt", "file": {}},
            ]
        }
    elif (
        url
        == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id/items/file1_id/permissions"
    ):
        # Mock response for file1 permissions
        mock_response.json.return_value = {
            "value": [
                {"grantedToV2": {"user": {"id": "user1", "displayName": "User One"}}}
            ]
        }
    elif (
        url
        == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id/items/file2_id/permissions"
    ):
        # Mock response for file2 permissions
        mock_response.json.return_value = {
            "value": [
                {"grantedToV2": {"user": {"id": "user2", "displayName": "User Two"}}}
            ]
        }
    elif (
        url
        == "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id/items"
    ):
        # Mock response for getting item details by path
        if "file1.txt" in url:
            mock_response.json.return_value = {
                "id": "file1_id",
                "name": "file1.txt",
                "@microsoft.graph.downloadUrl": "http://dummyurl/file1.txt",
            }
        elif "file2.txt" in url:
            mock_response.json.return_value = {
                "id": "file2_id",
                "name": "file2.txt",
                "@microsoft.graph.downloadUrl": "http://dummyurl/file2.txt",
            }
    else:
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"message": "Not Found"}}

    return mock_response


@pytest.fixture(autouse=True)
def mock_sharepoint_api_calls():
    with (
        patch.object(SharePointReader, "_get_access_token", return_value="dummy_token"),
        patch.object(
            SharePointReader,
            "_get_site_id_with_host_name",
            return_value="dummy_site_id",
        ),
        patch.object(
            SharePointReader,
            "_get_sharepoint_folder_id",
            return_value="dummy_folder_id",
        ),
        patch.object(SharePointReader, "_get_drive_id", return_value="dummy_drive_id"),
        patch.object(
            SharePointReader,
            "_send_get_with_retry",
            side_effect=mock_send_get_with_retry,
        ),
    ):
        yield


def test_list_resources(sharepoint_reader):
    # Setting the _drive_id_endpoint manually to avoid the AttributeError
    file_paths = sharepoint_reader.list_resources(
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        recursive=False,
    )
    assert len(file_paths) == 2
    assert file_paths[0] == Path("dummy_site_name/dummy_folder_path/file1.txt")
    assert file_paths[1] == Path("dummy_site_name/dummy_folder_path/file2.txt")


def test_load_documents_with_metadata(sharepoint_reader):
    # Setting the _drive_id_endpoint manually to avoid the AttributeError
    sharepoint_reader._drive_id_endpoint = (
        "https://graph.microsoft.com/v1.0/sites/dummy_site_id/drives/dummy_drive_id"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create mock files in the temporary directory
        file1_path = os.path.join(tmpdirname, "file1.txt")
        file2_path = os.path.join(tmpdirname, "file2.txt")
        with open(file1_path, "w") as f:
            f.write("File 1 content")
        with open(file2_path, "w") as f:
            f.write("File 2 content")

        # Prepare metadata for the mock files
        files_metadata = {
            file1_path: {
                "file_id": "file1_id",
                "file_name": "file1.txt",
                "url": "http://dummyurl/file1.txt",
                "file_path": file1_path,
            },
            file2_path: {
                "file_id": "file2_id",
                "file_name": "file2.txt",
                "url": "http://dummyurl/file2.txt",
                "file_path": file2_path,
            },
        }

        documents = sharepoint_reader._load_documents_with_metadata(
            files_metadata, tmpdirname, recursive=False
        )

        assert documents is not None
        assert len(documents) == 2
        assert documents[0].metadata["file_name"] == "file1.txt"
        assert documents[1].metadata["file_name"] == "file2.txt"
        assert documents[0].text == "File 1 content"
        assert documents[1].text == "File 2 content"


def test_required_exts():
    sharepoint_reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        drive_name="dummy_drive_name",
        required_exts=[".md"],
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        readme_file_path = os.path.join(tmpdirname, "readme.md")
        audio_file_path = os.path.join(tmpdirname, "audio.aac")
        with open(readme_file_path, "w") as f:
            f.write("Readme content")
        with open(audio_file_path, "wb") as f:
            f.write(bytearray([0xFF, 0xF1, 0x50, 0x80, 0x00, 0x7F, 0xFC, 0x00]))

        file_metadata = {
            readme_file_path: {
                "file_id": "readme_file_id",
                "file_name": "readme.md",
                "url": "http://dummyurl/readme.md",
                "file_path": readme_file_path,
            },
            audio_file_path: {
                "file_id": "audio_file_id",
                "file_name": "audio.aac",
                "url": "http://dummyurl/audio.aac",
                "file_path": audio_file_path,
            },
        }

        documents = sharepoint_reader._load_documents_with_metadata(
            file_metadata, tmpdirname, recursive=False
        )

        assert documents is not None
        assert len(documents) == 1
        assert documents[0].metadata["file_name"] == "readme.md"
        assert documents[0].text == "Readme content"


def test_custom_parsers_and_custom_folder(tmp_path):
    """Test that custom_parsers and custom_folder work together."""
    mock_parser = MagicMock()
    custom_parsers = {FileType.PDF: mock_parser}

    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        custom_parsers=custom_parsers,
        custom_folder=str(tmp_path),
    )

    assert reader.custom_parsers == custom_parsers
    assert reader.custom_folder == str(tmp_path)
    assert reader.custom_parser_manager is not None


def test_custom_parser_usage(tmp_path):
    """Test that custom parser is used for supported file types."""
    mock_parser = MagicMock()
    mock_parser.load_data.return_value = [Document(text="custom content")]

    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        custom_parsers={FileType.PDF: mock_parser},
        custom_folder=str(tmp_path),
    )

    # Simulate a PDF file in metadata
    file_path = tmp_path / "file.pdf"
    file_path.write_bytes(b"dummy")
    files_metadata = {str(file_path): {"file_name": "file.pdf", "file_path": str(file_path)}}

    docs = reader._load_documents_with_metadata(files_metadata, str(tmp_path), recursive=False)
    assert docs[0].text == "custom content"


def test_document_callback_functionality():
    """Test that document callback is properly stored and functional."""
    excluded_files = ["file1", "file2"]

    def document_filter(file_id: str) -> bool:
        return file_id not in excluded_files

    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        process_document_callback=document_filter,
    )

    assert reader.process_document_callback == document_filter
    assert document_filter("normal_file") is True
    assert document_filter("file1") is False
    assert document_filter("file2") is False


def test_event_system_page_events():
    """Test event system with page events."""
    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
    )

    page_events = []

    class PageEventHandler(BaseEventHandler):
        def handle(self, event):
            if isinstance(
                event,
                (
                    PageDataFetchStartedEvent,
                    PageDataFetchCompletedEvent,
                    PageSkippedEvent,
                ),
            ):
                page_events.append(event)

    dispatcher = get_dispatcher("llama_index.readers.microsoft_sharepoint.base")
    page_handler = PageEventHandler()
    dispatcher.add_event_handler(page_handler)

    # Simulate event flow
    dispatcher.event(PageDataFetchStartedEvent(page_id="page1"))
    dispatcher.event(
        PageDataFetchCompletedEvent(
            page_id="page1", document=Document(text="content1", id_="page1")
        )
    )
    dispatcher.event(PageSkippedEvent(page_id="page2"))

    assert len(page_events) == 3
    event_types = [type(event).__name__ for event in page_events]
    assert "PageDataFetchStartedEvent" in event_types
    assert "PageDataFetchCompletedEvent" in event_types
    assert "PageSkippedEvent" in event_types

    # Clean up
    if page_handler in dispatcher.event_handlers:
        dispatcher.event_handlers.remove(page_handler)


def test_event_system_page_failed_event():
    """Test event system with page failed event."""
    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
    )

    error_events = []

    class ErrorEventHandler(BaseEventHandler):
        def handle(self, event):
            if isinstance(event, PageFailedEvent):
                error_events.append(event)

    dispatcher = get_dispatcher("llama_index.readers.microsoft_sharepoint.base")
    error_handler = ErrorEventHandler()
    dispatcher.add_event_handler(error_handler)

    dispatcher.event(PageFailedEvent(page_id="page3", error="Network timeout"))

    assert len(error_events) == 1
    assert error_events[0].page_id == "page3"
    assert error_events[0].error == "Network timeout"

    # Clean up
    if error_handler in dispatcher.event_handlers:
        dispatcher.event_handlers.remove(error_handler)


def test_fail_on_error_default_true():
    """Test that fail_on_error defaults to True."""
    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
    )
    assert reader.fail_on_error is True


def test_fail_on_error_explicit_false():
    """Test that fail_on_error can be set to False."""
    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        fail_on_error=False,
    )
    assert reader.fail_on_error is False


def test_fail_on_error_explicit_true():
    """Test that fail_on_error can be explicitly set to True."""
    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_folder_path="dummy_folder_path",
        fail_on_error=True,
    )
    assert reader.fail_on_error is True


# (Optional) If you support page reading, add a test for it:
def test_page_reading(monkeypatch, tmp_path):
    """Test page reading support if sharepoint_type='page'."""
    # Setup
    called = {}

    def document_filter(page_name: str) -> bool:
        called[page_name] = True
        return page_name != "skip_page"

    # For page reading, we'll manually set custom_folder after creation to avoid validation
    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret", 
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_type=SharePointType.PAGE,  # Use enum instead of string
        process_document_callback=document_filter,
    )
    
    # Manually set custom_folder after creation
    reader.custom_folder = str(tmp_path)

    # Mock the authentication and API methods
    def mock_get_access_token(self):
        return "dummy_token"

    def mock_get_site_id_with_host_name(self, access_token, sharepoint_site_name):
        return "dummy_site_id"

    def mock_list_pages(self, site_id, token):
        return [
            {"id": "1", "name": "normal_page"},
            {"id": "2", "name": "skip_page"},
        ]

    def mock_get_site_pages_list_id(self, site_id, token=None):
        return "list_id"

    def mock_get_page_text(self, site_id, list_id, page_id, token):
        return {
            "id": f"{list_id}_{page_id}",
            "name": "normal_page" if page_id == "1" else "skip_page",
            "lastModifiedDateTime": "2024-01-01T00:00:00Z",
            "textContent": "content",
            "rawHtml": "<p>content</p>",
        }

    # Monkeypatch methods on the class
    monkeypatch.setattr(SharePointReader, "_get_access_token", mock_get_access_token)
    monkeypatch.setattr(SharePointReader, "_get_site_id_with_host_name", mock_get_site_id_with_host_name)
    monkeypatch.setattr(SharePointReader, "list_pages", mock_list_pages)
    monkeypatch.setattr(SharePointReader, "get_site_pages_list_id", mock_get_site_pages_list_id)
    monkeypatch.setattr(SharePointReader, "get_page_text", mock_get_page_text)

    # Call load_data without download_dir - should use custom_folder via PAGE logic
    docs = reader.load_data()
    assert len(docs) == 1
    assert docs[0].metadata["page_name"] == "normal_page" 
    assert "normal_page" in called
    assert "skip_page" in called
