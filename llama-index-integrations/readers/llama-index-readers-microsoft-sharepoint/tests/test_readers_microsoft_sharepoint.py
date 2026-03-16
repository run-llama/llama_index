import os
import pytest
import tempfile

from llama_index.core.readers.base import BaseReader
from llama_index.readers.microsoft_sharepoint import SharePointReader
from llama_index.readers.microsoft_sharepoint.base import SharePointType

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

    schema = reader.model_json_schema()
    assert schema is not None
    assert len(schema) > 0
    assert "client_id" in schema["properties"]
    assert "client_secret" in schema["properties"]
    assert "tenant_id" in schema["properties"]

    json = reader.model_dump_json(exclude_unset=True)

    new_reader = SharePointReader.model_validate_json(json)
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


# Page-related tests
@pytest.fixture()
def sharepoint_page_reader():
    """Fixture for SharePoint reader configured for pages."""
    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        sharepoint_site_name="dummy_site_name",
        sharepoint_host_name="dummy.sharepoint.com",
        sharepoint_relative_url="/sites/DummySite",
        sharepoint_type=SharePointType.PAGE,
    )
    reader._authorization_headers = {"Authorization": "Bearer dummy_token"}
    return reader


def mock_page_send_get_with_retry(url):
    """Mock responses for page-related API calls."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    if "lists?$filter=displayName" in url:
        # Mock response for Site Pages list lookup
        mock_response.json.return_value = {
            "value": [{"id": "site_pages_list_id", "displayName": "Site Pages"}]
        }
    elif "lists/site_pages_list_id/items?" in url and "expand=fields" in url:
        # Mock response for listing pages (note the ? to distinguish from single item)
        mock_response.json.return_value = {
            "value": [
                {
                    "id": "page1",
                    "lastModifiedDateTime": "2024-01-01T00:00:00Z",
                    "fields": {
                        "FileLeafRef": "Home.aspx",
                        "CanvasContent1": "<p>Welcome to the home page</p>",
                    },
                },
                {
                    "id": "page2",
                    "lastModifiedDateTime": "2024-01-02T00:00:00Z",
                    "fields": {
                        "FileLeafRef": "About.aspx",
                        "CanvasContent1": "<p>About us content</p>",
                    },
                },
            ]
        }
    elif "/items/page1?" in url:
        # Mock response for specific page1
        mock_response.json.return_value = {
            "id": "page1",
            "lastModifiedDateTime": "2024-01-01T00:00:00Z",
            "fields": {
                "FileLeafRef": "Home.aspx",
                "CanvasContent1": "<p>Welcome to the home page</p>",
            },
        }
    elif "/items/page2?" in url:
        # Mock response for specific page2
        mock_response.json.return_value = {
            "id": "page2",
            "lastModifiedDateTime": "2024-01-02T00:00:00Z",
            "fields": {
                "FileLeafRef": "About.aspx",
                "CanvasContent1": "<p>About us content</p>",
            },
        }
    else:
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"message": "Not Found"}}

    return mock_response


def test_get_site_pages_list_id(sharepoint_page_reader):
    """Test getting the Site Pages list ID."""
    with patch.object(
        SharePointReader,
        "_send_get_with_retry",
        side_effect=mock_page_send_get_with_retry,
    ):
        list_id = sharepoint_page_reader.get_site_pages_list_id("dummy_site_id")
        assert list_id == "site_pages_list_id"


def test_get_site_pages_list_id_not_found(sharepoint_page_reader):
    """Test getting Site Pages list ID when not found."""

    def mock_empty_response(url):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"value": []}
        return mock_response

    with patch.object(
        SharePointReader,
        "_send_get_with_retry",
        side_effect=mock_empty_response,
    ):
        with pytest.raises(ValueError, match="Site Pages list not found"):
            sharepoint_page_reader.get_site_pages_list_id("dummy_site_id")


def test_list_pages(sharepoint_page_reader):
    """Test listing SharePoint pages."""
    with patch.object(
        SharePointReader,
        "_send_get_with_retry",
        side_effect=mock_page_send_get_with_retry,
    ):
        pages = sharepoint_page_reader.list_pages("dummy_site_id")
        assert len(pages) == 2
        assert pages[0]["id"] == "page1"
        assert pages[0]["name"] == "Home.aspx"
        assert pages[1]["id"] == "page2"
        assert pages[1]["name"] == "About.aspx"


def test_get_page_text(sharepoint_page_reader):
    """Test getting page text content."""
    with patch.object(
        SharePointReader,
        "_send_get_with_retry",
        side_effect=mock_page_send_get_with_retry,
    ):
        page_data = sharepoint_page_reader.get_page_text(
            site_id="dummy_site_id",
            list_id="site_pages_list_id",
            page_id="page1",
        )
        assert page_data["id"] == "site_pages_list_id:page1"
        assert page_data["name"] == "Home.aspx"
        assert "Welcome to the home page" in page_data["textContent"]
        assert page_data["rawHtml"] == "<p>Welcome to the home page</p>"


def test_get_page_text_with_combined_id(sharepoint_page_reader):
    """Test getting page text with combined listId:pageId format."""
    with patch.object(
        SharePointReader,
        "_send_get_with_retry",
        side_effect=mock_page_send_get_with_retry,
    ):
        page_data = sharepoint_page_reader.get_page_text(
            site_id="dummy_site_id",
            list_id="ignored_list_id",
            page_id="site_pages_list_id:page1",
        )
        # Should extract list_id from combined page_id
        assert page_data["id"] == "site_pages_list_id:page1"
        assert page_data["name"] == "Home.aspx"


def test_combined_id_uses_colon_separator(sharepoint_page_reader):
    """Test that combined IDs use colon separator to avoid conflicts with IDs containing underscores."""
    with patch.object(
        SharePointReader,
        "_send_get_with_retry",
        side_effect=mock_page_send_get_with_retry,
    ):
        page_data = sharepoint_page_reader.get_page_text(
            site_id="dummy_site_id",
            list_id="site_pages_list_id",
            page_id="page1",
        )
        # Verify colon is used as separator
        assert ":" in page_data["id"]
        assert "_" not in page_data["id"] or page_data["id"].count(":") == 1


def test_load_pages_data(sharepoint_page_reader):
    """Test loading all pages as documents."""
    with (
        patch.object(SharePointReader, "_get_access_token", return_value="dummy_token"),
        patch.object(
            SharePointReader,
            "_get_site_id_with_host_name",
            return_value="dummy_site_id",
        ),
        patch.object(
            SharePointReader,
            "_send_get_with_retry",
            side_effect=mock_page_send_get_with_retry,
        ),
    ):
        documents = sharepoint_page_reader.load_pages_data()
        assert len(documents) == 2
        assert documents[0].metadata["page_name"] == "Home.aspx"
        assert documents[1].metadata["page_name"] == "About.aspx"
        assert documents[0].metadata["sharepoint_type"] == "page"
        assert "Welcome to the home page" in documents[0].text


def test_load_data_with_page_type(sharepoint_page_reader):
    """Test that load_data routes to load_pages_data when sharepoint_type is PAGE."""
    with (
        patch.object(SharePointReader, "_get_access_token", return_value="dummy_token"),
        patch.object(
            SharePointReader,
            "_get_site_id_with_host_name",
            return_value="dummy_site_id",
        ),
        patch.object(
            SharePointReader,
            "_send_get_with_retry",
            side_effect=mock_page_send_get_with_retry,
        ),
    ):
        documents = sharepoint_page_reader.load_data()
        assert len(documents) == 2
        assert all(doc.metadata["sharepoint_type"] == "page" for doc in documents)


def test_sharepoint_type_enum():
    """Test SharePointType enum values."""
    assert SharePointType.DRIVE.value == "drive"
    assert SharePointType.PAGE.value == "page"


def test_get_all_items_with_pagination():
    """Test pagination helper retrieves all items across multiple pages."""
    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
    )
    reader._authorization_headers = {"Authorization": "Bearer dummy_token"}

    # Mock responses for multiple pages
    page1_response = MagicMock()
    page1_response.json.return_value = {
        "value": [{"id": "item1"}, {"id": "item2"}],
        "@odata.nextLink": "https://graph.microsoft.com/v1.0/endpoint?$skiptoken=page2",
    }

    page2_response = MagicMock()
    page2_response.json.return_value = {
        "value": [{"id": "item3"}, {"id": "item4"}],
        "@odata.nextLink": "https://graph.microsoft.com/v1.0/endpoint?$skiptoken=page3",
    }

    page3_response = MagicMock()
    page3_response.json.return_value = {
        "value": [{"id": "item5"}],
        # No nextLink - last page
    }

    responses = [page1_response, page2_response, page3_response]
    call_count = 0

    def mock_get(url):
        nonlocal call_count
        response = responses[call_count]
        call_count += 1
        return response

    with patch.object(SharePointReader, "_send_get_with_retry", side_effect=mock_get):
        items = reader._get_all_items_with_pagination(
            "https://graph.microsoft.com/v1.0/endpoint"
        )

    assert len(items) == 5
    assert items[0]["id"] == "item1"
    assert items[4]["id"] == "item5"
    assert call_count == 3  # Verify all three pages were fetched


def test_get_all_items_with_pagination_single_page():
    """Test pagination helper with single page (no nextLink)."""
    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
    )
    reader._authorization_headers = {"Authorization": "Bearer dummy_token"}

    single_page_response = MagicMock()
    single_page_response.json.return_value = {
        "value": [{"id": "item1"}, {"id": "item2"}],
        # No nextLink
    }

    with patch.object(
        SharePointReader, "_send_get_with_retry", return_value=single_page_response
    ):
        items = reader._get_all_items_with_pagination(
            "https://graph.microsoft.com/v1.0/endpoint"
        )

    assert len(items) == 2
    assert items[0]["id"] == "item1"
    assert items[1]["id"] == "item2"


def test_get_all_items_with_pagination_empty_result():
    """Test pagination helper with empty result."""
    reader = SharePointReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
    )
    reader._authorization_headers = {"Authorization": "Bearer dummy_token"}

    empty_response = MagicMock()
    empty_response.json.return_value = {
        "value": [],
    }

    with patch.object(
        SharePointReader, "_send_get_with_retry", return_value=empty_response
    ):
        items = reader._get_all_items_with_pagination(
            "https://graph.microsoft.com/v1.0/endpoint"
        )

    assert len(items) == 0
