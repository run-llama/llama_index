"""Tests for new SharePointReader features: callbacks, custom parsers, event system."""

from unittest.mock import MagicMock
import pytest
import os

from llama_index.readers.microsoft_sharepoint import SharePointReader
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


class TestCustomParsersAndFolder:
    """Test custom parsers and custom folder functionality."""

    def test_custom_parsers_with_custom_folder(self):
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
            custom_folder="/tmp/test",
        )

        assert reader.custom_parsers == custom_parsers
        assert reader.custom_folder == "/tmp/test"
        assert reader.custom_parser_manager is not None

    def test_custom_parsers_with_default_folder(self):
        """Test that custom_parsers uses current directory when custom_folder not specified."""
        mock_parser = MagicMock()
        custom_parsers = {FileType.PDF: mock_parser}

        reader = SharePointReader(
            client_id="dummy_client_id",
            client_secret="dummy_client_secret",
            tenant_id="dummy_tenant_id",
            sharepoint_site_name="dummy_site_name",
            sharepoint_folder_path="dummy_folder_path",
            custom_parsers=custom_parsers,
        )

        assert reader.custom_parsers == custom_parsers
        assert reader.custom_folder == os.getcwd()
        assert reader.custom_parser_manager is not None

    def test_no_custom_parsers_no_folder(self):
        """Test that without custom_parsers, custom_folder is None and no parser manager is created."""
        reader = SharePointReader(
            client_id="dummy_client_id",
            client_secret="dummy_client_secret",
            tenant_id="dummy_tenant_id",
            sharepoint_site_name="dummy_site_name",
            sharepoint_folder_path="dummy_folder_path",
        )

        assert reader.custom_parsers == {}
        assert reader.custom_folder is None
        assert reader.custom_parser_manager is None

    def test_custom_folder_without_parsers_raises(self):
        """Test that custom_folder raises error when used without custom_parsers."""
        with pytest.raises(ValueError) as excinfo:
            SharePointReader(
                client_id="dummy_client_id",
                client_secret="dummy_client_secret",
                tenant_id="dummy_tenant_id",
                sharepoint_site_name="dummy_site_name",
                sharepoint_folder_path="dummy_folder_path",
                custom_folder="/tmp/test",
            )
        assert "custom_folder can only be used when custom_parsers are provided" in str(
            excinfo.value
        )

    def test_custom_parser_usage(self, tmp_path):
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


class TestCallbacks:
    """Test callback functionality."""

    def test_document_callback_functionality(self):
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

        # Test callback behavior
        assert document_filter("normal_file") is True
        assert document_filter("file1") is False
        assert document_filter("file2") is False


class TestEventSystem:
    """Test event system functionality."""

    def test_event_system_page_events(self):
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

        # Check that events were received
        assert len(page_events) == 3
        event_types = [type(event).__name__ for event in page_events]
        assert "PageDataFetchStartedEvent" in event_types
        assert "PageDataFetchCompletedEvent" in event_types
        assert "PageSkippedEvent" in event_types

        # Clean up
        if page_handler in dispatcher.event_handlers:
            dispatcher.event_handlers.remove(page_handler)

    def test_event_system_page_failed_event(self):
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


class TestErrorHandling:
    """Test error handling functionality."""

    def test_fail_on_error_default_true(self):
        """Test that fail_on_error defaults to True."""
        reader = SharePointReader(
            client_id="dummy_client_id",
            client_secret="dummy_client_secret",
            tenant_id="dummy_tenant_id",
            sharepoint_site_name="dummy_site_name",
            sharepoint_folder_path="dummy_folder_path",
        )

        assert reader.fail_on_error is True

    def test_fail_on_error_explicit_false(self):
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

    def test_fail_on_error_explicit_true(self):
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