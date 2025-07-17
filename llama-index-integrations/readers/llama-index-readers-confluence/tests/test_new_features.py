"""Tests for new ConfluenceReader features: callbacks, custom parsers, event system."""

from unittest.mock import MagicMock, patch
import pytest
import os

from llama_index.readers.confluence import ConfluenceReader
from llama_index.readers.confluence.event import (
    FileType,
    PageDataFetchStartedEvent,
    AttachmentProcessedEvent,
    AttachmentFailedEvent,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler


class TestCustomParsersAndFolder:
    """Test custom parsers and custom folder functionality."""

    def test_custom_folder_without_parsers_raises_error(self):
        """Test that custom_folder raises error when used without custom_parsers."""
        with pytest.raises(
            ValueError,
            match="custom_folder can only be used when custom_parsers are provided",
        ):
            ConfluenceReader(
                base_url="https://example.atlassian.net/wiki",
                api_token="test_token",
                custom_folder="/tmp/test",
            )

    def test_custom_parsers_with_custom_folder(self):
        """Test that custom_parsers and custom_folder work together."""
        mock_parser = MagicMock()
        custom_parsers = {FileType.PDF: mock_parser}

        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
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

        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            custom_parsers=custom_parsers,
        )

        assert reader.custom_parsers == custom_parsers
        assert reader.custom_folder == os.getcwd()
        assert reader.custom_parser_manager is not None

    def test_no_custom_parsers_no_folder(self):
        """Test that without custom_parsers, custom_folder is None and no parser manager is created."""
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", api_token="test_token"
        )

        assert reader.custom_parsers == {}
        assert reader.custom_folder is None
        assert reader.custom_parser_manager is None


class TestCallbacks:
    """Test callback functionality."""

    def test_attachment_callback_functionality(self):
        """Test that attachment callback is properly stored and functional."""

        def attachment_filter(
            media_type: str, file_size: int, title: str
        ) -> tuple[bool, str]:
            if file_size > 1000000:  # 1MB limit
                return False, "File too large"
            if media_type in ["application/zip"]:
                return False, "Unsupported file type"
            return True, ""

        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            process_attachment_callback=attachment_filter,
        )

        assert reader.process_attachment_callback == attachment_filter

        # Test callback behavior
        should_process, reason = attachment_filter(
            "application/pdf", 500000, "small.pdf"
        )
        assert should_process is True
        assert reason == ""

        should_process, reason = attachment_filter(
            "application/pdf", 2000000, "large.pdf"
        )
        assert should_process is False
        assert reason == "File too large"

        should_process, reason = attachment_filter(
            "application/zip", 500000, "archive.zip"
        )
        assert should_process is False
        assert reason == "Unsupported file type"

    def test_document_callback_functionality(self):
        """Test that document callback is properly stored and functional."""
        excluded_pages = ["page1", "page2"]

        def document_filter(page_id: str) -> bool:
            return page_id not in excluded_pages

        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            process_document_callback=document_filter,
        )

        assert reader.process_document_callback == document_filter

        # Test callback behavior
        assert document_filter("normal_page") is True
        assert document_filter("page1") is False
        assert document_filter("page2") is False

    @patch("html2text.HTML2Text")
    def test_document_callback_in_process_page(self, mock_html2text_class):
        """Test that document callback is used during page processing."""
        mock_text_maker = MagicMock()
        mock_text_maker.handle.return_value = "processed text"
        mock_html2text_class.return_value = mock_text_maker

        def document_filter(page_id: str) -> bool:
            return page_id != "skip_page"

        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            process_document_callback=document_filter,
        )
        reader.confluence = MagicMock()  # Mock the confluence client

        # Test normal page processing
        normal_page = {
            "id": "normal_page",
            "title": "Normal Page",
            "status": "current",
            "body": {"export_view": {"value": "<p>Content</p>"}},
            "_links": {"webui": "/pages/123"},
        }

        result = reader.process_page(normal_page, False, mock_text_maker)
        assert result is not None
        assert result.doc_id == "normal_page"

        # Test skipped page
        skip_page = {
            "id": "skip_page",
            "title": "Skip Page",
            "status": "current",
            "body": {"export_view": {"value": "<p>Content</p>"}},
            "_links": {"webui": "/pages/456"},
        }

        result = reader.process_page(skip_page, False, mock_text_maker)
        assert result is None


class TestEventSystem:
    """Test event system functionality."""

    def test_event_system_subscription_and_notification(self):
        """Test that event system can handle event subscriptions and notifications."""
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", api_token="test_token"
        )

        events_received = []

        class TestEventHandler(BaseEventHandler):
            def handle(self, event):
                events_received.append(event)

        class PageEventHandler(BaseEventHandler):
            def handle(self, event):
                if isinstance(event, PageDataFetchStartedEvent):
                    events_received.append(f"PAGE_EVENT: {event.page_id}")

        # Subscribe to events using new event system
        dispatcher = get_dispatcher("test_new_features_subscription")
        general_handler = TestEventHandler()
        page_handler = PageEventHandler()

        dispatcher.add_event_handler(general_handler)
        dispatcher.add_event_handler(page_handler)

        # Create and emit a page event
        page_event = PageDataFetchStartedEvent(page_id="test_page")
        dispatcher.event(page_event)

        # Check that both handlers received the event
        assert len(events_received) == 2
        assert "PAGE_EVENT: test_page" in events_received
        assert any(
            isinstance(event, PageDataFetchStartedEvent) for event in events_received
        )

        # Clean up
        for handler in [general_handler, page_handler]:
            if handler in dispatcher.event_handlers:
                dispatcher.event_handlers.remove(handler)

    def test_event_system_attachment_events(self):
        """Test event system with attachment events."""
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", api_token="test_token"
        )

        attachment_events = []

        class AttachmentEventHandler(BaseEventHandler):
            def handle(self, event):
                if isinstance(event, (AttachmentProcessedEvent, AttachmentFailedEvent)):
                    attachment_events.append(event)

        dispatcher = get_dispatcher("test_new_features_attachment")
        attachment_handler = AttachmentEventHandler()
        dispatcher.add_event_handler(attachment_handler)

        # Test attachment processed event
        processed_event = AttachmentProcessedEvent(
            page_id="page123",
            attachment_id="att456",
            attachment_name="document.pdf",
            attachment_type=FileType.PDF,
            attachment_size=1024,
            attachment_link="http://example.com/att456",
        )

        dispatcher.event(processed_event)

        # Test attachment failed event
        failed_event = AttachmentFailedEvent(
            page_id="page123",
            attachment_id="att789",
            attachment_name="broken.pdf",
            attachment_type=FileType.PDF,
            attachment_size=2048,
            attachment_link="http://example.com/att789",
            error="Processing failed",
        )

        dispatcher.event(failed_event)

        assert len(attachment_events) == 2
        assert any(
            isinstance(event, AttachmentProcessedEvent) for event in attachment_events
        )
        assert any(
            isinstance(event, AttachmentFailedEvent) for event in attachment_events
        )

        # Clean up
        if attachment_handler in dispatcher.event_handlers:
            dispatcher.event_handlers.remove(attachment_handler)

    def test_event_system_handler_removal(self):
        """Test event system handler removal functionality."""
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", api_token="test_token"
        )

        events_received = []

        class TestEventHandler(BaseEventHandler):
            def handle(self, event):
                events_received.append(event)

        # Add and then remove event handler
        dispatcher = get_dispatcher("test_new_features_removal")
        event_handler = TestEventHandler()
        dispatcher.add_event_handler(event_handler)
        if event_handler in dispatcher.event_handlers:
            dispatcher.event_handlers.remove(event_handler)

        # Create and emit event
        page_event = PageDataFetchStartedEvent(page_id="test_page")
        dispatcher.event(page_event)

        # Should not receive any events since we removed the handler
        assert len(events_received) == 0


class TestErrorHandling:
    """Test error handling functionality."""

    def test_fail_on_error_default_true(self):
        """Test that fail_on_error defaults to True."""
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", api_token="test_token"
        )

        assert reader.fail_on_error is True

    def test_fail_on_error_explicit_false(self):
        """Test that fail_on_error can be set to False."""
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            fail_on_error=False,
        )

        assert reader.fail_on_error is False

    def test_fail_on_error_explicit_true(self):
        """Test that fail_on_error can be explicitly set to True."""
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            fail_on_error=True,
        )

        assert reader.fail_on_error is True


class TestLogging:
    """Test logging functionality."""

    def test_custom_logger(self):
        """Test that custom logger is properly stored."""
        import logging

        custom_logger = logging.getLogger("test_confluence_logger")

        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="test_token",
            logger=custom_logger,
        )

        assert reader.logger == custom_logger

    def test_default_logger(self):
        """Test that default logger is used when none provided."""
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", api_token="test_token"
        )

        # Should use internal logger
        assert reader.logger is not None
        assert hasattr(reader.logger, "info")
        assert hasattr(reader.logger, "error")
        assert hasattr(reader.logger, "warning")
