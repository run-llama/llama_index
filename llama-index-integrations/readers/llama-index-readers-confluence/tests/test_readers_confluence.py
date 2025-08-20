from unittest.mock import patch, MagicMock

import pytest
from llama_index.readers.confluence import ConfluenceReader
from llama_index.readers.confluence.event import (
    FileType,
    PageDataFetchStartedEvent,
    AttachmentProcessedEvent,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler


class MockConfluence:
    def __init__(self, *args, **kwargs) -> None:
        pass


@pytest.fixture(autouse=True)
def mock_atlassian_confluence(monkeypatch):
    monkeypatch.setattr("llama_index.readers.confluence", MockConfluence)


def test_confluence_reader_with_oauth2():
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        oauth2={
            "client_id": "example_client_id",
            "token": {"access_token": "example_token", "token_type": "Bearer"},
        },
    )
    assert reader.confluence is not None


def test_confluence_reader_with_api_token():
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
    )
    assert reader.confluence is not None


def test_confluence_reader_with_cookies():
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        cookies={"key": "value"},
    )
    assert reader.confluence is not None


def test_confluence_reader_with_client_args():
    with patch("atlassian.Confluence") as MockConstructor:
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="example_api_token",
            client_args={"backoff_and_retry": True},
        )
        assert reader.confluence is not None
        MockConstructor.assert_called_once_with(
            url="https://example.atlassian.net/wiki",
            token="example_api_token",
            cloud=True,
            backoff_and_retry=True,
        )


def test_confluence_reader_with_basic_auth():
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        user_name="example_user",
        password="example_password",
    )
    assert reader.confluence is not None


def test_confluence_reader_with_env_api_token(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_API_TOKEN", "env_api_token")
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
    )
    assert reader.confluence is not None
    monkeypatch.delenv("CONFLUENCE_API_TOKEN")


def test_confluence_reader_with_env_basic_auth(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_USERNAME", "env_user")
    monkeypatch.setenv("CONFLUENCE_PASSWORD", "env_password")
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
    )
    assert reader.confluence is not None
    monkeypatch.delenv("CONFLUENCE_USERNAME")
    monkeypatch.delenv("CONFLUENCE_PASSWORD")


def test_confluence_reader_without_credentials():
    with pytest.raises(ValueError) as excinfo:
        ConfluenceReader(base_url="https://example.atlassian.net/wiki")
    assert "Must set one of environment variables" in str(excinfo.value)


def test_confluence_reader_with_incomplete_basic_auth():
    with pytest.raises(ValueError) as excinfo:
        ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", user_name="example_user"
        )
    assert "Must set one of environment variables" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", password="example_password"
        )
    assert "Must set one of environment variables" in str(excinfo.value)


# Test new features
def test_confluence_reader_with_custom_folder_without_parsers():
    """Test that custom_folder raises error when used without custom_parsers."""
    with pytest.raises(ValueError) as excinfo:
        ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="example_api_token",
            custom_folder="/tmp/test",
        )
    assert "custom_folder can only be used when custom_parsers are provided" in str(
        excinfo.value
    )


def test_confluence_reader_with_custom_parsers_and_folder():
    """Test that custom_parsers and custom_folder work together."""
    mock_parser = MagicMock(spec=BaseReader)
    custom_parsers = {FileType.PDF: mock_parser}

    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
        custom_parsers=custom_parsers,
        custom_folder="/tmp/test",
    )

    assert reader.custom_parsers == custom_parsers
    assert reader.custom_folder == "/tmp/test"
    assert reader.custom_parser_manager is not None


def test_confluence_reader_with_custom_parsers_default_folder():
    """Test that custom_parsers uses default folder when custom_folder not specified."""
    import os

    mock_parser = MagicMock(spec=BaseReader)
    custom_parsers = {FileType.PDF: mock_parser}

    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
        custom_parsers=custom_parsers,
    )

    assert reader.custom_parsers == custom_parsers
    assert reader.custom_folder == os.getcwd()
    assert reader.custom_parser_manager is not None


def test_confluence_reader_callbacks():
    """Test that callbacks are properly stored and can be used."""

    def attachment_callback(
        media_type: str, file_size: int, title: str
    ) -> tuple[bool, str]:
        if file_size > 1000000:  # 1MB
            return False, "File too large"
        return True, ""

    def document_callback(page_id: str) -> bool:
        return page_id != "excluded_page"

    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
        process_attachment_callback=attachment_callback,
        process_document_callback=document_callback,
    )

    assert reader.process_attachment_callback == attachment_callback
    assert reader.process_document_callback == document_callback

    # Test callback functionality
    should_process, reason = reader.process_attachment_callback(
        "application/pdf", 2000000, "large_file.pdf"
    )
    assert should_process is False
    assert reason == "File too large"

    should_process, reason = reader.process_attachment_callback(
        "application/pdf", 500000, "small_file.pdf"
    )
    assert should_process is True
    assert reason == ""

    should_process = reader.process_document_callback("normal_page")
    assert should_process is True

    should_process = reader.process_document_callback("excluded_page")
    assert should_process is False


def test_confluence_reader_event_system():
    """Test that the new event system works correctly."""
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
    )

    # Test event handling
    events_received = []

    class TestEventHandler(BaseEventHandler):
        def handle(self, event):
            events_received.append(event)

    class PageEventHandler(BaseEventHandler):
        def handle(self, event):
            if isinstance(event, PageDataFetchStartedEvent):
                events_received.append(f"PAGE: {event.page_id}")

    # Add event handlers to dispatcher
    dispatcher = get_dispatcher("llama_index.readers.confluence.base")
    test_handler = TestEventHandler()
    page_handler = PageEventHandler()

    dispatcher.add_event_handler(test_handler)
    dispatcher.add_event_handler(page_handler)

    # Create and emit events manually to test the system
    page_event = PageDataFetchStartedEvent(page_id="test_page")
    attachment_event = AttachmentProcessedEvent(
        page_id="test_page",
        attachment_id="att_123",
        attachment_name="test.pdf",
        attachment_type=FileType.PDF,
        attachment_size=1000,
        attachment_link="http://example.com/att_123",
    )

    dispatcher.event(page_event)
    dispatcher.event(attachment_event)

    # Check that events were received
    assert len(events_received) == 3  # page_handler + 2 from test_handler
    assert "PAGE: test_page" in events_received
    assert any(
        isinstance(event, PageDataFetchStartedEvent) for event in events_received
    )
    assert any(isinstance(event, AttachmentProcessedEvent) for event in events_received)

    # Clean up handlers
    for handler in [test_handler, page_handler]:
        if handler in dispatcher.event_handlers:
            dispatcher.event_handlers.remove(handler)


def test_confluence_reader_fail_on_error_setting():
    """Test that fail_on_error setting is properly stored."""
    # Test default (True)
    reader1 = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
    )
    assert reader1.fail_on_error is True

    # Test explicit False
    reader2 = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
        fail_on_error=False,
    )
    assert reader2.fail_on_error is False

    # Test explicit True
    reader3 = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
        fail_on_error=True,
    )
    assert reader3.fail_on_error is True


@patch("html2text.HTML2Text")
def test_confluence_reader_process_page_with_callbacks(mock_html2text_class):
    """Test that callbacks are properly used during page processing."""
    mock_text_maker = MagicMock()
    mock_text_maker.handle.return_value = "processed text"
    mock_html2text_class.return_value = mock_text_maker

    # Mock the confluence API
    mock_confluence = MagicMock()

    def document_callback(page_id: str) -> bool:
        return page_id != "skip_this_page"

    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
        process_document_callback=document_callback,
    )
    reader.confluence = mock_confluence

    # Test page that should be processed
    page_data = {
        "id": "normal_page",
        "title": "Test Page",
        "status": "current",
        "body": {"export_view": {"value": "<p>Test content</p>"}},
        "_links": {"webui": "/pages/123"},
    }

    result = reader.process_page(page_data, False, mock_text_maker)
    assert result is not None
    assert result.doc_id == "normal_page"
    assert result.metadata["title"] == "Test Page"

    # Test page that should be skipped
    page_data_skip = {
        "id": "skip_this_page",
        "title": "Skip This Page",
        "status": "current",
        "body": {"export_view": {"value": "<p>Skip content</p>"}},
        "_links": {"webui": "/pages/456"},
    }

    result_skip = reader.process_page(page_data_skip, False, mock_text_maker)
    assert result_skip is None


def test_confluence_reader_logger_setting():
    """Test that custom logger is properly stored."""
    import logging

    custom_logger = logging.getLogger("test_logger")

    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
        logger=custom_logger,
    )

    assert reader.logger == custom_logger
