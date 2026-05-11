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
    monkeypatch.setattr("atlassian.Confluence", MockConfluence)


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
    """Test that custom_folder is accepted even without custom_parsers."""
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
        custom_folder="/tmp/test",
    )
    assert reader.custom_parser_manager.custom_folder == "/tmp/test"
    assert reader.custom_parser_manager is not None


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

    assert reader.custom_parser_manager.custom_parsers[FileType.PDF] is mock_parser
    assert reader.custom_parser_manager.custom_folder == "/tmp/test"
    assert reader.custom_parser_manager is not None


def test_confluence_reader_default_parser_manager_always_set():
    """Test that custom_parser_manager is always created (defaults loaded even without user parsers)."""
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
    )
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

    assert reader.custom_parser_manager.custom_parsers[FileType.PDF] is mock_parser
    assert reader.custom_parser_manager.custom_folder == os.getcwd()
    assert reader.custom_parser_manager is not None


def test_confluence_reader_page_html_and_html_parsers_are_independent():
    """FileType.PAGE_HTML and FileType.HTML override independently."""
    from llama_index.readers.confluence.event import FileType

    html_parser = MagicMock(spec=BaseReader)
    page_html_parser = MagicMock(spec=BaseReader)

    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
        custom_parsers={
            FileType.HTML: html_parser,
            FileType.PAGE_HTML: page_html_parser,
        },
    )
    manager = reader.custom_parser_manager
    assert manager.custom_parsers[FileType.HTML] is html_parser
    assert manager.custom_parsers[FileType.PAGE_HTML] is page_html_parser
    assert (
        manager.custom_parsers[FileType.HTML]
        is not manager.custom_parsers[FileType.PAGE_HTML]
    )


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


@patch(
    "llama_index.readers.confluence.base.CustomParserManager.process_with_custom_parser"
)
def test_confluence_reader_process_page_with_callbacks(mock_process):
    """Test that callbacks are properly used during page processing."""
    mock_process.return_value = ("processed text", {})

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

    result = reader.process_page(page_data, False)
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

    result_skip = reader.process_page(page_data_skip, False)
    assert result_skip is None


@patch(
    "llama_index.readers.confluence.base.CustomParserManager.process_with_custom_parser"
)
def test_process_page_parser_extra_info_is_merged(mock_process):
    """Custom parser extra_info keys are merged into the final Document."""
    mock_process.return_value = ("parsed text", {"custom_key": "custom_val"})

    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
    )
    reader.confluence = MagicMock()

    page_data = {
        "id": "page_123",
        "title": "My Page",
        "status": "current",
        "body": {"export_view": {"value": "<p>content</p>"}},
        "_links": {"webui": "/pages/123"},
    }

    result = reader.process_page(page_data, False)

    assert result is not None
    assert result.metadata["custom_key"] == "custom_val"
    # Defaults are still present
    assert result.metadata["title"] == "My Page"
    assert result.metadata["page_id"] == "page_123"
    assert result.metadata["status"] == "current"


@patch(
    "llama_index.readers.confluence.base.CustomParserManager.process_with_custom_parser"
)
def test_process_page_defaults_win_over_parser_extra_info(mock_process):
    """Default keys (title, page_id, status, url) always win over parser extra_info."""
    mock_process.return_value = (
        "parsed text",
        {"title": "parser title", "page_id": "parser_id", "extra": "value"},
    )

    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
    )
    reader.confluence = MagicMock()

    page_data = {
        "id": "page_456",
        "title": "Real Title",
        "status": "current",
        "body": {"export_view": {"value": "<p>content</p>"}},
        "_links": {"webui": "/pages/456"},
    }

    result = reader.process_page(page_data, False)

    assert result is not None
    # Confluence-sourced defaults win
    assert result.metadata["title"] == "Real Title"
    assert result.metadata["page_id"] == "page_456"
    # Non-conflicting parser key is still present
    assert result.metadata["extra"] == "value"


@patch(
    "llama_index.readers.confluence.base.CustomParserManager.process_with_custom_parser"
)
def test_process_page_include_attachments_uses_string_attachment_text(mock_process):
    """Regression test: include_attachments path should not fail on tuple parser outputs."""

    def mock_parser(file_type, file_content, extension):
        if file_type == FileType.PAGE_HTML:
            return "page body\n", {}
        if file_type == FileType.IMAGE:
            return "image extracted\n", {}
        return "", {}

    mock_process.side_effect = mock_parser

    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
    )

    request_response = MagicMock()
    request_response.status_code = 200
    request_response.content = b"fake-image-bytes"

    reader.confluence = MagicMock()
    reader.confluence.get_attachments_from_content.return_value = {
        "results": [
            {
                "id": "att-1",
                "title": "diagram.png",
                "metadata": {"mediaType": "image/png"},
                "extensions": {"fileSize": 1234},
                "_links": {
                    "download": "/download/attachments/123/diagram.png",
                    "webui": "/wiki/attachments/123/diagram.png",
                },
            }
        ]
    }
    reader.confluence.request.return_value = request_response

    page_data = {
        "id": "page-1",
        "title": "Attachment Test",
        "status": "current",
        "body": {"export_view": {"value": "<p>content</p>"}},
        "_links": {"webui": "/pages/123"},
    }

    result = reader.process_page(page_data, True)

    assert result is not None
    assert "page body" in result.text
    assert "# diagram.png" in result.text
    assert "image extracted" in result.text


@patch(
    "llama_index.readers.confluence.base.CustomParserManager.process_with_custom_parser"
)
def test_process_attachment_returns_strings_for_supported_media_types(mock_process):
    """Attachment processor should always return text list entries as strings."""
    mock_process.return_value = ("pdf extracted\n", {})

    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
    )

    request_response = MagicMock()
    request_response.status_code = 200
    request_response.content = b"fake-pdf-bytes"

    reader.confluence = MagicMock()
    reader.confluence.get_attachments_from_content.return_value = {
        "results": [
            {
                "id": "att-pdf",
                "title": "report.pdf",
                "metadata": {"mediaType": "application/pdf"},
                "extensions": {"fileSize": 2048},
                "_links": {
                    "download": "/download/attachments/123/report.pdf",
                    "webui": "/wiki/attachments/123/report.pdf",
                },
            }
        ]
    }
    reader.confluence.request.return_value = request_response

    texts = reader.process_attachment("page-1")

    assert len(texts) == 1
    assert isinstance(texts[0], str)
    assert "# report.pdf" in texts[0]
    assert "pdf extracted" in texts[0]


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
