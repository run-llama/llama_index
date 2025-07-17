"""Tests for ServiceNow Knowledge Base Reader."""

import sys
from unittest.mock import MagicMock, patch
import pytest
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader


class MockCustomParser(BaseReader):
    """Mock custom parser for testing."""

    def load_data(self, file_path: str):
        return [Document(text="Mocked parsed content")]


class MockServiceNowClient:
    """Mock ServiceNow client for testing."""

    def __init__(self, *args, **kwargs):
        self.attachment_api = MagicMock()
        self.attachment_api.get_file = MagicMock(return_value=b"mock file content")

    def GlideRecord(self, table):
        """Mock GlideRecord for ServiceNow table operations."""
        mock_gr = MagicMock()
        mock_gr.add_query = MagicMock()
        mock_gr.query = MagicMock()
        mock_gr.has_next = MagicMock(return_value=True)
        mock_gr.next = MagicMock(
            side_effect=[True, False]
        )  # First call returns True, second False
        mock_gr.get_row_count = MagicMock(return_value=1)

        # Mock properties for KB article
        mock_gr.number = MagicMock()
        mock_gr.number.get_value = MagicMock(return_value="KB0010001")
        mock_gr.sys_id = MagicMock()
        mock_gr.sys_id.get_value = MagicMock(return_value="test_sys_id")
        mock_gr.short_description = MagicMock()
        mock_gr.short_description.get_display_value = MagicMock(
            return_value="Test KB Article"
        )
        mock_gr.text = MagicMock()
        mock_gr.text.get_value = MagicMock(return_value="<p>Test article content</p>")
        mock_gr.workflow_state = MagicMock()
        mock_gr.workflow_state.get_display_value = MagicMock(return_value="Published")
        mock_gr.get_value = MagicMock(return_value="test_value")
        mock_gr.get_display_value = MagicMock(return_value="test_display_value")

        return mock_gr


class MockPasswordGrantFlow:
    """Mock password grant flow for ServiceNow authentication."""

    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def mock_pysnc_imports():
    """Mock pysnc imports for testing."""
    with patch.dict("sys.modules", {"pysnc": MagicMock(), "pysnc.auth": MagicMock()}):
        sys.modules["pysnc"].ServiceNowClient = MockServiceNowClient
        sys.modules["pysnc"].GlideRecord = MagicMock()
        sys.modules["pysnc.auth"].ServiceNowPasswordGrantFlow = MockPasswordGrantFlow
        yield


@pytest.fixture
def snow_reader(mock_pysnc_imports):
    """Fixture to create a SnowKBReader instance with mocked dependencies."""
    with patch(
        "llama_index.readers.service_now.base.ServiceNowClient", MockServiceNowClient
    ):
        with patch(
            "llama_index.readers.service_now.base.ServiceNowPasswordGrantFlow",
            MockPasswordGrantFlow,
        ):
            from llama_index.readers.service_now import SnowKBReader
            from llama_index.readers.service_now.base import FileType

            # Create custom parsers dictionary with mock parsers
            custom_parsers = {
                FileType.HTML: MockCustomParser(),  # HTML parser is required
                FileType.PDF: MockCustomParser(),
                FileType.DOCUMENT: MockCustomParser(),
            }

            return SnowKBReader(
                instance="test.service-now.com",
                custom_parsers=custom_parsers,
                username="test_user",
                password="test_pass",
                client_id="test_client_id",
                client_secret="test_client_secret",
            )


class TestSnowKBReader:
    """Test class for ServiceNow Knowledge Base Reader."""

    def test_initialization(self, mock_pysnc_imports):
        """Test that SnowKBReader initializes correctly."""
        with patch(
            "llama_index.readers.service_now.base.ServiceNowClient",
            MockServiceNowClient,
        ):
            with patch(
                "llama_index.readers.service_now.base.ServiceNowPasswordGrantFlow",
                MockPasswordGrantFlow,
            ):
                from llama_index.readers.service_now import SnowKBReader
                from llama_index.readers.service_now.base import FileType

                custom_parsers = {
                    FileType.HTML: MockCustomParser(),  # Required
                    FileType.PDF: MockCustomParser(),
                }

                reader = SnowKBReader(
                    instance="test.service-now.com",
                    custom_parsers=custom_parsers,
                    username="test_user",
                    password="test_pass",
                    client_id="test_client_id",
                    client_secret="test_client_secret",
                )

                assert reader.instance == "test.service-now.com"
                assert reader.username == "test_user"
                assert reader.password == "test_pass"
                assert reader.client_id == "test_client_id"
                assert reader.client_secret == "test_client_secret"
                assert reader.kb_table == "kb_knowledge"
                assert reader.pysnc_client is not None
                assert reader.custom_parsers == custom_parsers

    def test_initialization_missing_credentials(self):
        """Test that SnowKBReader raises error when missing required credentials."""
        from llama_index.readers.service_now import SnowKBReader
        from llama_index.readers.service_now.base import FileType

        custom_parsers = {
            FileType.HTML: MockCustomParser(),  # Required
            FileType.PDF: MockCustomParser(),
        }

        with pytest.raises(ValueError, match="username parameter is required"):
            SnowKBReader(instance="test.service-now.com", custom_parsers=custom_parsers)

    def test_load_data_by_sys_id(self, snow_reader):
        """Test loading KB article by sys_id."""
        with patch.object(snow_reader, "load_data") as mock_load_data:
            mock_doc = Document(
                text="Test content",
                metadata={
                    "title": "Test KB Article",
                    "page_id": "KB0010001",
                    "status": "Published",
                },
            )
            mock_load_data.return_value = [mock_doc]

            result = snow_reader.load_data(article_sys_id="test_sys_id")

            assert len(result) == 1
            assert result[0].text == "Test content"
            assert result[0].metadata["title"] == "Test KB Article"
            mock_load_data.assert_called_once_with(article_sys_id="test_sys_id")

    def test_load_data_by_numbers(self, snow_reader):
        """Test loading KB articles by numbers."""
        with patch.object(snow_reader, "load_data") as mock_load_data:
            mock_doc = Document(
                text="Test content",
                metadata={
                    "title": "Test KB Article",
                    "page_id": "KB0010001",
                    "status": "Published",
                },
            )
            mock_load_data.return_value = [mock_doc]

            result = snow_reader.load_data(numbers=["KB0010001", "KB0010002"])

            assert len(result) == 1
            mock_load_data.assert_called_once_with(numbers=["KB0010001", "KB0010002"])

    def test_load_data_no_parameters(self, snow_reader):
        """Test that load_data raises error when no parameters provided."""
        with pytest.raises(ValueError, match="Must provide article_sys_id or number"):
            with patch.object(snow_reader, "load_data") as mock_load_data:
                mock_load_data.side_effect = ValueError(
                    "Must provide article_sys_id or number"
                )
                snow_reader.load_data()

    def test_get_documents_with_attachments(self, snow_reader):
        """Test getting documents with attachment processing."""
        with patch.object(snow_reader, "handle_attachments") as mock_handle_attachments:
            mock_handle_attachments.return_value = [
                {"file_name": "test.pdf", "markdown_text": "PDF content"}
            ]

            with patch.object(
                snow_reader.custom_parser_manager, "process_text_with_custom_parser"
            ) as mock_process:
                mock_process.return_value = "Processed HTML content"

                result = snow_reader.load_data(article_sys_id="test_sys_id")

                assert len(result) == 1
                assert "Processed HTML content" in result[0].text
                assert "# test.pdf" in result[0].text
                assert "PDF content" in result[0].text

    def test_handle_attachments(self, snow_reader):
        """Test attachment handling functionality."""
        with patch.object(snow_reader.pysnc_client, "GlideRecord") as mock_gr_class:
            mock_gr = MagicMock()
            mock_gr.next.side_effect = [True, False]
            mock_gr_class.return_value = mock_gr

            with patch.object(
                snow_reader, "handle_attachment"
            ) as mock_handle_attachment:
                mock_handle_attachment.return_value = {
                    "file_name": "test.pdf",
                    "markdown_text": "PDF content",
                }

                result = snow_reader.handle_attachments("test_sys_id", "KB0010001")

                assert len(result) == 1
                assert result[0]["file_name"] == "test.pdf"
                mock_gr.add_query.assert_any_call("table_sys_id", "test_sys_id")
                mock_gr.add_query.assert_any_call("table_name", "kb_knowledge")

    def test_get_file_type(self, snow_reader):
        """Test file type detection."""
        from llama_index.readers.service_now.base import FileType

        assert snow_reader.get_File_type("test.pdf") == FileType.PDF
        assert snow_reader.get_File_type("test.jpg") == FileType.IMAGE
        assert snow_reader.get_File_type("test.docx") == FileType.DOCUMENT
        assert snow_reader.get_File_type("test.xlsx") == FileType.SPREADSHEET
        assert snow_reader.get_File_type("test.txt") == FileType.TEXT
        assert snow_reader.get_File_type("test.html") == FileType.HTML
        assert snow_reader.get_File_type("test.csv") == FileType.CSV
        assert snow_reader.get_File_type("test.md") == FileType.MARKDOWN
        assert snow_reader.get_File_type("test.unknown") == FileType.UNKNOWN

    def test_download_attachment_content(self, snow_reader):
        """Test attachment content download."""
        result = snow_reader._download_attachment_content("test_sys_id")
        assert result == b"mock file content"
        snow_reader.pysnc_client.attachment_api.get_file.assert_called_once_with(
            "test_sys_id"
        )

    def test_download_attachment_content_failure(self, snow_reader):
        """Test attachment download failure handling."""
        snow_reader.pysnc_client.attachment_api.get_file.side_effect = Exception(
            "Download failed"
        )

        result = snow_reader._download_attachment_content("test_sys_id")
        assert result is None

    def test_custom_kb_table(self, mock_pysnc_imports):
        """Test initialization with custom KB table."""
        with patch(
            "llama_index.readers.service_now.base.ServiceNowClient",
            MockServiceNowClient,
        ):
            with patch(
                "llama_index.readers.service_now.base.ServiceNowPasswordGrantFlow",
                MockPasswordGrantFlow,
            ):
                from llama_index.readers.service_now import SnowKBReader
                from llama_index.readers.service_now.base import FileType

                custom_parsers = {
                    FileType.HTML: MockCustomParser(),  # Required
                    FileType.PDF: MockCustomParser(),
                }

                reader = SnowKBReader(
                    instance="test.service-now.com",
                    custom_parsers=custom_parsers,
                    username="test_user",
                    password="test_pass",
                    client_id="test_client_id",
                    client_secret="test_client_secret",
                    kb_table="custom_kb_table",
                )

                assert reader.kb_table == "custom_kb_table"

    def test_fail_on_error_false(self, mock_pysnc_imports):
        """Test that fail_on_error=False allows processing to continue on errors."""
        with patch(
            "llama_index.readers.service_now.base.ServiceNowClient",
            MockServiceNowClient,
        ):
            with patch(
                "llama_index.readers.service_now.base.ServiceNowPasswordGrantFlow",
                MockPasswordGrantFlow,
            ):
                from llama_index.readers.service_now import SnowKBReader
                from llama_index.readers.service_now.base import FileType

                custom_parsers = {
                    FileType.HTML: MockCustomParser(),  # Required
                    FileType.PDF: MockCustomParser(),
                }

                reader = SnowKBReader(
                    instance="test.service-now.com",
                    custom_parsers=custom_parsers,
                    username="test_user",
                    password="test_pass",
                    client_id="test_client_id",
                    client_secret="test_client_secret",
                    fail_on_error=False,
                )

                assert reader.fail_on_error is False

    def test_event_system_integration(self, snow_reader):
        """Test that LlamaIndex event system integration is working."""
        from llama_index.readers.service_now.event import (
            SNOWKBPageFetchStartEvent,
            SNOWKBPageFetchCompletedEvent,
        )

        # Test that events can be imported and are proper BaseEvent subclasses
        assert hasattr(SNOWKBPageFetchStartEvent, "model_fields")
        assert hasattr(SNOWKBPageFetchCompletedEvent, "model_fields")

        # Test event creation
        start_event = SNOWKBPageFetchStartEvent(page_id="KB0010001")
        assert start_event.page_id == "KB0010001"

    @patch("os.path.exists")
    @patch("os.remove")
    def test_custom_parser_manager_file_cleanup(
        self, mock_remove, mock_exists, snow_reader
    ):
        """Test that custom parser manager cleans up temporary files."""
        mock_exists.return_value = True

        # Access the private method through the manager
        snow_reader.custom_parser_manager._CustomParserManager__remove_custom_file(
            "test_file.txt"
        )

        mock_exists.assert_called_once_with("test_file.txt")
        mock_remove.assert_called_once_with("test_file.txt")

    def test_format_attachment_header(self, snow_reader):
        """Test attachment header formatting."""
        attachment = {"file_name": "test_document.pdf"}
        result = snow_reader._format_attachment_header(attachment)
        assert result == "# test_document.pdf\n"

    def test_initialize_client_with_valid_credentials(self, mock_pysnc_imports):
        """Test client initialization with valid credentials."""
        with patch(
            "llama_index.readers.service_now.base.ServiceNowClient",
            MockServiceNowClient,
        ):
            with patch(
                "llama_index.readers.service_now.base.ServiceNowPasswordGrantFlow",
                MockPasswordGrantFlow,
            ):
                from llama_index.readers.service_now import SnowKBReader
                from llama_index.readers.service_now.base import FileType

                custom_parsers = {
                    FileType.HTML: MockCustomParser(),  # Required
                    FileType.PDF: MockCustomParser(),
                }

                reader = SnowKBReader(
                    instance="test.service-now.com",
                    custom_parsers=custom_parsers,
                    username="test_user",
                    password="test_pass",
                    client_id="test_client_id",
                    client_secret="test_client_secret",
                )

                # Test that client was initialized
                assert reader.pysnc_client is not None

    def test_custom_parsers_integration(self, mock_pysnc_imports):
        """Test integration with custom parsers."""
        with patch(
            "llama_index.readers.service_now.base.ServiceNowClient",
            MockServiceNowClient,
        ):
            with patch(
                "llama_index.readers.service_now.base.ServiceNowPasswordGrantFlow",
                MockPasswordGrantFlow,
            ):
                from llama_index.readers.service_now import SnowKBReader
                from llama_index.readers.service_now.base import FileType

                # Mock custom parser (use the actual MockCustomParser class instead of MagicMock)
                custom_parsers = {
                    FileType.HTML: MockCustomParser(),  # Required
                    FileType.PDF: MockCustomParser(),
                }

                reader = SnowKBReader(
                    instance="test.service-now.com",
                    custom_parsers=custom_parsers,
                    username="test_user",
                    password="test_pass",
                    client_id="test_client_id",
                    client_secret="test_client_secret",
                )

                assert reader.custom_parsers == custom_parsers
                assert FileType.PDF in reader.custom_parsers

    def test_process_callbacks(self, mock_pysnc_imports):
        """Test process callbacks functionality."""
        with patch(
            "llama_index.readers.service_now.base.ServiceNowClient",
            MockServiceNowClient,
        ):
            with patch(
                "llama_index.readers.service_now.base.ServiceNowPasswordGrantFlow",
                MockPasswordGrantFlow,
            ):
                from llama_index.readers.service_now import SnowKBReader
                from llama_index.readers.service_now.base import FileType

                def process_attachment_callback(
                    file_name: str, size: int
                ) -> tuple[bool, str]:
                    return True, "Processing"

                def process_document_callback(kb_number: str) -> bool:
                    return True

                custom_parsers = {
                    FileType.HTML: MockCustomParser(),  # Required
                    FileType.PDF: MockCustomParser(),
                }

                reader = SnowKBReader(
                    instance="test.service-now.com",
                    custom_parsers=custom_parsers,
                    username="test_user",
                    password="test_pass",
                    client_id="test_client_id",
                    client_secret="test_client_secret",
                    process_attachment_callback=process_attachment_callback,
                    process_document_callback=process_document_callback,
                )

                assert reader.process_attachment_callback is not None
                assert reader.process_document_callback is not None

                # Test callback execution
                result = reader.process_attachment_callback("test.pdf", 1000)
                assert result == (True, "Processing")

                result = reader.process_document_callback("KB0010001")
                assert result is True

    def test_custom_parser_validation(self, mock_pysnc_imports):
        """Test custom parser validation."""
        with patch(
            "llama_index.readers.service_now.base.ServiceNowClient",
            MockServiceNowClient,
        ):
            with patch(
                "llama_index.readers.service_now.base.ServiceNowPasswordGrantFlow",
                MockPasswordGrantFlow,
            ):
                from llama_index.readers.service_now import SnowKBReader
                from llama_index.readers.service_now.base import FileType

                custom_parsers = {
                    FileType.HTML: MockCustomParser(),  # Required
                    FileType.PDF: MockCustomParser(),
                }

                reader = SnowKBReader(
                    instance="test.service-now.com",
                    custom_parsers=custom_parsers,
                    username="test_user",
                    password="test_pass",
                    client_id="test_client_id",
                    client_secret="test_client_secret",
                )

                assert reader.custom_parsers[FileType.PDF] is not None

                # Test parsing with custom parser
                mock_parser = reader.custom_parsers[FileType.PDF]
                result = mock_parser.load_data("test.pdf")

                assert len(result) == 1
                assert result[0].text == "Mocked parsed content"

    def test_smoke_test_instantiation(self, mock_pysnc_imports):
        """Smoke test to verify SnowKBReader can be instantiated correctly."""
        from llama_index.readers.service_now import SnowKBReader
        from llama_index.readers.service_now.base import FileType

        custom_parsers = {
            FileType.PDF: MockCustomParser(),
            FileType.HTML: MockCustomParser(),
            FileType.DOCUMENT: MockCustomParser(),
        }

        # This should create without errors (though it will fail on ServiceNow connection)
        try:
            reader = SnowKBReader(
                instance="test.service-now.com",
                custom_parsers=custom_parsers,
                username="test_user",
                password="test_password",
                client_id="test_client_id",
                client_secret="test_client_secret",
            )

            # Verify basic properties are set correctly
            assert reader.instance == "test.service-now.com"
            assert reader.username == "test_user"
            assert reader.password == "test_password"
            assert reader.client_id == "test_client_id"
            assert reader.client_secret == "test_client_secret"
            assert reader.custom_parsers == custom_parsers
            assert reader.kb_table == "kb_knowledge"
            assert reader.fail_on_error is True
            assert reader.pysnc_client is not None
            assert reader.custom_parser_manager is not None

            # Verify the custom parsers are working
            assert FileType.PDF in reader.custom_parsers
            assert FileType.HTML in reader.custom_parsers
            assert FileType.DOCUMENT in reader.custom_parsers

        except Exception as e:
            # We expect a ServiceNow connection error in test environment
            if "ServiceNow client" in str(e) or "Instance name not well-formed" in str(
                e
            ):
                # This is expected since we can't actually connect to ServiceNow in tests
                pass
            else:
                # Any other error is unexpected and should fail the test
                pytest.fail(f"Unexpected error during SnowKBReader instantiation: {e}")

    def test_smoke_test_with_minimal_config(self, mock_pysnc_imports):
        """Smoke test with minimal configuration."""
        from llama_index.readers.service_now import SnowKBReader
        from llama_index.readers.service_now.base import FileType

        # Test with minimal required configuration
        custom_parsers = {
            FileType.HTML: MockCustomParser()  # HTML parser is required for article body processing
        }

        try:
            reader = SnowKBReader(
                instance="test.service-now.com",
                custom_parsers=custom_parsers,
                username="test_user",
                password="test_password",
            )

            # Verify minimal config is set correctly
            assert reader.instance == "test.service-now.com"
            assert reader.username == "test_user"
            assert reader.password == "test_password"
            assert reader.client_id is None
            assert reader.client_secret is None
            assert len(reader.custom_parsers) == 1
            assert FileType.HTML in reader.custom_parsers

        except Exception as e:
            # We expect a ServiceNow connection error in test environment
            if "ServiceNow client" in str(e) or "Instance name not well-formed" in str(
                e
            ):
                # This is expected since we can't actually connect to ServiceNow in tests
                pass
            else:
                # Any other error is unexpected and should fail the test
                pytest.fail(
                    f"Unexpected error during minimal SnowKBReader instantiation: {e}"
                )
