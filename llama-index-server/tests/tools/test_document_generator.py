import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from llama_index.server.tools.document_generator import (
    OUTPUT_DIR,
    DocumentGenerator,
)


class TestDocumentGenerator:
    @pytest.fixture()
    def env_setup(self):  # type: ignore
        os.environ["FILESERVER_URL_PREFIX"] = "http://test-server"
        yield
        os.environ.pop("FILESERVER_URL_PREFIX", None)

    def test_validate_file_name(self) -> None:
        # Valid names
        assert DocumentGenerator._validate_file_name("valid-name") == "valid-name"

        # Invalid names
        with pytest.raises(ValueError):
            DocumentGenerator._validate_file_name("/invalid/path")

    @patch("os.makedirs")
    @patch("builtins.open")
    def test_write_to_file(self, mock_open, mock_makedirs):  # type: ignore
        content = BytesIO(b"test")
        DocumentGenerator._write_to_file(content, "path/file.txt")

        mock_makedirs.assert_called_once()
        mock_open.assert_called_once()
        mock_open.return_value.__enter__.return_value.write.assert_called_once_with(
            b"test"
        )

    @patch("markdown.markdown")
    def test_html_generation(self, mock_markdown):  # type: ignore
        mock_markdown.return_value = "<h1>Test</h1>"

        # Test HTML content generation
        assert DocumentGenerator._generate_html_content("# Test") == "<h1>Test</h1>"

        # Test full HTML generation
        html = DocumentGenerator._generate_html("<h1>Test</h1>")
        assert "<!DOCTYPE html>" in html
        assert "<h1>Test</h1>" in html

    @patch("xhtml2pdf.pisa.pisaDocument")
    def test_pdf_generation(self, mock_pisa):  # type: ignore
        # Success case
        mock_pisa.return_value = MagicMock(err=None)
        assert isinstance(DocumentGenerator._generate_pdf("test"), BytesIO)

        # Error case
        mock_pisa.return_value = MagicMock(err="Error")
        with pytest.raises(ValueError):
            DocumentGenerator._generate_pdf("test")

    @patch.multiple(
        DocumentGenerator,
        _generate_html_content=MagicMock(return_value="<h1>Test</h1>"),
        _generate_html=MagicMock(
            return_value="<html><body><h1>Test</h1></body></html>"
        ),
        _generate_pdf=MagicMock(return_value=BytesIO(b"pdf")),
        _write_to_file=MagicMock(),
    )
    def test_generate_document(self, env_setup):  # type: ignore
        # HTML generation
        url = DocumentGenerator.generate_document("# Test", "html", "test-doc")
        assert url == f"http://test-server/{OUTPUT_DIR}/test-doc.html"

        # PDF generation
        url = DocumentGenerator.generate_document("# Test", "pdf", "test-doc")
        assert url == f"http://test-server/{OUTPUT_DIR}/test-doc.pdf"

        # Invalid type
        with pytest.raises(ValueError):
            DocumentGenerator.generate_document("# Test", "invalid", "test-doc")

    def test_to_tool(self):  # type: ignore
        tool = DocumentGenerator().to_tool()
        # Check the function is correct
        assert tool.fn == DocumentGenerator.generate_document
        assert callable(tool.fn)
