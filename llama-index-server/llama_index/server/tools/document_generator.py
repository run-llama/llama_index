import logging
import os
import re
from enum import Enum
from io import BytesIO

from llama_index.core.tools.function_tool import FunctionTool

OUTPUT_DIR = "output/tools"


class DocumentType(Enum):
    PDF = "pdf"
    HTML = "html"


COMMON_STYLES = """
body {
    font-family: Arial, sans-serif;
    line-height: 1.3;
    color: #333;
}
h1, h2, h3, h4, h5, h6 {
    margin-top: 1em;
    margin-bottom: 0.5em;
}
p {
    margin-bottom: 0.7em;
}
code {
    background-color: #f4f4f4;
    padding: 2px 4px;
    border-radius: 4px;
}
pre {
    background-color: #f4f4f4;
    padding: 10px;
    border-radius: 4px;
    overflow-x: auto;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 1em;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
th {
    background-color: #f2f2f2;
    font-weight: bold;
}
"""

HTML_SPECIFIC_STYLES = """
body {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
"""

PDF_SPECIFIC_STYLES = """
@page {
    size: letter;
    margin: 2cm;
}
body {
    font-size: 11pt;
}
h1 { font-size: 18pt; }
h2 { font-size: 16pt; }
h3 { font-size: 14pt; }
h4, h5, h6 { font-size: 12pt; }
pre, code {
    font-family: Courier, monospace;
    font-size: 0.9em;
}
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        {common_styles}
        {specific_styles}
    </style>
</head>
<body>
    {content}
</body>
</html>
"""


class DocumentGenerator:
    @classmethod
    def _generate_html_content(cls, original_content: str) -> str:
        """
        Generate HTML content from the original markdown content.
        """
        try:
            import markdown  # type: ignore
        except ImportError:
            raise ImportError(
                "Failed to import required modules. Please install markdown."
            )

        # Convert markdown to HTML with fenced code and table extensions
        return markdown.markdown(original_content, extensions=["fenced_code", "tables"])

    @classmethod
    def _generate_pdf(cls, html_content: str) -> BytesIO:
        """
        Generate a PDF from the HTML content.
        """
        try:
            from xhtml2pdf import pisa
        except ImportError:
            raise ImportError(
                "Failed to import required modules. Please install xhtml2pdf."
            )

        pdf_html = HTML_TEMPLATE.format(
            common_styles=COMMON_STYLES,
            specific_styles=PDF_SPECIFIC_STYLES,
            content=html_content,
        )

        buffer = BytesIO()
        pdf = pisa.pisaDocument(
            BytesIO(pdf_html.encode("UTF-8")), buffer, encoding="UTF-8"
        )

        if pdf.err:
            logging.error(f"PDF generation failed: {pdf.err}")
            raise ValueError("PDF generation failed")

        buffer.seek(0)
        return buffer

    @classmethod
    def _generate_html(cls, html_content: str) -> str:
        """
        Generate a complete HTML document with the given HTML content.
        """
        return HTML_TEMPLATE.format(
            common_styles=COMMON_STYLES,
            specific_styles=HTML_SPECIFIC_STYLES,
            content=html_content,
        )

    @classmethod
    def generate_document(
        cls, original_content: str, document_type: str, file_name: str
    ) -> str:
        """
        To generate document as PDF or HTML file.
        Parameters:
            original_content: str (markdown style)
            document_type: str (pdf or html) specify the type of the file format based on the use case
            file_name: str (name of the document file) must be a valid file name, no extensions needed
        Returns:
            str (URL to the document file): A file URL ready to serve.
        """
        try:
            document_type = DocumentType(document_type.lower())
        except ValueError:
            raise ValueError(
                f"Invalid document type: {document_type}. Must be 'pdf' or 'html'."
            )
        # Always generate html content first
        html_content = cls._generate_html_content(original_content)

        # Based on the type of document, generate the corresponding file
        if document_type == DocumentType.PDF:
            content = cls._generate_pdf(html_content)
            file_extension = "pdf"
        elif document_type == DocumentType.HTML:
            content = BytesIO(cls._generate_html(html_content).encode("utf-8"))
            file_extension = "html"
        else:
            raise ValueError(f"Unexpected document type: {document_type}")

        file_name = cls._validate_file_name(file_name)
        file_path = os.path.join(OUTPUT_DIR, f"{file_name}.{file_extension}")

        cls._write_to_file(content, file_path)

        return f"{os.getenv('FILESERVER_URL_PREFIX')}/{file_path}"

    @staticmethod
    def _write_to_file(content: BytesIO, file_path: str) -> None:
        """
        Write the content to a file.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file:
                file.write(content.getvalue())
        except Exception as e:
            raise

    @staticmethod
    def _validate_file_name(file_name: str) -> str:
        """
        Validate the file name.
        """
        # Don't allow directory traversal
        if os.path.isabs(file_name):
            raise ValueError("File name is not allowed.")
        # Don't allow special characters
        if re.match(r"^[a-zA-Z0-9_.-]+$", file_name):
            return file_name
        else:
            raise ValueError("File name is not allowed to contain special characters.")

    @classmethod
    def _validate_packages(cls) -> None:
        try:
            import markdown  # noqa: F401
            import xhtml2pdf  # noqa: F401
        except ImportError:
            raise ImportError(
                "Failed to import required modules. Please install markdown and xhtml2pdf "
                "using `pip install markdown xhtml2pdf`"
            )

    def to_tool(self) -> FunctionTool:
        self._validate_packages()
        return FunctionTool.from_defaults(self.generate_document)
