"""Docs parser.

Contains parsers for docx, pdf files.

"""
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class PDFReader(BaseReader):
    """PDF parser."""

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 is required to read PDF files: `pip install PyPDF2`"
            )
        text_list = []
        with open(file, "rb") as fp:
            # Create a PDF object
            pdf = PyPDF2.PdfReader(fp)

            # Get the number of pages in the PDF document
            num_pages = len(pdf.pages)

            # Iterate over every page
            for page in range(num_pages):
                # Extract the text from the page
                page_text = pdf.pages[page].extract_text()
                text_list.append(page_text)
        text = "\n".join(text_list)
        return [Document(text, extra_info=extra_info)]


class DocxReader(BaseReader):
    """Docx parser."""

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "docx2txt is required to read Microsoft Word files: "
                "`pip install docx2txt`"
            )

        text = docx2txt.process(file)

        return [Document(text, extra_info=extra_info)]
