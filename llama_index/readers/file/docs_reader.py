"""Docs parser.

Contains parsers for docx, pdf files.

"""
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class PDFMinerReader(BaseReader):
    """PDF parser based on pdfminer.six."""

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        try:
            from io import StringIO

            from pdfminer.converter import TextConverter
            from pdfminer.layout import LAParams
            from pdfminer.pdfinterp import (PDFPageInterpreter,
                                            PDFResourceManager)
            from pdfminer.pdfpage import PDFPage as PDF_Page

            def _extract_text_from_page(page):
                resource_manager = PDFResourceManager()
                output_string = StringIO()
                codec = 'utf-8'
                laparams = LAParams()
                device = TextConverter(resource_manager, output_string, codec=codec, laparams=laparams)
                interpreter = PDFPageInterpreter(resource_manager, device)
                interpreter.process_page(page)
                text = output_string.getvalue()
                device.close()
                output_string.close()
                return text
            
        except ImportError:
            raise ImportError(
                "pdfminer.six is required to read PDF files: `pip install pdfminer.six`"
            )
        with open(file, 'rb') as fp:
            reader = PDF_Page.get_pages(fp)

            # Iterate over every page
            docs = []
            for i, page in enumerate(reader):
                # Extract the text from the page
                page_text = _extract_text_from_page(page)

                metadata = {"page_label": i, "file_name": file.name}
                if extra_info is not None:
                    metadata.update(extra_info)

                docs.append(Document(page_text, extra_info=metadata))
            return docs


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
        metadata = {"file_name": file.name}
        if extra_info is not None:
            metadata.update(extra_info)

        return [Document(text, extra_info=extra_info)]
