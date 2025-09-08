import os
from typing import List, Union, Optional
from pathlib import Path
from io import BytesIO

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from .event import FileType

# PDF Reader
class PDFReader(BaseReader):
    def load_data(self, file_path: Union[str, Path], **kwargs) -> List[Document]:
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("Please install pytesseract and pdf2image for PDFReader.")

        text = ""
        images = convert_from_path(str(file_path))
        for i, image in enumerate(images):
            image_text = pytesseract.image_to_string(image)
            text += f"Page {i + 1}:\n{image_text}\n\n"
        return [Document(text=text, metadata={"file_path": str(file_path)})]

# HTML Reader
class HTMLReader(BaseReader):
    def load_data(self, file_path: Union[str, Path], **kwargs) -> List[Document]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Please install beautifulsoup4 for HTMLReader.")

        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return [Document(text=text, metadata={"file_path": str(file_path)})]

# TXT Reader
class TXTReader(BaseReader):
    def load_data(self, file_path: Union[str, Path], **kwargs) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(text=text, metadata={"file_path": str(file_path)})]

# DOCX Reader
class DocxReader(BaseReader):
    def load_data(self, file_path: Union[str, Path], **kwargs) -> List[Document]:
        try:
            import docx2txt
        except ImportError:
            raise ImportError("Please install docx2txt for DocxReader.")

        text = docx2txt.process(str(file_path))
        return [Document(text=text, metadata={"file_path": str(file_path)})]

# PPTX Reader
class PptxReader(BaseReader):
    def load_data(self, file_path: Union[str, Path], **kwargs) -> List[Document]:
        try:
            from pptx import Presentation
        except ImportError:
            raise ImportError("Please install python-pptx for PptxReader.")

        text = ""
        presentation = Presentation(str(file_path))
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + " "
        return [Document(text=text.strip(), metadata={"file_path": str(file_path)})]

# CSV Reader
class CSVReader(BaseReader):
    def load_data(self, file_path: Union[str, Path], **kwargs) -> List[Document]:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Please install pandas for CSVReader.")

        df = pd.read_csv(file_path, low_memory=False)
        text_rows = []
        for _, row in df.iterrows():
            text_rows.append(", ".join(row.astype(str)))
        text = "\n".join(text_rows)
        return [Document(text=text, metadata={"file_path": str(file_path)})]

# XLSX Reader
class ExcelReader(BaseReader):
    def load_data(self, file_path: Union[str, Path], **kwargs) -> List[Document]:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Please install pandas for ExcelReader.")

        sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        text = ""
        for sheet_name, sheet_data in sheets.items():
            text += f"{sheet_name}:\n"
            for _, row in sheet_data.iterrows():
                text += "\t".join(str(value) for value in row) + "\n"
            text += "\n"
        return [Document(text=text.strip(), metadata={"file_path": str(file_path)})]

# IMAGE Reader (OCR)
class ImageReader(BaseReader):
    def load_data(self, file_path: Union[str, Path], **kwargs) -> List[Document]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError("Please install pytesseract and Pillow for ImageReader.")

        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return [Document(text=text, metadata={"file_path": str(file_path)})]

# Usage Example for SharePointReader:
# from .file_parsers import PDFReader, HTMLReader, DocxReader, PptxReader, CSVReader, ExcelReader, ImageReader
# custom_parsers = {
#     FileType.PDF: PDFReader(),
#     FileType.HTML: HTMLReader(),
#     FileType.DOCUMENT: DocxReader(),
#     FileType.PRESENTATION: PptxReader(),
#     FileType.CSV: CSVReader(),
#     FileType.SPREADSHEET: ExcelReader(),
#     FileType.IMAGE: ImageReader(),
# }
# reader = SharePointReader(..., custom_parsers=custom_parsers, custom_folder="/tmp")
