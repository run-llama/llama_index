import io
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import pdfplumber
from paddleocr import (
    AsyncPaddleOCRClient,
    Model,
    OCROptions,
    PPStructureV3Options,
    PaddleOCR,
    PaddleOCRClient,
    PaddleOCRVLOptions,
)
from PIL import Image

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


def _get_token(token: Optional[str]) -> Optional[str]:
    return token or os.environ.get("PADDLEOCR_ACCESS_TOKEN")


# Models that use parse_document(); all others use ocr()
_PARSE_DOCUMENT_MODELS = {
    "PP-StructureV3",
    "PaddleOCR-VL",
    "PaddleOCR-VL-1.5",
    "PaddleOCR-VL-1.6",
}


class PDFPaddleOCRReader(BaseReader):
    def __init__(self, use_angle_cls: bool = True, lang: str = "en"):
        """Initialize PaddleOCR with given parameters"""
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def extract_text_from_image(self, image_data):
        """
        Extract text from image data using PaddleOCR
        """
        try:
            # Convert image data to PIL Image
            image = Image.open(io.BytesIO(image_data))

            # Save temporary image file for PaddleOCR
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                image.save(temp_file.name)
                temp_file_path = temp_file.name

            # Use PaddleOCR to recognize text in the image
            result = self.ocr.predict(temp_file_path)

            # Clean up temporary file
            Path(temp_file_path).unlink()

            # Extract text from recognition results
            extracted_text = ""
            for line in result:
                for text in line["rec_texts"]:
                    extracted_text += text + " "

            return extracted_text.strip()

        except Exception as e:
            logging.error(f"Error in image OCR recognition: {e!s}")
            return ""

    def is_text_meaningful(self, text):
        """
        Check if the extracted text is meaningful
        """
        if not text or len(text.strip()) < 5:
            return False

        # Filter out cases that are likely just page numbers
        if re.match(r"^\d{1,3}$", text.strip()):
            return False

        # Filter out cases that are likely just headers or footers
        common_footers = ["page", "of", "total", "copyright", "all rights reserved"]
        if any(footer in text.lower() for footer in common_footers):
            return len(text.strip()) > 10

        return True

    def extract_page_elements(self, pdf_path, page_num):
        """
        Extract all elements (text and images) from a PDF page, maintaining original order
        """
        elements = []

        try:
            # Use pdfplumber to extract text and position information
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]

                    # Extract text and their positions
                    words = page.extract_words(keep_blank_chars=True)
                    for word in words:
                        elements.append(("text", word["text"], word["top"]))

            # Use PyMuPDF to extract images and their positions
            doc = fitz.open(pdf_path)
            pdf_page = doc.load_page(page_num)

            # Get all images in the page
            image_list = pdf_page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                # Extract image
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Get image position
                image_rects = pdf_page.get_image_rects(xref)
                if image_rects:
                    position = image_rects[0].y0
                else:
                    position = 0

                elements.append(("image", image_bytes, position))

            doc.close()

            # Sort elements by position (top to bottom)
            elements.sort(key=lambda x: x[2])

        except Exception as e:
            logging.error(f"Error occurred while extracting page elements: {e!s}")

        return elements

    def load_data(
        self, file_path: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Load data from PDF using PaddleOCR for image content"""
        documents = []
        file_path = Path(file_path)

        try:
            # Use PyMuPDF to get the total number of pages
            doc = fitz.open(file_path)
            total_pages = len(doc)
            doc.close()

            # Process each page
            for page_num in range(total_pages):
                logging.info(f"Processing page {page_num + 1}/{total_pages}...")

                # Extract all elements from the page (sorted by position)
                elements = self.extract_page_elements(file_path, page_num)

                page_text = ""
                for element_type, content, position in elements:
                    if element_type == "text":
                        # Directly add text
                        if self.is_text_meaningful(content):
                            page_text += f"[Text Content]: {content} "
                    elif element_type == "image":
                        # Perform OCR on the image
                        ocr_text = self.extract_text_from_image(content)
                        if ocr_text and self.is_text_meaningful(ocr_text):
                            page_text += f"[Image Content]: {ocr_text} "

                # Create Document object and add page number as metadata
                if page_text.strip():
                    metadata = {"page": page_num + 1, "source": str(file_path)}
                    if extra_info:
                        metadata.update(extra_info)

                    document = Document(text=page_text.strip(), metadata=metadata)
                    documents.append(document)

        except Exception as e:
            logging.error(f"Error occurred while reading PDF: {e!s}")
            # Return a Document containing error information
            error_doc = Document(
                text=f"Error occurred while reading PDF: {e!s}",
                metadata={"source": str(file_path), "error": True},
            )
            return [error_doc]

        return documents


def _get_token(token: Optional[str]) -> Optional[str]:
    return token or os.environ.get("PADDLEOCR_ACCESS_TOKEN")


class PaddleOCRAPIReader(BaseReader):
    """
    Reader using PaddleOCR official SDK, supports images and PDF files.

    use_parse=False (default): calls ocr(), returns plain text.
    use_parse=True: calls parse_document() if the model supports it, otherwise falls back to ocr().
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = "https://paddleocr.aistudio-app.com",
        model: Optional[str] = None,
        use_parse: bool = False,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
    ):
        super().__init__()
        self._token = token
        self._base_url = base_url
        valid = {m.value for m in Model}
        if model is not None:
            m = model.value if isinstance(model, Model) else model
            if m not in valid:
                raise ValueError(f"Invalid model {m!r}. Valid models: {sorted(valid)}")
            self._model = m
        else:
            self._model = "PP-OCRv6"
        self._use_parse = use_parse
        self._use_doc_orientation_classify = use_doc_orientation_classify
        self._use_doc_unwarping = use_doc_unwarping

    def _get_token(self) -> Optional[str]:
        return _get_token(self._token)

    def _is_parse_model(self) -> bool:
        return self._use_parse and self._model in _PARSE_DOCUMENT_MODELS

    def _build_options(self):
        kwargs = {
            "use_doc_orientation_classify": self._use_doc_orientation_classify,
            "use_doc_unwarping": self._use_doc_unwarping,
        }
        if self._is_parse_model():
            if self._model == "PP-StructureV3":
                return PPStructureV3Options(**kwargs)
            return PaddleOCRVLOptions(**kwargs)
        return OCROptions(**kwargs)

    def _pages_to_documents(
        self, result, file_path: Path, extra_info: dict | None
    ) -> list[Document]:
        docs = []
        for i, page in enumerate(getattr(result, "pages", None) or []):
            if self._is_parse_model():
                text = page.markdown_text or ""
            else:
                texts = (page.pruned_result or {}).get("rec_texts", [])
                text = " ".join(t for t in texts if t and t.strip())
            if text.strip():
                metadata: dict = {"page": i + 1, "source": str(file_path)}
                if extra_info:
                    metadata.update(extra_info)
                docs.append(Document(text=text.strip(), metadata=metadata))
        return docs

    def load_data(
        self, file_path: Path, extra_info: dict | None = None
    ) -> list[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with PaddleOCRClient(
            token=self._get_token(), base_url=self._base_url
        ) as client:
            if self._is_parse_model():
                result = client.parse_document(
                    file_path=str(file_path),
                    model=self._model,
                    options=self._build_options(),
                )
            else:
                result = client.ocr(
                    file_path=str(file_path),
                    model=self._model,
                    options=self._build_options(),
                )
        return self._pages_to_documents(result, file_path, extra_info)

    async def aload_data(
        self, file_path: Path, extra_info: dict | None = None
    ) -> list[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        async with AsyncPaddleOCRClient(
            token=self._get_token(), base_url=self._base_url
        ) as client:
            if self._is_parse_model():
                result = await client.parse_document(
                    file_path=str(file_path),
                    model=self._model,
                    options=self._build_options(),
                )
            else:
                result = await client.ocr(
                    file_path=str(file_path),
                    model=self._model,
                    options=self._build_options(),
                )
        return self._pages_to_documents(result, file_path, extra_info)
