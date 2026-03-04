import unittest
from unittest.mock import Mock, patch, ANY
import io
from pathlib import Path

from llama_index.readers.paddle_ocr import PDFPaddleOCRReader
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader


class TestPDFPaddleOcrReader(unittest.TestCase):
    """Test suite for PDFPaddleOCRReader class"""

    def setUp(self):
        """Set up test fixtures"""
        self.reader = PDFPaddleOCRReader(lang="en")

    def test_class(self):
        names_of_base_classes = [b.__name__ for b in PDFPaddleOCRReader.__mro__]
        assert BaseReader.__name__ in names_of_base_classes

    @patch("PIL.Image.open")
    @patch("tempfile.NamedTemporaryFile")
    @patch("pathlib.Path.unlink")
    def test_extract_text_from_image_success(
        self, mock_unlink, mock_tempfile, mock_image_open
    ):
        """Test successful text extraction from image"""
        # Mock image data
        mock_image = Mock()
        mock_image_open.return_value = mock_image

        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "/tmp/temp.png"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp

        # Mock PaddleOCR result
        mock_ocr_result = [
            {"rec_texts": ["Hello", "World"]},
            {"rec_texts": ["Test", "Text"]},
        ]
        self.reader.ocr.predict = Mock(return_value=mock_ocr_result)

        # Call method
        image_data = b"fake_image_data"
        result = self.reader.extract_text_from_image(image_data)

        # Assertions
        print(f"Result: '{result}'")
        print(f"Expected: 'Hello World Test Text'")
        self.assertEqual(result, "Hello World Test Text")

        # Check if mock_image_open was called once with a BytesIO type argument
        mock_image_open.assert_called_once_with(ANY)
        args, _ = mock_image_open.call_args
        self.assertIsInstance(args[0], io.BytesIO)

        # Check if the BytesIO object contains the correct data
        args[0].seek(0)
        self.assertEqual(args[0].read(), image_data)

        mock_image.save.assert_called_once_with("/tmp/temp.png")
        self.reader.ocr.predict.assert_called_once_with("/tmp/temp.png")
        mock_unlink.assert_called_once()

    @patch("PIL.Image.open")
    def test_extract_text_from_image_failure(self, mock_image_open):
        """Test text extraction from image when an exception occurs"""
        # Mock an exception
        mock_image_open.side_effect = Exception("Image open failed")

        # Call method
        image_data = b"fake_image_data"
        result = self.reader.extract_text_from_image(image_data)

        # Assertions
        assert result == ""

    def test_is_text_meaningful_empty_text(self):
        """Test is_text_meaningful with empty text"""
        assert not self.reader.is_text_meaningful("")
        assert not self.reader.is_text_meaningful("   ")
        assert not self.reader.is_text_meaningful(None)

    def test_is_text_meaningful_short_text(self):
        """Test is_text_meaningful with short text"""
        assert not self.reader.is_text_meaningful("a")
        assert not self.reader.is_text_meaningful("ab")
        assert not self.reader.is_text_meaningful("abc")
        assert not self.reader.is_text_meaningful("abcd")

    def test_is_text_meaningful_page_number(self):
        """Test is_text_meaningful with page numbers"""
        assert not self.reader.is_text_meaningful("1")
        assert not self.reader.is_text_meaningful("10")
        assert not self.reader.is_text_meaningful("100")
        assert not self.reader.is_text_meaningful(" 100 ")

    def test_is_text_meaningful_footer_text(self):
        """Test is_text_meaningful with footer text"""
        # Short footer text should be filtered
        assert not self.reader.is_text_meaningful("page 1")
        assert not self.reader.is_text_meaningful("copyright")

        # Longer footer text should be kept
        assert self.reader.is_text_meaningful("copyright 2023 by some company")

    def test_is_text_meaningful_meaningful_text(self):
        """Test is_text_meaningful with meaningful text"""
        assert self.reader.is_text_meaningful("This is a meaningful sentence.")
        assert self.reader.is_text_meaningful("Deep learning models")
        assert self.reader.is_text_meaningful("Variational autoencoder")

    @patch("pdfplumber.open")
    @patch("fitz.open")
    def test_extract_page_elements_success(self, mock_fitz_open, mock_pdfplumber_open):
        """Test successful extraction of page elements"""
        # Mock pdfplumber
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_words.return_value = [
            {"text": "Hello", "top": 100},
            {"text": "World", "top": 120},
        ]
        mock_pdf.pages = [mock_page]
        mock_pdfplumber_open.return_value.__enter__ = Mock(return_value=mock_pdf)
        mock_pdfplumber_open.return_value.__exit__ = Mock(return_value=None)

        # Mock PyMuPDF
        mock_doc = Mock()
        mock_pdf_page = Mock()
        mock_pdf_page.get_images.return_value = [(1,)]
        mock_pdf_page.get_image_rects.return_value = [Mock(y0=150)]
        mock_doc.load_page.return_value = mock_pdf_page
        mock_doc.extract_image.return_value = {"image": b"fake_image_data"}
        mock_fitz_open.return_value = mock_doc

        # Call method
        pdf_path = "/fake/path.pdf"
        result = self.reader.extract_page_elements(pdf_path, 0)

        # Assertions
        assert len(result) == 3  # 2 text elements + 1 image element
        assert result[0] == ("text", "Hello", 100)
        assert result[1] == ("text", "World", 120)
        assert result[2][0] == "image"
        assert result[2][1] == b"fake_image_data"
        assert result[2][2] == 150

        # Verify mocks were called correctly
        mock_pdfplumber_open.assert_called_once_with(pdf_path)
        mock_fitz_open.assert_called_once_with(pdf_path)
        mock_page.extract_words.assert_called_once_with(keep_blank_chars=True)
        mock_pdf_page.get_images.assert_called_once_with(full=True)

    @patch("pdfplumber.open")
    @patch("fitz.open")
    def test_extract_page_elements_exception(
        self, mock_fitz_open, mock_pdfplumber_open
    ):
        """Test extract_page_elements when an exception occurs"""
        # Mock an exception
        mock_pdfplumber_open.side_effect = Exception("PDF open failed")

        # Call method
        pdf_path = "/fake/path.pdf"
        result = self.reader.extract_page_elements(pdf_path, 0)

        # Assertions
        assert result == []

    @patch("pdfplumber.open")
    @patch("fitz.open")
    def test_extract_page_elements_no_images(
        self, mock_fitz_open, mock_pdfplumber_open
    ):
        """Test extract_page_elements when there are no images"""
        # Mock pdfplumber
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_words.return_value = [
            {"text": "Hello", "top": 100},
            {"text": "World", "top": 120},
        ]
        mock_pdf.pages = [mock_page]
        mock_pdfplumber_open.return_value.__enter__ = Mock(return_value=mock_pdf)
        mock_pdfplumber_open.return_value.__exit__ = Mock(return_value=None)

        # Mock PyMuPDF with no images
        mock_doc = Mock()
        mock_pdf_page = Mock()
        mock_pdf_page.get_images.return_value = []  # No images
        mock_doc.load_page.return_value = mock_pdf_page
        mock_fitz_open.return_value = mock_doc

        # Call method
        pdf_path = "/fake/path.pdf"
        result = self.reader.extract_page_elements(pdf_path, 0)

        # Assertions
        assert len(result) == 2  # Only text elements
        assert result[0] == ("text", "Hello", 100)
        assert result[1] == ("text", "World", 120)

    @patch.object(PDFPaddleOCRReader, "extract_page_elements")
    @patch("fitz.open")
    def test_load_data_success(self, mock_fitz_open, mock_extract_page_elements):
        """Test successful loading of data from PDF"""
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)  # 1 page
        mock_fitz_open.return_value = mock_doc

        # Mock extract_page_elements to return meaningful elements
        mock_extract_page_elements.return_value = [
            ("text", "Hello", 100),
            ("image", b"fake_image_data", 150),
        ]

        # Mock extract_text_from_image and is_text_meaningful
        self.reader.extract_text_from_image = Mock(return_value="Extracted text")
        self.reader.is_text_meaningful = Mock(side_effect=lambda x: len(x) > 3)

        # Call method
        pdf_path = "/fake/path.pdf"
        result = self.reader.load_data(pdf_path)

        # Assertions
        self.assertEqual(len(result), 1)  # One document
        self.assertIsInstance(result[0], Document)
        self.assertIn("Hello", result[0].text)
        self.assertIn("Extracted text", result[0].text)
        self.assertEqual(result[0].metadata["page"], 1)

        # Use Path object for path comparison to avoid OS differences
        self.assertEqual(Path(result[0].metadata["source"]), Path(pdf_path))

        # Verify mocks were called correctly - accept Path object
        mock_fitz_open.assert_called_once_with(ANY)
        args, _ = mock_fitz_open.call_args
        self.assertIsInstance(args[0], (str, Path))  # Allow string or Path object

        # Convert to string for comparison
        path_str = str(args[0])
        self.assertTrue(path_str.replace("\\", "/").endswith("/fake/path.pdf"))

        mock_extract_page_elements.assert_called_once_with(ANY, 0)
        args, _ = mock_extract_page_elements.call_args
        self.assertIsInstance(args[0], (str, Path))  # Allow string or Path object

        # Convert to string for comparison
        path_str = str(args[0])
        self.assertTrue(path_str.replace("\\", "/").endswith("/fake/path.pdf"))

        self.reader.extract_text_from_image.assert_called_once_with(b"fake_image_data")

    @patch.object(PDFPaddleOCRReader, "extract_page_elements")
    @patch("fitz.open")
    def test_load_data_no_meaningful_text(
        self, mock_fitz_open, mock_extract_page_elements
    ):
        """Test load_data when no meaningful text is found"""
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)  # 1 page
        mock_fitz_open.return_value = mock_doc

        # Mock extract_page_elements to return only non-meaningful elements
        mock_extract_page_elements.return_value = [
            ("text", "1", 100),  # Page number (not meaningful)
            ("text", "copyright", 150),  # Footer (not meaningful)
        ]

        # Mock is_text_meaningful to return False for all text
        self.reader.is_text_meaningful = Mock(return_value=False)

        # Call method
        pdf_path = "/fake/path.pdf"
        result = self.reader.load_data(pdf_path)

        # Assertions
        assert len(result) == 0  # No documents created

    @patch("fitz.open")
    def test_load_data_exception(self, mock_fitz_open):
        """Test load_data when an exception occurs"""
        # Mock an exception
        mock_fitz_open.side_effect = Exception("PDF open failed")

        # Call method
        pdf_path = "/fake/path.pdf"
        result = self.reader.load_data(pdf_path)

        # Assertions
        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert "Error occurred while reading PDF" in result[0].text
        assert result[0].metadata["error"] is True

    @patch("fitz.open")
    def test_load_data_with_extra_info(self, mock_fitz_open):
        """Test load_data with extra_info parameter"""
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)  # 1 page
        mock_fitz_open.return_value = mock_doc

        # Mock extract_page_elements to return meaningful text
        self.reader.extract_page_elements = Mock(return_value=[("text", "Hello", 100)])
        self.reader.is_text_meaningful = Mock(return_value=True)

        # Call method with extra_info
        pdf_path = "/fake/path.pdf"
        extra_info = {"author": "Test Author", "title": "Test Title"}
        result = self.reader.load_data(pdf_path, extra_info=extra_info)

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata["author"], "Test Author")
        self.assertEqual(result[0].metadata["title"], "Test Title")
        self.assertEqual(result[0].metadata["page"], 1)

        # Use Path object for path comparison to avoid OS differences
        self.assertEqual(Path(result[0].metadata["source"]), Path(pdf_path))

    def test_load_data_invalid_path(self):
        """Test load_data with invalid file path"""
        # Call method with non-existent path
        pdf_path = "/non/existent/path.pdf"
        result = self.reader.load_data(pdf_path)

        # Assertions
        assert len(result) == 1
        assert "Error occurred while reading PDF" in result[0].text
        assert result[0].metadata["error"] is True
