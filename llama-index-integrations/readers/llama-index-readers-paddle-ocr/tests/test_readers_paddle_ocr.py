import asyncio
import pytest
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.paddle_ocr import PDFPaddleOCRReader, PaddleOCRAPIReader


class TestPDFPaddleOcrReader(unittest.TestCase):
    def setUp(self):
        self.reader = PDFPaddleOCRReader(lang="en")

    def test_class(self):
        assert issubclass(PDFPaddleOCRReader, BaseReader)

    def test_is_text_meaningful_empty(self):
        assert not self.reader.is_text_meaningful("")
        assert not self.reader.is_text_meaningful("   ")
        assert not self.reader.is_text_meaningful(None)
        assert not self.reader.is_text_meaningful("abcd")

    def test_is_text_meaningful_page_number(self):
        assert not self.reader.is_text_meaningful("100")

    def test_is_text_meaningful_footer(self):
        assert not self.reader.is_text_meaningful("page 1")
        assert self.reader.is_text_meaningful("copyright 2023 by some company")

    def test_is_text_meaningful_normal(self):
        assert self.reader.is_text_meaningful("This is a meaningful sentence.")

    @patch("PIL.Image.open")
    @patch("tempfile.NamedTemporaryFile")
    @patch("pathlib.Path.unlink")
    def test_extract_text_from_image_success(
        self, mock_unlink, mock_tempfile, mock_image_open
    ):
        mock_image = Mock()
        mock_image_open.return_value = mock_image
        mock_temp = Mock()
        mock_temp.name = "/tmp/temp.png"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp

        mock_ocr_result = [{"rec_texts": ["Hello", "World"]}]
        self.reader.ocr.predict = Mock(return_value=mock_ocr_result)

        result = self.reader.extract_text_from_image(b"fake_image_data")

        assert result == "Hello World"
        mock_image.save.assert_called_once_with("/tmp/temp.png")
        self.reader.ocr.predict.assert_called_once_with("/tmp/temp.png")

    @patch("PIL.Image.open")
    def test_extract_text_from_image_failure(self, mock_image_open):
        mock_image_open.side_effect = Exception("Image open failed")
        result = self.reader.extract_text_from_image(b"fake_image_data")
        assert result == ""

    @patch("pdfplumber.open")
    @patch("fitz.open")
    def test_extract_page_elements_success(self, mock_fitz_open, mock_pdfplumber_open):
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_words.return_value = [
            {"text": "Hello", "top": 100},
            {"text": "World", "top": 120},
        ]
        mock_pdf.pages = [mock_page]
        mock_pdfplumber_open.return_value.__enter__ = Mock(return_value=mock_pdf)
        mock_pdfplumber_open.return_value.__exit__ = Mock(return_value=None)

        mock_doc = Mock()
        mock_pdf_page = Mock()
        mock_pdf_page.get_images.return_value = [(1,)]
        mock_pdf_page.get_image_rects.return_value = [Mock(y0=150)]
        mock_doc.load_page.return_value = mock_pdf_page
        mock_doc.extract_image.return_value = {"image": b"fake_image_data"}
        mock_fitz_open.return_value = mock_doc

        result = self.reader.extract_page_elements("/fake/path.pdf", 0)

        assert len(result) == 3
        assert result[0] == ("text", "Hello", 100)
        assert result[1] == ("text", "World", 120)
        assert result[2][0] == "image"
        assert result[2][1] == b"fake_image_data"

    @patch("pdfplumber.open")
    @patch("fitz.open")
    def test_extract_page_elements_exception(self, mock_fitz_open, mock_pdfplumber_open):
        mock_pdfplumber_open.side_effect = Exception("PDF open failed")
        result = self.reader.extract_page_elements("/fake/path.pdf", 0)
        assert result == []

    @patch.object(PDFPaddleOCRReader, "extract_page_elements")
    @patch("fitz.open")
    def test_load_data_success(self, mock_fitz_open, mock_extract_page_elements):
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.close = Mock()
        mock_fitz_open.return_value = mock_doc
        mock_extract_page_elements.return_value = [
            ("text", "Hello World", 100),
            ("image", b"fake_image_data", 150),
        ]
        self.reader.extract_text_from_image = Mock(return_value="Extracted text")
        self.reader.is_text_meaningful = Mock(side_effect=lambda x: len(x) > 3)

        result = self.reader.load_data("/fake/path.pdf")

        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert "Hello World" in result[0].text
        assert "Extracted text" in result[0].text
        assert result[0].metadata["page"] == 1

    @patch("fitz.open")
    def test_load_data_exception(self, mock_fitz_open):
        mock_fitz_open.side_effect = Exception("PDF open failed")
        result = self.reader.load_data("/fake/path.pdf")
        assert len(result) == 1
        assert "Error occurred while reading PDF" in result[0].text
        assert result[0].metadata["error"] is True

    @patch.object(PDFPaddleOCRReader, "extract_page_elements")
    @patch("fitz.open")
    def test_load_data_with_extra_info(self, mock_fitz_open, mock_extract_page_elements):
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.close = Mock()
        mock_fitz_open.return_value = mock_doc
        self.reader.extract_page_elements = Mock(
            return_value=[("text", "Hello World", 100)]
        )
        self.reader.is_text_meaningful = Mock(return_value=True)

        result = self.reader.load_data("/fake/path.pdf", extra_info={"author": "Test"})

        assert result[0].metadata["author"] == "Test"
        assert result[0].metadata["page"] == 1


class TestPaddleOCRAPIReader(unittest.TestCase):
    def _make_ocr_result(self, texts_per_page):
        result = Mock()
        pages = []
        for texts in texts_per_page:
            page = Mock()
            page.pruned_result = {"rec_texts": texts}
            page.markdown_text = None
            pages.append(page)
        result.pages = pages
        return result

    def _make_parse_result(self, markdowns_per_page):
        result = Mock()
        pages = []
        for md in markdowns_per_page:
            page = Mock()
            page.markdown_text = md
            page.pruned_result = {}
            pages.append(page)
        result.pages = pages
        return result

    def test_class(self):
        assert issubclass(PaddleOCRAPIReader, BaseReader)

    def test_init_defaults(self):
        reader = PaddleOCRAPIReader()
        assert reader._token is None
        assert reader._base_url == "https://paddleocr.aistudio-app.com"
        assert reader._model == "PP-StructureV3"

    def test_init_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Invalid model"):
            PaddleOCRAPIReader(model="NotAModel")

    def test_init_string_model(self):
        assert PaddleOCRAPIReader(model="PP-OCRv5")._model == "PP-OCRv5"

    def test_init_enum_model(self):
        from paddleocr import Model
        assert PaddleOCRAPIReader(model=Model.PP_OCRV6)._model == "PP-OCRv6"

    def test_is_parse_model_true(self):
        assert PaddleOCRAPIReader(model="PP-StructureV3")._is_parse_model()
        assert PaddleOCRAPIReader(model="PaddleOCR-VL-1.6")._is_parse_model()

    def test_is_parse_model_false(self):
        assert not PaddleOCRAPIReader(model="PP-OCRv6")._is_parse_model()
        assert not PaddleOCRAPIReader(model="PP-OCRv5")._is_parse_model()

    def test_build_options_structure_v3(self):
        from paddleocr import PPStructureV3Options
        opts = PaddleOCRAPIReader(model="PP-StructureV3")._build_options()
        assert isinstance(opts, PPStructureV3Options)
        assert opts.use_doc_orientation_classify is True
        assert opts.use_doc_unwarping is True

    def test_build_options_vl_model(self):
        from paddleocr import PaddleOCRVLOptions
        opts = PaddleOCRAPIReader(model="PaddleOCR-VL-1.6")._build_options()
        assert isinstance(opts, PaddleOCRVLOptions)
        assert opts.use_doc_orientation_classify is True
        assert opts.use_doc_unwarping is True

    def test_build_options_ocr_model(self):
        from paddleocr import OCROptions
        opts = PaddleOCRAPIReader(model="PP-OCRv6")._build_options()
        assert isinstance(opts, OCROptions)
        assert opts.use_doc_orientation_classify is True
        assert opts.use_doc_unwarping is True

    def test_get_token_from_init(self):
        assert PaddleOCRAPIReader(token="mytoken")._get_token() == "mytoken"

    @patch.dict("os.environ", {"PADDLEOCR_ACCESS_TOKEN": "envtoken"})
    def test_get_token_from_env(self):
        assert PaddleOCRAPIReader()._get_token() == "envtoken"

    @patch("llama_index.readers.paddle_ocr.base.Path.exists", return_value=True)
    @patch("llama_index.readers.paddle_ocr.base.PaddleOCRClient")
    def test_load_data_parse_model_returns_markdown(self, mock_client_cls, mock_exists):
        mock_client = Mock()
        mock_client.parse_document.return_value = self._make_parse_result(
            ["# Title\n\nSome text"]
        )
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_cls.return_value = mock_client

        docs = PaddleOCRAPIReader(token="tok", model="PP-StructureV3").load_data(
            Path("doc.pdf")
        )

        _, kwargs = mock_client.parse_document.call_args
        assert kwargs.get("options") is not None
        assert kwargs["options"].use_doc_orientation_classify is True
        assert kwargs["options"].use_doc_unwarping is True
        mock_client.ocr.assert_not_called()
        assert len(docs) == 1
        assert docs[0].text == "# Title\n\nSome text"
        assert docs[0].metadata["page"] == 1

    @patch("llama_index.readers.paddle_ocr.base.Path.exists", return_value=True)
    @patch("llama_index.readers.paddle_ocr.base.PaddleOCRClient")
    def test_load_data_ocr_model_returns_plain_text(self, mock_client_cls, mock_exists):
        mock_client = Mock()
        mock_client.ocr.return_value = self._make_ocr_result([["Hello", "World"]])
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_cls.return_value = mock_client

        docs = PaddleOCRAPIReader(token="tok", model="PP-OCRv6").load_data(
            Path("scan.jpg")
        )

        mock_client.ocr.assert_called_once()
        mock_client.parse_document.assert_not_called()
        _, kwargs = mock_client.ocr.call_args
        assert kwargs.get("options") is not None
        assert kwargs["options"].use_doc_orientation_classify is True
        assert kwargs["options"].use_doc_unwarping is True
        assert docs[0].text == "Hello World"

    def test_load_data_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PaddleOCRAPIReader(token="tok").load_data(Path("/nonexistent/file.pdf"))

    @patch("llama_index.readers.paddle_ocr.base.Path.exists", return_value=True)
    @patch("llama_index.readers.paddle_ocr.base.PaddleOCRClient")
    def test_load_data_multipage(self, mock_client_cls, mock_exists):
        mock_client = Mock()
        mock_client.parse_document.return_value = self._make_parse_result(
            ["# Page 1", "## Page 2"]
        )
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_cls.return_value = mock_client

        docs = PaddleOCRAPIReader(token="tok").load_data(Path("doc.pdf"))

        assert len(docs) == 2
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2

    @patch("llama_index.readers.paddle_ocr.base.Path.exists", return_value=True)
    @patch("llama_index.readers.paddle_ocr.base.PaddleOCRClient")
    def test_load_data_empty_page_skipped(self, mock_client_cls, mock_exists):
        mock_client = Mock()
        mock_client.parse_document.return_value = self._make_parse_result(["", "  "])
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_cls.return_value = mock_client

        docs = PaddleOCRAPIReader(token="tok").load_data(Path("doc.pdf"))
        assert len(docs) == 0

    @patch("llama_index.readers.paddle_ocr.base.Path.exists", return_value=True)
    @patch("llama_index.readers.paddle_ocr.base.PaddleOCRClient")
    def test_load_data_extra_info(self, mock_client_cls, mock_exists):
        mock_client = Mock()
        mock_client.parse_document.return_value = self._make_parse_result(["# Title"])
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_cls.return_value = mock_client

        docs = PaddleOCRAPIReader(token="tok").load_data(
            Path("scan.jpg"), extra_info={"author": "Test"}
        )
        assert docs[0].metadata["author"] == "Test"
        assert docs[0].metadata["page"] == 1

    def test_aload_data_is_coroutine(self):
        assert asyncio.iscoroutinefunction(PaddleOCRAPIReader.aload_data)

    @patch("llama_index.readers.paddle_ocr.base.Path.exists", return_value=True)
    @patch("llama_index.readers.paddle_ocr.base.AsyncPaddleOCRClient")
    def test_aload_data_parse_model(self, mock_async_client_cls, mock_exists):
        mock_client = AsyncMock()
        mock_client.parse_document = AsyncMock(
            return_value=self._make_parse_result(["# Async Title"])
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_async_client_cls.return_value = mock_client

        docs = asyncio.get_event_loop().run_until_complete(
            PaddleOCRAPIReader(token="tok").aload_data(Path("scan.jpg"))
        )

        assert len(docs) == 1
        assert docs[0].text == "# Async Title"

    @patch("llama_index.readers.paddle_ocr.base.Path.exists", return_value=True)
    @patch("llama_index.readers.paddle_ocr.base.AsyncPaddleOCRClient")
    def test_aload_data_ocr_model(self, mock_async_client_cls, mock_exists):
        mock_client = AsyncMock()
        mock_client.ocr = AsyncMock(
            return_value=self._make_ocr_result([["Async", "OCR"]])
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_async_client_cls.return_value = mock_client

        docs = asyncio.get_event_loop().run_until_complete(
            PaddleOCRAPIReader(token="tok", model="PP-OCRv6").aload_data(Path("scan.jpg"))
        )

        mock_client.ocr.assert_called_once()
        mock_client.parse_document.assert_not_called()
        assert len(docs) == 1
        assert docs[0].text == "Async OCR"

    def test_aload_data_file_not_found(self):
        async def _run():
            await PaddleOCRAPIReader(token="tok").aload_data(Path("/nonexistent/file.pdf"))

        with pytest.raises(FileNotFoundError):
            asyncio.get_event_loop().run_until_complete(_run())

    def test_build_options_custom_flags(self):
        from paddleocr import PPStructureV3Options
        opts = PaddleOCRAPIReader(
            model="PP-StructureV3",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )._build_options()
        assert isinstance(opts, PPStructureV3Options)
        assert opts.use_doc_orientation_classify is False
        assert opts.use_doc_unwarping is False

    @patch("llama_index.readers.paddle_ocr.base.Path.exists", return_value=True)
    @patch("llama_index.readers.paddle_ocr.base.PaddleOCRClient")
    def test_pages_to_documents_no_pages(self, mock_client_cls, mock_exists):
        mock_client = Mock()
        result = Mock()
        result.pages = None
        mock_client.parse_document.return_value = result
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_cls.return_value = mock_client

        docs = PaddleOCRAPIReader(token="tok").load_data(Path("doc.pdf"))
        assert docs == []
