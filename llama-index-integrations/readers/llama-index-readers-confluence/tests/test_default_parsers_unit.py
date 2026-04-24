import sys
import types
from unittest.mock import patch

from llama_index.readers.confluence.default_parsers import (
    DefaultCsvParser,
    DefaultDocParser,
    DefaultHtmlParser,
    DefaultImageParser,
    DefaultMsgParser,
    DefaultPdfParser,
    DefaultPptParser,
    DefaultSvgParser,
    DefaultTxtParser,
    DefaultXlsParser,
    _error_text,
)


class _FakeRow:
    def __init__(self, values):
        self._values = values

    def __iter__(self):
        return iter(self._values)

    def astype(self, _):
        return [str(v) for v in self._values]


class _FakeFrame:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, _FakeRow(row)


def _assert_error_contract(text: str, file_type: str, file_name: str, exc_name: str):
    assert f"error processing {file_type}" in text
    assert "exception message:" in text
    assert f"{exc_name}:" in text
    assert f"file name: {file_name}" in text


def test_error_text_contract_uses_basename_and_exception_details():
    exc = ValueError("bad payload")
    text = _error_text("csv", "/tmp/foo/bar/report.csv", exc)

    assert "error processing csv" in text
    assert "exception message: ValueError: bad payload" in text
    assert "file name: report.csv" in text


@patch("builtins.open", side_effect=OSError("permission denied"))
def test_default_txt_parser_returns_unified_error_text(_):
    parser = DefaultTxtParser()
    docs = parser.lazy_load_data("/tmp/private/note.txt")

    assert len(docs) == 1
    _assert_error_contract(docs[0].text, "text", "note.txt", "OSError")


def test_default_html_parser_success(tmp_path):
    html_file = tmp_path / "sample.html"
    html_file.write_text("<h1>Title</h1><p>Body</p>", encoding="utf-8")

    fake_bs4 = types.SimpleNamespace(
        BeautifulSoup=lambda content, _parser: types.SimpleNamespace(
            get_text=lambda separator, strip: "Title Body"
        )
    )

    with patch.dict(sys.modules, {"bs4": fake_bs4}):
        docs = DefaultHtmlParser().lazy_load_data(str(html_file))

    assert docs[0].text == "Title Body"


def test_default_html_parser_error(tmp_path):
    html_file = tmp_path / "broken.html"
    html_file.write_text("<h1>x</h1>", encoding="utf-8")

    class _BrokenSoup:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("parse failed")

    fake_bs4 = types.SimpleNamespace(BeautifulSoup=_BrokenSoup)

    with patch.dict(sys.modules, {"bs4": fake_bs4}):
        docs = DefaultHtmlParser().lazy_load_data(str(html_file))

    _assert_error_contract(docs[0].text, "html", "broken.html", "RuntimeError")


def test_default_pdf_parser_success():
    fake_pdf2image = types.SimpleNamespace(convert_from_path=lambda _path: [object()])
    fake_tesseract = types.SimpleNamespace(image_to_string=lambda _img: "ocr text")

    with patch.dict(
        sys.modules,
        {
            "pdf2image": fake_pdf2image,
            "pytesseract": fake_tesseract,
        },
    ):
        docs = DefaultPdfParser().lazy_load_data("/tmp/test.pdf")

    assert "Page 1:" in docs[0].text
    assert "ocr text" in docs[0].text


def test_default_pdf_parser_error():
    def _raise(_path):
        raise RuntimeError("corrupt pdf")

    fake_pdf2image = types.SimpleNamespace(convert_from_path=_raise)
    fake_tesseract = types.SimpleNamespace(image_to_string=lambda _img: "unused")

    with patch.dict(
        sys.modules,
        {
            "pdf2image": fake_pdf2image,
            "pytesseract": fake_tesseract,
        },
    ):
        docs = DefaultPdfParser().lazy_load_data("/tmp/test.pdf")

    _assert_error_contract(docs[0].text, "pdf", "test.pdf", "RuntimeError")


def test_default_xls_parser_success():
    fake_pandas = types.SimpleNamespace(
        read_excel=lambda *_args, **_kwargs: {
            "Sheet1": _FakeFrame([["Alice", 30], ["Bob", 25]]),
            "Sheet2": _FakeFrame([["Ops", 1]]),
        }
    )

    with patch.dict(sys.modules, {"pandas": fake_pandas}):
        docs = DefaultXlsParser().lazy_load_data("/tmp/data.xlsx")

    assert "Sheet1:" in docs[0].text
    assert "Alice" in docs[0].text
    assert "Sheet2:" in docs[0].text


def test_default_xls_parser_error():
    def _raise(*_args, **_kwargs):
        raise ValueError("invalid xlsx")

    fake_pandas = types.SimpleNamespace(read_excel=_raise)

    with patch.dict(sys.modules, {"pandas": fake_pandas}):
        docs = DefaultXlsParser().lazy_load_data("/tmp/data.xlsx")

    _assert_error_contract(docs[0].text, "spreadsheet", "data.xlsx", "ValueError")


def test_default_csv_parser_success():
    fake_pandas = types.SimpleNamespace(
        read_csv=lambda *_args, **_kwargs: _FakeFrame([["A", 1], ["B", 2]])
    )

    with patch.dict(sys.modules, {"pandas": fake_pandas}):
        docs = DefaultCsvParser().lazy_load_data("/tmp/data.csv")

    assert "A, 1" in docs[0].text
    assert "B, 2" in docs[0].text


def test_default_csv_parser_error():
    def _raise(*_args, **_kwargs):
        raise RuntimeError("bad csv")

    fake_pandas = types.SimpleNamespace(read_csv=_raise)

    with patch.dict(sys.modules, {"pandas": fake_pandas}):
        docs = DefaultCsvParser().lazy_load_data("/tmp/data.csv")

    _assert_error_contract(docs[0].text, "csv", "data.csv", "RuntimeError")


def test_default_ppt_parser_success():
    fake_shape = types.SimpleNamespace(text="Slide text")
    fake_slide = types.SimpleNamespace(shapes=[fake_shape])
    fake_presentation = types.SimpleNamespace(slides=[fake_slide])
    fake_pptx = types.SimpleNamespace(Presentation=lambda _path: fake_presentation)

    with patch.dict(sys.modules, {"pptx": fake_pptx}):
        docs = DefaultPptParser().lazy_load_data("/tmp/slides.pptx")

    assert docs[0].text == "Slide text"


def test_default_ppt_parser_error():
    def _raise(_path):
        raise RuntimeError("broken ppt")

    fake_pptx = types.SimpleNamespace(Presentation=_raise)

    with patch.dict(sys.modules, {"pptx": fake_pptx}):
        docs = DefaultPptParser().lazy_load_data("/tmp/slides.pptx")

    _assert_error_contract(docs[0].text, "presentation", "slides.pptx", "RuntimeError")


def test_default_doc_parser_error():
    fake_docx2txt = types.SimpleNamespace(
        process=lambda _path: (_ for _ in ()).throw(RuntimeError("bad docx"))
    )

    with patch.dict(sys.modules, {"docx2txt": fake_docx2txt}):
        docs = DefaultDocParser().lazy_load_data("/tmp/doc.docx")

    _assert_error_contract(docs[0].text, "document", "doc.docx", "RuntimeError")


def test_default_image_parser_error():
    class _BrokenImageModule:
        @staticmethod
        def open(_path):
            raise OSError("cannot open image")

    fake_pil = types.SimpleNamespace(Image=_BrokenImageModule)
    fake_tesseract = types.SimpleNamespace(image_to_string=lambda _img: "unused")

    with patch.dict(
        sys.modules,
        {
            "PIL": fake_pil,
            "pytesseract": fake_tesseract,
        },
    ):
        docs = DefaultImageParser().lazy_load_data("/tmp/photo.png")

    _assert_error_contract(docs[0].text, "image", "photo.png", "OSError")


def test_default_msg_parser_success():
    class _MsgCtx:
        subject = "Subject"
        sender = "sender@example.com"
        to = "to@example.com"
        cc = "cc@example.com"
        body = "Body"

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    fake_extract_msg = types.SimpleNamespace(Message=lambda _path: _MsgCtx())

    with patch.dict(sys.modules, {"extract_msg": fake_extract_msg}):
        docs = DefaultMsgParser().lazy_load_data("/tmp/mail.msg")

    assert "Subject: Subject" in docs[0].text
    assert "Body" in docs[0].text


def test_default_msg_parser_error():
    class _BrokenMsg:
        def __enter__(self):
            raise RuntimeError("bad msg")

        def __exit__(self, *_exc):
            return False

    fake_extract_msg = types.SimpleNamespace(Message=lambda _path: _BrokenMsg())

    with patch.dict(sys.modules, {"extract_msg": fake_extract_msg}):
        docs = DefaultMsgParser().lazy_load_data("/tmp/mail.msg")

    _assert_error_contract(docs[0].text, "msg", "mail.msg", "RuntimeError")


def test_default_svg_parser_empty_when_drawing_is_none():
    fake_tesseract = types.SimpleNamespace(image_to_string=lambda _img: "unused")

    class _ImageModule:
        @staticmethod
        def open(_data):
            return types.SimpleNamespace(
                __enter__=lambda self: self,
                __exit__=lambda self, *_exc: False,
            )

    fake_pil = types.SimpleNamespace(Image=_ImageModule)
    fake_renderpm = types.SimpleNamespace(drawToFile=lambda *_args, **_kwargs: None)
    fake_svglib = types.SimpleNamespace(svg2rlg=lambda _path: None)

    with patch.dict(
        sys.modules,
        {
            "pytesseract": fake_tesseract,
            "PIL": fake_pil,
            "reportlab.graphics": types.SimpleNamespace(renderPM=fake_renderpm),
            "reportlab.graphics.renderPM": fake_renderpm,
            "svglib.svglib": fake_svglib,
        },
    ):
        docs = DefaultSvgParser().lazy_load_data("/tmp/shape.svg")

    assert docs[0].text == ""


def test_default_svg_parser_error():
    fake_tesseract = types.SimpleNamespace(image_to_string=lambda _img: "unused")

    class _ImageModule:
        @staticmethod
        def open(_data):
            raise RuntimeError("image decode failed")

    fake_pil = types.SimpleNamespace(Image=_ImageModule)
    fake_renderpm = types.SimpleNamespace(drawToFile=lambda *_args, **_kwargs: None)
    fake_svglib = types.SimpleNamespace(svg2rlg=lambda _path: object())

    with patch.dict(
        sys.modules,
        {
            "pytesseract": fake_tesseract,
            "PIL": fake_pil,
            "reportlab.graphics": types.SimpleNamespace(renderPM=fake_renderpm),
            "reportlab.graphics.renderPM": fake_renderpm,
            "svglib.svglib": fake_svglib,
        },
    ):
        docs = DefaultSvgParser().lazy_load_data("/tmp/shape.svg")

    _assert_error_contract(docs[0].text, "svg", "shape.svg", "RuntimeError")
