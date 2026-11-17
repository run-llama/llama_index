import io
import os
import pypdf
import pytest
import tempfile
from fpdf import FPDF
from llama_index.readers.file import PDFReader
from pathlib import Path
from typing import Dict, List


class _RecordingFileSystem:
    """A minimal non local filesystem that records the path passed to open."""

    def __init__(self, pdf_bytes: bytes) -> None:
        self.pdf_bytes = pdf_bytes
        self.opened_paths: List[str] = []

    def open(self, path: str, mode: str = "rb"):
        self.opened_paths.append(path)
        return io.BytesIO(self.pdf_bytes)


@pytest.fixture()
def multi_page_pdf() -> FPDF:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, text="Page 1 Content", align="C")
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, text="Page 2 Content", align="C")
    return pdf


@pytest.fixture()
def extra_info() -> Dict[str, str]:
    return {"ABC": "abc", "DEF": "def"}


def test_pdfreader_loads_data_into_full_document(multi_page_pdf: FPDF) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".pdf"
    ) as temp_file:
        multi_page_pdf.output(temp_file.name)
        temp_file_path = Path(temp_file.name)

    reader = PDFReader(return_full_document=True)
    docs = reader.load_data(temp_file_path)

    assert len(docs) == 1
    assert docs[0].text == "\n".join(
        f"Page {page + 1} Content" for page in range(multi_page_pdf.pages_count)
    )

    os.remove(temp_file.name)


def test_pdfreader_loads_data_into_multiple_documents(multi_page_pdf: FPDF) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".pdf"
    ) as temp_file:
        multi_page_pdf.output(temp_file.name)
        temp_file_path = Path(temp_file.name)

    reader = PDFReader(return_full_document=False)
    docs = reader.load_data(temp_file_path)

    assert len(docs) == multi_page_pdf.pages_count
    for page in range(multi_page_pdf.pages_count):
        assert docs[page].text == f"Page {page + 1} Content"

    os.remove(temp_file.name)


def test_pdfreader_loads_metadata_into_full_document(
    multi_page_pdf: FPDF, extra_info: Dict[str, str]
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".pdf"
    ) as temp_file:
        multi_page_pdf.output(temp_file.name)
        temp_file_path = Path(temp_file.name)

    expected_metadata = {"file_name": temp_file_path.name}
    expected_metadata.update(extra_info)

    reader = PDFReader(return_full_document=True)
    docs = reader.load_data(temp_file_path, extra_info)

    assert len(docs) == 1
    assert docs[0].metadata == expected_metadata

    os.remove(temp_file.name)


def test_pdfreader_loads_metadata_into_multiple_documents(
    multi_page_pdf: FPDF, extra_info: Dict[str, str]
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".pdf"
    ) as temp_file:
        multi_page_pdf.output(temp_file.name)
        temp_file_path = Path(temp_file.name)

    expected_metadata = {"file_name": temp_file_path.name}
    expected_metadata.update(extra_info)

    reader = PDFReader(return_full_document=False)
    docs = reader.load_data(temp_file_path, extra_info)
    pypdf_pdf = pypdf.PdfReader(temp_file_path)

    assert len(docs) == multi_page_pdf.pages_count
    for page in range(multi_page_pdf.pages_count):
        expected_metadata["page_label"] = pypdf_pdf.page_labels[page]
        assert docs[page].metadata == expected_metadata

    os.remove(temp_file.name)


def _pdf_bytes(multi_page_pdf: FPDF) -> bytes:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".pdf"
    ) as temp_file:
        multi_page_pdf.output(temp_file.name)

    with open(temp_file.name, "rb") as f:
        pdf_bytes = f.read()

    os.remove(temp_file.name)
    return pdf_bytes


def test_pdfreader_preserves_remote_uri_for_fs_open(
    multi_page_pdf: FPDF,
) -> None:
    fs = _RecordingFileSystem(_pdf_bytes(multi_page_pdf))

    reader = PDFReader(return_full_document=True)
    docs = reader.load_data("s3://bucket/folder/example.pdf", fs=fs)

    assert fs.opened_paths == ["s3://bucket/folder/example.pdf"]
    assert len(docs) == 1
    assert docs[0].metadata["file_name"] == "example.pdf"
    assert "Page 1 Content" in docs[0].text


def test_pdfreader_preserves_remote_uri_per_page(
    multi_page_pdf: FPDF,
) -> None:
    fs = _RecordingFileSystem(_pdf_bytes(multi_page_pdf))

    reader = PDFReader(return_full_document=False)
    docs = reader.load_data("https://example.com/folder/example.pdf", fs=fs)

    assert fs.opened_paths == ["https://example.com/folder/example.pdf"]
    assert len(docs) == multi_page_pdf.pages_count
    assert docs[0].metadata["file_name"] == "example.pdf"
    assert docs[0].text == "Page 1 Content"
