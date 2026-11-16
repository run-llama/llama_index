import os
import pypdf
import pytest
import tempfile
from io import BytesIO
from fpdf import FPDF
from fsspec import AbstractFileSystem
from llama_index.readers.file import PDFReader
from pathlib import Path, PurePosixPath
from typing import Dict, Union
from unittest.mock import MagicMock


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


@pytest.mark.parametrize("return_full_document", [False, True])
@pytest.mark.parametrize(
    "file",
    [
        "s3://bucket/folder/example.pdf",
        "https://example.com/folder/example.pdf",
        "bucket/folder/example.pdf",
        PurePosixPath("bucket/folder/example.pdf"),
    ],
)
def test_pdfreader_preserves_remote_paths(
    multi_page_pdf: FPDF,
    extra_info: Dict[str, str],
    file: Union[str, PurePosixPath],
    return_full_document: bool,
) -> None:
    fs = MagicMock(spec=AbstractFileSystem)
    fs.open.return_value = BytesIO(multi_page_pdf.output())
    reader = PDFReader(return_full_document=return_full_document)

    docs = reader.load_data(file, extra_info=extra_info, fs=fs)

    fs.open.assert_called_once_with(str(file), "rb")
    expected_texts = ["Page 1 Content", "Page 2 Content"]
    if return_full_document:
        expected_texts = ["\n".join(expected_texts)]
    assert [doc.text for doc in docs] == expected_texts
    for page, doc in enumerate(docs):
        expected_metadata = {"file_name": "example.pdf", **extra_info}
        if not return_full_document:
            expected_metadata["page_label"] = str(page + 1)
        assert doc.metadata == expected_metadata


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
