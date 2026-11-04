import builtins

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.semanticscholar import SemanticScholarReader


def test_class():
    names_of_base_classes = [b.__name__ for b in SemanticScholarReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_get_full_text_docs_closes_pdf_file_handle(tmp_path, monkeypatch):
    from PyPDF2 import PdfWriter

    pdf_path = tmp_path / "paper.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    reader = SemanticScholarReader.__new__(SemanticScholarReader)
    reader.base_dir = str(tmp_path)
    monkeypatch.setattr(
        reader, "_download_pdf", lambda paper_id, url, persist_dir: str(pdf_path)
    )

    doc = Document(
        text="",
        extra_info={
            "openAccessPdf": "http://example.com/paper.pdf",
            "externalIds": {},
            "paperId": "paper1",
        },
    )

    opened_files = []
    real_open = builtins.open

    def tracking_open(*args, **kwargs):
        f = real_open(*args, **kwargs)
        opened_files.append(f)
        return f

    monkeypatch.setattr(builtins, "open", tracking_open)
    result = reader._get_full_text_docs([doc])

    assert len(result) == 1
    read_handles = [f for f in opened_files if "r" in f.mode]
    assert read_handles
    assert all(f.closed for f in read_handles)
