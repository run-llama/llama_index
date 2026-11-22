from llama_index.core.readers.base import BaseReader
from llama_index.readers.semanticscholar import SemanticScholarReader


def test_class():
    names_of_base_classes = [b.__name__ for b in SemanticScholarReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_get_full_text_docs_reads_pdf(tmp_path, monkeypatch):
    from llama_index.core.schema import Document
    from pypdf import PdfWriter

    pdf_path = tmp_path / "downloaded.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    reader = SemanticScholarReader(base_dir=str(tmp_path / "pdfs"))
    monkeypatch.setattr(
        reader, "_download_pdf", lambda paper_id, url, persist_dir: str(pdf_path)
    )
    metadata = {
        "openAccessPdf": "https://example.com/paper.pdf",
        "externalIds": {},
        "paperId": "abc123",
    }

    docs = reader._get_full_text_docs([Document(text="", extra_info=metadata)])

    assert len(docs) == 1
    assert docs[0].extra_info == metadata
