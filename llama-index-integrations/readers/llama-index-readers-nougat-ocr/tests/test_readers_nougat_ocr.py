from pathlib import Path

from llama_index.core.readers.base import BaseReader
from llama_index.readers.nougat_ocr import PDFNougatOCR


def test_class():
    names_of_base_classes = [b.__name__ for b in PDFNougatOCR.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_load_data_reads_utf8_output(tmp_path, monkeypatch):
    """
    Nougat writes UTF-8 Markdown, so the reader must decode its output as
    UTF-8 instead of relying on the system locale.
    """
    reader = PDFNougatOCR()
    monkeypatch.chdir(tmp_path)
    output_folder = tmp_path / "output"
    output_folder.mkdir()
    markdown = "# Titre\n\n$\\alpha > 0$ — naïve\n\n中文段落\n"
    (output_folder / "paper.mmd").write_bytes(markdown.encode("utf-8"))
    monkeypatch.setattr(reader, "nougat_ocr", lambda file_path: "")

    documents = reader.load_data(Path("paper.pdf"))

    assert documents is not None
    assert documents[0].text == markdown
