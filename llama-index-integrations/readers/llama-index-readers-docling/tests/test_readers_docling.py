import json
from pathlib import Path
from unittest.mock import MagicMock
from llama_index.readers.docling.base import DoclingReader
from docling_core.types import Document as DLDocument

ROOT_DIR_PATH = Path(__file__).resolve().parent


def test_lazy_load_data_with_md_export(monkeypatch):
    with open(ROOT_DIR_PATH / "data" / "inp_dl_doc.json") as f:
        in_json_str = f.read()
    mock_dl_doc = DLDocument.model_validate_json(in_json_str)
    mock_response = MagicMock()
    mock_response.output = mock_dl_doc

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.convert_single",
        lambda *args, **kwargs: mock_response,
    )

    reader = DoclingReader()
    doc_iter = reader.lazy_load_data(file_path="foo.pdf")
    act_li_docs = list(doc_iter)
    assert len(act_li_docs) == 1

    act_data = {"root": [li_doc.model_dump() for li_doc in act_li_docs]}
    with open(ROOT_DIR_PATH / "data" / "out_li_doc_with_md.json") as f:
        exp_data = json.load(f)
    assert act_data == exp_data


def test_lazy_load_data_with_json_export(monkeypatch):
    with open(ROOT_DIR_PATH / "data" / "inp_dl_doc.json") as f:
        in_json_str = f.read()
    mock_dl_doc = DLDocument.model_validate_json(in_json_str)
    mock_response = MagicMock()
    mock_response.output = mock_dl_doc

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.convert_single",
        lambda *args, **kwargs: mock_response,
    )

    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    doc_iter = reader.lazy_load_data(file_path="foo.pdf")
    act_li_docs = list(doc_iter)
    assert len(act_li_docs) == 1

    act_data = {"root": [li_doc.model_dump() for li_doc in act_li_docs]}
    with open(ROOT_DIR_PATH / "data" / "out_li_doc_with_json.json") as f:
        exp_data = json.load(f)
    assert act_data == exp_data


if __name__ == "__main__":
    test_lazy_load_data_with_md_export()
