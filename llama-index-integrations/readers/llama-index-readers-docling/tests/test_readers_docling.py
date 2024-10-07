import json
from unittest.mock import MagicMock
from llama_index.readers.docling.base import DoclingReader
from docling_core.types import Document as DLDocument


def test_lazy_load_data_with_md_export(mocker):
    with open("tests/data/inp_dl_doc.json") as f:
        in_json_str = f.read()
    mock_dl_doc = DLDocument.model_validate_json(in_json_str)
    mock_response = MagicMock()
    mock_response.output = mock_dl_doc
    mocker.patch(
        "docling.document_converter.DocumentConverter.__init__", return_value=None
    )
    mocker.patch(
        "docling.document_converter.DocumentConverter.convert_single",
        return_value=mock_response,
    )

    reader = DoclingReader()
    doc_iter = reader.lazy_load_data(file_path="foo.pdf")
    act_li_docs = list(doc_iter)
    assert len(act_li_docs) == 1

    act_data = {"root": [li_doc.model_dump() for li_doc in act_li_docs]}
    with open("tests/data/out_li_doc_with_md.json") as f:
        exp_data = json.load(f)
    assert act_data == exp_data


def test_lazy_load_data_with_json_export(mocker):
    with open("tests/data/inp_dl_doc.json") as f:
        in_json_str = f.read()
    mock_dl_doc = DLDocument.model_validate_json(in_json_str)
    mock_response = MagicMock()
    mock_response.output = mock_dl_doc
    mocker.patch(
        "docling.document_converter.DocumentConverter.__init__", return_value=None
    )
    mocker.patch(
        "docling.document_converter.DocumentConverter.convert_single",
        return_value=mock_response,
    )

    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    doc_iter = reader.lazy_load_data(file_path="foo.pdf")
    act_li_docs = list(doc_iter)
    assert len(act_li_docs) == 1

    act_data = {"root": [li_doc.model_dump() for li_doc in act_li_docs]}
    with open("tests/data/out_li_doc_with_json.json") as f:
        exp_data = json.load(f)
    assert act_data == exp_data
