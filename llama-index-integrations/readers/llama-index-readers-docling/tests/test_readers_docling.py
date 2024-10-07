from unittest.mock import MagicMock
from llama_index.readers.docling.base import DoclingReader
from docling_core.types import Document as DLDocument

in_json_str = """{
  "name": "foo",
  "description": {
    "logs": []
  },
  "main_text": [
    {
      "text": "Test subtitle",
      "type": "subtitle-level-1",
      "name": "Section-header"
    },
    {
      "text": "This is a test paragraph.",
      "type": "paragraph",
      "name": "Text"
    }
  ],
  "file-info": {
    "filename": "foo.pdf",
    "document-hash": "123"
  }
}
"""

out_json_obj = {
    "root": [
        {
            "id_": "123",
            "embedding": None,
            "metadata": {"dl_doc_hash": "123"},
            "excluded_embed_metadata_keys": ["dl_doc_hash"],
            "excluded_llm_metadata_keys": ["dl_doc_hash"],
            "relationships": {},
            "text": '{"_name":"foo","type":"pdf-document","description":{"title":null,"abstract":null,"authors":null,"affiliations":null,"subjects":null,"keywords":null,"publication_date":null,"languages":null,"license":null,"publishers":null,"url_refs":null,"references":null,"publication":null,"reference_count":null,"citation_count":null,"citation_date":null,"advanced":null,"analytics":null,"logs":[],"collection":null,"acquisition":null},"file-info":{"filename":"foo.pdf","filename-prov":null,"document-hash":"123","#-pages":null,"collection-name":null,"description":null,"page-hashes":null},"main-text":[{"prov":null,"text":"Test subtitle","type":"subtitle-level-1","name":"Section-header","font":null},{"prov":null,"text":"This is a test paragraph.","type":"paragraph","name":"Text","font":null}],"figures":null,"tables":null,"bitmaps":null,"equations":null,"footnotes":null,"page-dimensions":null,"page-footers":null,"page-headers":null,"_s3_data":null,"identifiers":null}',
            "mimetype": "text/plain",
            "start_char_idx": None,
            "end_char_idx": None,
            "text_template": "{metadata_str}\n\n{content}",
            "metadata_template": "{key}: {value}",
            "metadata_seperator": "\n",
            "class_name": "Document",
        }
    ]
}

out_md_obj = {
    "root": [
        {
            "id_": "123",
            "embedding": None,
            "metadata": {"dl_doc_hash": "123"},
            "excluded_embed_metadata_keys": ["dl_doc_hash"],
            "excluded_llm_metadata_keys": ["dl_doc_hash"],
            "relationships": {},
            "text": "## Test subtitle\n\nThis is a test paragraph.",
            "mimetype": "text/plain",
            "start_char_idx": None,
            "end_char_idx": None,
            "text_template": "{metadata_str}\n\n{content}",
            "metadata_template": "{key}: {value}",
            "metadata_seperator": "\n",
            "class_name": "Document",
        }
    ]
}


def test_lazy_load_data_with_md_export(monkeypatch):
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
    assert act_data == out_md_obj


def test_lazy_load_data_with_json_export(monkeypatch):
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
    assert act_data == out_json_obj


if __name__ == "__main__":
    test_lazy_load_data_with_md_export()
