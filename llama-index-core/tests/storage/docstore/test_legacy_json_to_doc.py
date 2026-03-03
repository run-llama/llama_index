import pytest

from llama_index.core.constants import DATA_KEY, TYPE_KEY
from llama_index.core.schema import Document, TextNode, ImageNode, IndexNode
from llama_index.core.storage.docstore.utils import legacy_json_to_doc


def _make_legacy_dict(node_cls, doc_id: str, *, extra: dict | None = None) -> dict:
    data = {
        "text": "hello",
        "extra_info": {},
        "doc_id": doc_id,
        "relationships": {},
    }
    if extra:
        data.update(extra)
    return {TYPE_KEY: node_cls.get_type(), DATA_KEY: data}


@pytest.mark.parametrize(
    ("node_cls", "doc_id", "extra"),
    [
        (Document, "doc-123", None),
        (TextNode, "text-456", None),
        (ImageNode, "img-789", {"image": None}),
        (IndexNode, "idx-999", {"index_id": "index-abc"}),
    ],
)
def test_legacy_json_to_doc_preserves_doc_id(node_cls, doc_id, extra):
    doc_dict = _make_legacy_dict(node_cls, doc_id, extra=extra)
    node = legacy_json_to_doc(doc_dict)
    assert node.id_ == doc_id


def test_legacy_json_to_doc_unknown_type_raises():
    doc_dict = {
        TYPE_KEY: "not-a-real-node-type",
        DATA_KEY: {"text": "hello", "extra_info": {}, "doc_id": "x", "relationships": {}},
    }
    with pytest.raises(ValueError):
        legacy_json_to_doc(doc_dict)
