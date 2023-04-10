"""Test docstore."""


from gpt_index.constants import TYPE_KEY
from gpt_index.data_structs.node_v2 import Node, NodeType
from gpt_index.docstore import DocumentStore
from gpt_index.readers.schema.base import Document


def test_docstore() -> None:
    """Test docstore."""
    doc = Document("hello world", doc_id="d1", extra_info={"foo": "bar"})
    node = Node("my node", doc_id="d2", node_info={"node": "info"})

    # test get document
    docstore = DocumentStore.from_documents([doc, node])
    gd1 = docstore.get_document("d1")
    assert gd1 == doc
    gd2 = docstore.get_document("d2")
    assert gd2 == node

    # test serialize/deserialize
    doc_dict = docstore.serialize_to_dict()
    d1_expected: dict = {
        "text": "hello world",
        "doc_id": "d1",
        "embedding": None,
        "extra_info": {"foo": "bar"},
        TYPE_KEY: "Document",
    }
    d2_expected: dict = {
        "text": "my node",
        "doc_id": "d2",
        "embedding": None,
        "extra_info": None,
        "node_info": {"node": "info"},
        "relationships": {},
        TYPE_KEY: NodeType.TEXT,
    }
    doc_dict["docs"]["d1"].pop("doc_hash")
    doc_dict["docs"]["d2"].pop("doc_hash")
    assert doc_dict["docs"]["d1"] == d1_expected
    assert doc_dict["docs"]["d2"] == d2_expected

    docstore_loaded = DocumentStore.load_from_dict(doc_dict)
    assert docstore_loaded == docstore
