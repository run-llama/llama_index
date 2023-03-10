"""Test docstore."""

from typing import Dict, Type

from gpt_index.data_structs.data_structs import IndexStruct, Node
from gpt_index.docstore import DocumentStore
from gpt_index.readers.schema.base import Document


def test_docstore() -> None:
    """Test docstore."""
    doc = Document("hello world", doc_id="d1", extra_info={"foo": "bar"})
    node = Node("my node", doc_id="d2", node_info={"node": "info"})

    type_to_struct: Dict[str, Type[IndexStruct]] = {"node": Node}

    # test get document
    docstore = DocumentStore.from_documents([doc, node])
    gd1 = docstore.get_document("d1")
    assert gd1 == doc
    gd2 = docstore.get_document("d2")
    assert gd2 == node

    # test serialize/deserialize
    doc_dict = docstore.serialize_to_dict()
    d1_hash = "950fec7ace0281237e5983582c6bb3d76e03cc3931faa695a0fbf039edc69d8c"
    d2_hash = "40d67b32bf21864f2578065c951bb862f99ab0bed0bf527a8680a42b7a027f18"
    assert doc_dict["docs"] == {
        "d1": {
            "text": "hello world",
            "doc_id": "d1",
            "doc_hash": d1_hash,
            "embedding": None,
            "extra_info": {"foo": "bar"},
            "__type__": "Document",
        },
        "d2": {
            "text": "my node",
            "doc_id": "d2",
            "doc_hash": d2_hash,
            "embedding": None,
            "extra_info": None,
            "node_info": {"node": "info"},
            "index": 0,
            "child_indices": [],
            "ref_doc_id": None,
            "__type__": "node",
        },
    }

    docstore_loaded = DocumentStore.load_from_dict(doc_dict, type_to_struct)
    assert docstore_loaded == docstore
