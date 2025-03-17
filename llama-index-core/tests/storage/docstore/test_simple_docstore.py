"""Test docstore."""

from pathlib import Path

import pytest
from llama_index.core.schema import Document, TextNode, NodeRelationship
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore


@pytest.fixture()
def simple_docstore(simple_kvstore: SimpleKVStore) -> SimpleDocumentStore:
    return SimpleDocumentStore(simple_kvstore=simple_kvstore)


def test_docstore(simple_docstore: SimpleDocumentStore) -> None:
    """Test docstore."""
    doc = Document(text="hello world", id_="d1", metadata={"foo": "bar"})
    node = TextNode(text="my node", id_="d2", metadata={"node": "info"})

    # test get document
    docstore = simple_docstore
    docstore.add_documents([doc, node])
    gd1 = docstore.get_document("d1")
    assert gd1 == doc
    gd2 = docstore.get_document("d2")
    assert gd2 == node


def test_docstore_persist(tmp_path: Path) -> None:
    """Test docstore."""
    persist_path = str(tmp_path / "test_file.txt")
    doc = Document(text="hello world", id_="d1", metadata={"foo": "bar"})
    node = TextNode(text="my node", id_="d2", metadata={"node": "info"})

    # add documents and then persist to dir
    docstore = SimpleDocumentStore()
    docstore.add_documents([doc, node])
    docstore.persist(persist_path)

    # load from persist dir and get documents
    new_docstore = SimpleDocumentStore.from_persist_path(persist_path)
    gd1 = new_docstore.get_document("d1")
    assert gd1 == doc
    gd2 = new_docstore.get_document("d2")
    assert gd2 == node


def test_docstore_dict() -> None:
    doc = Document(text="hello world", id_="d1", metadata={"foo": "bar"})
    node = TextNode(text="my node", id_="d2", metadata={"node": "info"})

    # add documents and then save to dict
    docstore = SimpleDocumentStore()
    docstore.add_documents([doc, node])
    save_dict = docstore.to_dict()

    # load from dict and get documents
    new_docstore = SimpleDocumentStore.from_dict(save_dict)
    gd1 = new_docstore.get_document("d1")
    assert gd1 == doc
    gd2 = new_docstore.get_document("d2")
    assert gd2 == node


def test_docstore_delete_document() -> None:
    doc = Document(text="hello world", id_="d1", metadata={"foo": "bar"})
    node = TextNode(text="my node", id_="d2", metadata={"node": "info"})

    docstore = SimpleDocumentStore()
    docstore.add_documents([doc, node])
    docstore.delete_document("d1")

    assert docstore._kvstore.get("d1", docstore._node_collection) is None
    assert docstore._kvstore.get("d1", docstore._metadata_collection) is None
    assert docstore._kvstore.get("d1", docstore._ref_doc_collection) is None

    assert docstore._kvstore.get("d2", docstore._node_collection) is not None
    assert docstore._kvstore.get("d2", docstore._metadata_collection) is not None


def test_docstore_delete_ref_doc() -> None:
    ref_doc = Document(text="hello world", id_="d1", metadata={"foo": "bar"})
    doc = Document(text="hello world", id_="d2", metadata={"foo": "bar"})
    doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
    node = TextNode(text="my node", id_="d3", metadata={"node": "info"})
    node.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

    docstore = SimpleDocumentStore()
    docstore.add_documents([ref_doc, doc, node])
    docstore.delete_ref_doc("d1")

    assert docstore._kvstore.get("d1", docstore._node_collection) is None
    assert docstore._kvstore.get("d1", docstore._metadata_collection) is None
    assert docstore._kvstore.get("d1", docstore._ref_doc_collection) is None
    assert docstore._kvstore.get("d2", docstore._node_collection) is None
    assert docstore._kvstore.get("d2", docstore._metadata_collection) is None
    assert docstore._kvstore.get("d2", docstore._ref_doc_collection) is None
    assert docstore._kvstore.get("d3", docstore._node_collection) is None
    assert docstore._kvstore.get("d3", docstore._metadata_collection) is None
    assert docstore._kvstore.get("d3", docstore._ref_doc_collection) is None


def test_docstore_delete_ref_doc_not_in_docstore() -> None:
    ref_doc = Document(text="hello world", id_="d1", metadata={"foo": "bar"})
    doc = Document(text="hello world", id_="d2", metadata={"foo": "bar"})
    doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
    node = TextNode(text="my node", id_="d3", metadata={"node": "info"})
    node.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

    docstore = SimpleDocumentStore()
    docstore.add_documents([doc, node])
    assert docstore._kvstore.get("d1", docstore._ref_doc_collection) is not None

    docstore.delete_ref_doc("d1")

    assert docstore._kvstore.get("d1", docstore._node_collection) is None
    assert docstore._kvstore.get("d1", docstore._metadata_collection) is None
    assert docstore._kvstore.get("d1", docstore._ref_doc_collection) is None
    assert docstore._kvstore.get("d2", docstore._node_collection) is None
    assert docstore._kvstore.get("d2", docstore._metadata_collection) is None
    assert docstore._kvstore.get("d2", docstore._ref_doc_collection) is None
    assert docstore._kvstore.get("d3", docstore._node_collection) is None
    assert docstore._kvstore.get("d3", docstore._metadata_collection) is None
    assert docstore._kvstore.get("d3", docstore._ref_doc_collection) is None


def test_docstore_delete_all_ref_doc_nodes() -> None:
    ref_doc = Document(text="hello world", id_="d1", metadata={"foo": "bar"})
    doc = Document(text="hello world", id_="d2", metadata={"foo": "bar"})
    doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
    node = TextNode(text="my node", id_="d3", metadata={"node": "info"})
    node.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

    docstore = SimpleDocumentStore()
    docstore.add_documents([ref_doc, doc, node])

    assert docstore._kvstore.get("d1", docstore._ref_doc_collection)["node_ids"] == [
        "d2",
        "d3",
    ]

    docstore.delete_document("d2")
    assert docstore._kvstore.get("d1", docstore._node_collection) is not None
    assert docstore._kvstore.get("d1", docstore._metadata_collection) is not None
    assert docstore._kvstore.get("d1", docstore._ref_doc_collection) is not None
    assert docstore._kvstore.get("d1", docstore._ref_doc_collection)["node_ids"] == [
        "d3"
    ]

    docstore.delete_document("d3")
    assert docstore._kvstore.get("d1", docstore._node_collection) is None
    assert docstore._kvstore.get("d1", docstore._metadata_collection) is None
    assert docstore._kvstore.get("d1", docstore._ref_doc_collection) is None
