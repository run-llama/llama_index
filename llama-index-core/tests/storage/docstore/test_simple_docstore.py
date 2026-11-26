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


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("batch", [False, True])
async def test_hash_updates_preserve_ref_doc_cleanup(
    batch: bool, asynchronous: bool
) -> None:
    ref_doc = Document(text="source", id_="source")
    nodes = [
        TextNode(
            id_=node_id,
            text=node_id,
            relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
        )
        for node_id in ("first", "second")
    ]
    docstore = SimpleDocumentStore()
    docstore.add_documents([ref_doc, *nodes])

    if asynchronous:
        if batch:
            await docstore.aset_document_hashes({"first": "updated", "new": "fresh"})
        else:
            await docstore.aset_document_hash("first", "updated")
            await docstore.aset_document_hash("new", "fresh")
    elif batch:
        docstore.set_document_hashes({"first": "updated", "new": "fresh"})
    else:
        docstore.set_document_hash("first", "updated")
        docstore.set_document_hash("new", "fresh")

    assert docstore.get_document_hash("first") == "updated"
    assert docstore.get_document_hash("second") == nodes[1].hash
    assert docstore.get_document_hash("new") == "fresh"
    assert docstore.get_document("first") == nodes[0]
    docstore.delete_document("first")
    ref_doc_info = docstore.get_ref_doc_info("source")
    assert ref_doc_info is not None
    assert ref_doc_info.node_ids == ["second"]
    docstore.delete_document("second")
    assert docstore.get_ref_doc_info("source") is None
    assert not docstore.document_exists("source")
    assert docstore.get_document_hash("source") is None
