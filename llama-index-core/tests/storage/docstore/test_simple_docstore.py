"""Test docstore."""

from pathlib import Path

import pytest
from llama_index.core.schema import Document, TextNode, NodeRelationship, RelatedNodeInfo
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
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("new_source", ["new", None])
@pytest.mark.parametrize(("store_text", "allow_update"), [(True, True), (False, False)])
async def test_docstore_reassign_node_source(
    use_async: bool, new_source: str | None, store_text: bool, allow_update: bool
) -> None:
    docstore = SimpleDocumentStore()
    node = TextNode(
        id_="chunk",
        text="original",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="old")},
    )
    if use_async:
        await docstore.async_add_documents([node], store_text=store_text)
    else:
        docstore.add_documents([node], store_text=store_text)
    updated = TextNode(
        id_="chunk",
        text="updated",
        relationships=(
            {NodeRelationship.SOURCE: RelatedNodeInfo(node_id=new_source)}
            if new_source is not None
            else {}
        ),
    )
    if use_async:
        await docstore.async_add_documents([updated], allow_update=allow_update)
    else:
        docstore.add_documents([updated], allow_update=allow_update)

    docstore.delete_ref_doc("old", raise_error=False)

    assert docstore.get_node("chunk") == updated
    if new_source is not None:
        assert docstore.get_ref_doc_info(new_source).node_ids == ["chunk"]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_docstore_swap_node_sources(use_async: bool, tmp_path: Path) -> None:
    docstore = SimpleDocumentStore()
    nodes = [
        TextNode(
            id_=node_id,
            text=node_id,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=source)},
        )
        for node_id, source in [("a", "left"), ("b", "right"), ("c", "left")]
    ]
    updated = [
        TextNode(
            id_=node_id,
            text="updated " + node_id,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=source)},
        )
        for node_id, source in [("a", "right"), ("b", "left")]
    ]
    if use_async:
        await docstore.async_add_documents(nodes)
        await docstore.async_add_documents(updated, batch_size=1)
    else:
        docstore.add_documents(nodes)
        docstore.add_documents(updated, batch_size=1)

    persist_path = str(tmp_path / "docstore.json")
    docstore.persist(persist_path)
    restored = SimpleDocumentStore.from_persist_path(persist_path)
    assert set(restored.get_ref_doc_info("left").node_ids) == {"b", "c"}
    assert restored.get_ref_doc_info("right").node_ids == ["a"]
    restored.delete_ref_doc("left")
    assert restored.get_node("a") == updated[0]
    assert not restored.document_exists("b")
    assert not restored.document_exists("c")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("store_text", [False, True])
async def test_docstore_last_source_in_batch(use_async: bool, store_text: bool) -> None:
    docstore = SimpleDocumentStore()
    versions = [
        TextNode(
            id_="chunk",
            text=source,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=source)},
        )
        for source in ["old", "intermediate", "final"]
    ]
    if use_async:
        await docstore.async_add_documents(versions[:1], store_text=store_text)
        await docstore.async_add_documents(
            versions[1:], store_text=store_text, batch_size=1
        )
    else:
        docstore.add_documents(versions[:1], store_text=store_text)
        docstore.add_documents(versions[1:], store_text=store_text, batch_size=1)

    assert docstore.get_ref_doc_info("old").node_ids == []
    assert docstore.get_ref_doc_info("intermediate").node_ids == []
    assert docstore.get_ref_doc_info("final").node_ids == ["chunk"]
    docstore.delete_ref_doc("old")
    docstore.delete_ref_doc("intermediate")
    assert docstore.get_document_hash("chunk") == versions[-1].hash
    if store_text:
        assert docstore.get_node("chunk") == versions[-1]
    else:
        assert not docstore.document_exists("chunk")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_docstore_update_with_same_source(use_async: bool) -> None:
    docstore = SimpleDocumentStore()
    original = TextNode(
        id_="chunk",
        text="original",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="source")},
    )
    updated = original.model_copy(update={"text": "updated"})
    if use_async:
        await docstore.async_add_documents([original])
        with pytest.raises(ValueError, match="Set allow_update to True"):
            await docstore.async_add_documents([updated], allow_update=False)
        assert docstore.get_node("chunk") == original
        await docstore.async_add_documents([updated])
    else:
        docstore.add_documents([original])
        with pytest.raises(ValueError, match="Set allow_update to True"):
            docstore.add_documents([updated], allow_update=False)
        assert docstore.get_node("chunk") == original
        docstore.add_documents([updated])

    assert docstore.get_ref_doc_info("source").node_ids == ["chunk"]
    assert docstore.get_node("chunk") == updated
    docstore.delete_ref_doc("source")
    assert not docstore.document_exists("chunk")
