from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from llama_index.core.schema import Document, NodeRelationship, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore
from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
)


async def update_hashes(
    store: KVDocumentStore, setter: str, hashes: Dict[str, str]
) -> None:
    if setter == "single":
        for doc_id, doc_hash in hashes.items():
            store.set_document_hash(doc_id, doc_hash)
    elif setter == "batch":
        store.set_document_hashes(hashes)
    elif setter == "async_single":
        for doc_id, doc_hash in hashes.items():
            await store.aset_document_hash(doc_id, doc_hash)
    elif setter == "async_batch":
        await store.aset_document_hashes(hashes)
    else:
        raise ValueError(f"Unknown test setter: {setter}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setter", ["single", "batch", "async_single", "async_batch", "none"]
)
@pytest.mark.parametrize("reload", [False, True])
async def test_hash_update_keeps_reference_cleanup(
    setter: str, reload: bool, tmp_path: Path
) -> None:
    source = Document(id_="source", text="source")
    nodes = [
        TextNode(
            id_=node_id,
            text=node_id,
            relationships={NodeRelationship.SOURCE: source.as_related_node_info()},
        )
        for node_id in ("n1", "n2")
    ]
    store = SimpleDocumentStore()
    store.add_documents(nodes)
    old_n1_hash = store.get_document_hash("n1")
    old_n2_hash = store.get_document_hash("n2")
    if setter != "none":
        await update_hashes(store, setter, {"n1": "changed-hash"})

    if reload:
        path = str(tmp_path / "docstore.json")
        store.persist(path)
        store = SimpleDocumentStore.from_persist_path(path)

    assert store.get_document_hash("n1") == (
        "changed-hash" if setter != "none" else old_n1_hash
    )
    assert await store.aget_document_hash("n1") == store.get_document_hash("n1")
    assert store.get_node("n1").ref_doc_id == "source"
    assert store.get_document_hash("n2") == old_n2_hash

    # Keep deletion synchronous to isolate the hash setter's metadata update.
    store.delete_document("n1")
    info = store.get_ref_doc_info("source")
    assert not store.document_exists("n1")
    assert store.get_node("n2").get_content() == "n2"
    assert info is not None
    assert info.node_ids == ["n2"]


class RecordingKVStore(SimpleKVStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads: List[Tuple[str, str, str]] = []
        self.batch_calls: List[Tuple[str, List[Tuple[str, dict]], str, int]] = []

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> dict | None:
        self.reads.append(("get", key, collection))
        return super().get(key, collection=collection)

    async def aget(self, key: str, collection: str = DEFAULT_COLLECTION) -> dict | None:
        self.reads.append(("aget", key, collection))
        return super().get(key, collection=collection)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        raise AssertionError("Hash updates must not scan the full store")

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        raise AssertionError("Hash updates must not scan the full store")

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.batch_calls.append(("put_all", kv_pairs, collection, batch_size))
        for key, value in kv_pairs:
            self.put(key, value, collection=collection)

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.batch_calls.append(("aput_all", kv_pairs, collection, batch_size))
        for key, value in kv_pairs:
            await self.aput(key, value, collection=collection)


@pytest.mark.asyncio
@pytest.mark.parametrize("setter", ["single", "batch", "async_single", "async_batch"])
async def test_hash_update_preserves_metadata_and_creates_missing_row(
    setter: str,
) -> None:
    backend = RecordingKVStore()
    store = KVDocumentStore(
        backend, namespace="scope", metadata_collection_suffix="/meta", batch_size=7
    )
    existing = {"doc_hash": "old", "ref_doc_id": "source", "other": {"tag": "keep"}}
    backend.put("existing", existing, collection="scope/meta")
    hashes = {"existing": "updated", "missing": "created"}

    await update_hashes(store, setter, hashes)

    assert backend.get("existing", collection="scope/meta") == {
        **existing,
        "doc_hash": "updated",
    }
    assert backend.get("missing", collection="scope/meta") == {"doc_hash": "created"}
    assert store.get_document_hash("existing") == "updated"
    assert store.get_document_hash("missing") == "created"
    if setter in ("batch", "async_batch"):
        assert len(backend.batch_calls) == 1
        method, pairs, collection, batch_size = backend.batch_calls[0]
        assert method == ("put_all" if setter == "batch" else "aput_all")
        assert [key for key, _ in pairs] == ["existing", "missing"]
        assert collection == "scope/meta"
        assert batch_size == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("setter", ["batch", "async_batch"])
async def test_empty_hash_batch_keeps_backend_call_without_reads(setter: str) -> None:
    backend = RecordingKVStore()
    store = KVDocumentStore(
        backend, namespace="scope", metadata_collection_suffix="/meta", batch_size=7
    )
    await update_hashes(store, setter, {})
    assert backend.reads == []
    assert backend.batch_calls == [
        ("put_all" if setter == "batch" else "aput_all", [], "scope/meta", 7)
    ]
