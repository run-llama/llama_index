"""
Failure-atomicity coverage for IngestionPipeline.

A run that fails mid-transformations must leave the docstore and the vector
store exactly as they were, so the same inputs can be safely retried.
"""

from typing import Any, Sequence

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.ingestion.pipeline import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    BaseNode,
    Document,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
    TransformComponent,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore


class RaisingTransform(TransformComponent):
    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        raise RuntimeError("simulated mid-run failure")

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        raise RuntimeError("simulated mid-run failure")


class StrictDeleteVectorStore(SimpleVectorStore):
    """
    Rejects deleting a ref_doc_id with no stored vectors, like non-idempotent
    real backends: repeated deletes for the same document must not happen.
    """

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        if ref_doc_id not in self.data.text_id_to_ref_doc_id.values():
            raise ValueError(f"cannot delete unknown ref_doc_id: {ref_doc_id}")
        super().delete(ref_doc_id, **delete_kwargs)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self.delete(ref_doc_id, **delete_kwargs)


def _chunks_of(ref_doc_id: str, count: int) -> list[TextNode]:
    nodes = [TextNode(text=f"updated chunk {i}", id_=f"node-{i}") for i in range(count)]
    for node in nodes:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=ref_doc_id
        )
    return nodes


def test_duplicates_only_failed_run_leaves_docstore_clean() -> None:
    docstore = SimpleDocumentStore()
    vector_store = SimpleVectorStore()
    doc = Document(text="one sentence. another sentence.", doc_id="1")

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), RaisingTransform()],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )
    with pytest.raises(RuntimeError):
        pipeline.run(documents=[doc])

    # the failed run must not have marked the document as seen
    assert docstore.get_document_hash("1") is None
    assert docstore.docs == {}

    # a clean retry must ingest the document instead of skipping it
    retry = IngestionPipeline(
        transformations=[SentenceSplitter(), MockEmbedding(embed_dim=8)],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )
    nodes = retry.run(documents=[doc])
    assert len(nodes) > 0
    assert docstore.get_document_hash("1") == doc.hash


def test_upserts_failed_rerun_preserves_old_vectors() -> None:
    docstore = SimpleDocumentStore()
    vector_store = SimpleVectorStore()

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), MockEmbedding(embed_dim=8)],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    pipeline.run(documents=[Document(text="old content. more text.", doc_id="1")])
    old_hash = docstore.get_document_hash("1")
    old_vectors = dict(vector_store.data.embedding_dict)
    assert len(old_vectors) > 0

    failing = IngestionPipeline(
        transformations=[SentenceSplitter(), RaisingTransform()],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    with pytest.raises(RuntimeError):
        failing.run(documents=[Document(text="new content entirely.", doc_id="1")])

    # the old ingested state must survive the failed re-ingest untouched
    # (note: top-level Documents never get ref_doc_info entries, so the hash
    # and the vectors are the observable docstore/vector-store state here)
    assert vector_store.data.embedding_dict == old_vectors
    assert docstore.get_document_hash("1") == old_hash
    assert docstore.get_node("1") is not None


def test_upserts_success_still_replaces_old_vectors() -> None:
    docstore = SimpleDocumentStore()
    vector_store = SimpleVectorStore()

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), MockEmbedding(embed_dim=8)],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    pipeline.run(documents=[Document(text="old content. more text.", doc_id="1")])
    old_ids = set(vector_store.data.embedding_dict)

    pipeline.run(documents=[Document(text="new content entirely.", doc_id="1")])
    new_ids = set(vector_store.data.embedding_dict)

    # deferred deletion must still replace, not accumulate or wipe
    assert new_ids
    assert not (old_ids & new_ids)
    assert (
        docstore.get_document_hash("1")
        == Document(text="new content entirely.", doc_id="1").hash
    )


@pytest.mark.asyncio
async def test_async_duplicates_only_failed_run_leaves_docstore_clean() -> None:
    docstore = SimpleDocumentStore()
    vector_store = SimpleVectorStore()
    doc = Document(text="one sentence. another sentence.", doc_id="1")

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), RaisingTransform()],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )
    with pytest.raises(RuntimeError):
        await pipeline.arun(documents=[doc])

    assert docstore.get_document_hash("1") is None
    assert docstore.docs == {}

    # a clean retry must ingest the document instead of skipping it
    retry = IngestionPipeline(
        transformations=[SentenceSplitter(), MockEmbedding(embed_dim=8)],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )
    nodes = await retry.arun(documents=[doc])
    assert len(nodes) > 0
    assert docstore.get_document_hash("1") == doc.hash


@pytest.mark.asyncio
async def test_async_upserts_failed_rerun_preserves_old_vectors() -> None:
    docstore = SimpleDocumentStore()
    vector_store = SimpleVectorStore()

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), MockEmbedding(embed_dim=8)],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    await pipeline.arun(
        documents=[Document(text="old content. more text.", doc_id="1")]
    )
    old_hash = docstore.get_document_hash("1")
    old_vectors = dict(vector_store.data.embedding_dict)
    assert len(old_vectors) > 0

    failing = IngestionPipeline(
        transformations=[SentenceSplitter(), RaisingTransform()],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    with pytest.raises(RuntimeError):
        await failing.arun(
            documents=[Document(text="new content entirely.", doc_id="1")]
        )

    assert vector_store.data.embedding_dict == old_vectors
    assert docstore.get_document_hash("1") == old_hash


def test_upserts_multi_node_update_deletes_old_vectors_once() -> None:
    """
    Regression: when an updated document arrives as multiple nodes sharing one
    ref_doc_id, the old entries must be deleted exactly once. Scheduling one
    delete per node breaks non-idempotent vector backends (the second delete
    targets a document that no longer has vectors) and would abort the run
    after the old vectors are already gone.
    """
    docstore = SimpleDocumentStore()
    vector_store = StrictDeleteVectorStore()

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), MockEmbedding(embed_dim=8)],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    pipeline.run(documents=[Document(text="v1 content.", doc_id="source-doc")])
    assert set(vector_store.data.text_id_to_ref_doc_id.values()) == {"source-doc"}

    result = pipeline.run(nodes=_chunks_of("source-doc", 3))

    assert len(result) == 3
    ref_doc_ids = set(vector_store.data.text_id_to_ref_doc_id.values())
    assert ref_doc_ids == {"source-doc"}
    assert len(vector_store.data.text_id_to_ref_doc_id) == 3


@pytest.mark.asyncio
async def test_async_upserts_multi_node_update_deletes_old_vectors_once() -> None:
    docstore = SimpleDocumentStore()
    vector_store = StrictDeleteVectorStore()

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), MockEmbedding(embed_dim=8)],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    await pipeline.arun(documents=[Document(text="v1 content.", doc_id="source-doc")])

    result = await pipeline.arun(nodes=_chunks_of("source-doc", 3))

    assert len(result) == 3
    assert set(vector_store.data.text_id_to_ref_doc_id.values()) == {"source-doc"}
    assert len(vector_store.data.text_id_to_ref_doc_id) == 3


def test_upserts_and_delete_failed_run_does_not_purge() -> None:
    """
    UPSERTS_AND_DELETE: a document missing from the input is scheduled for
    purge, but the purge must not happen if the run fails mid-transformations.
    """
    docstore = SimpleDocumentStore()
    vector_store = SimpleVectorStore()

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), MockEmbedding(embed_dim=8)],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    )
    pipeline.run(
        documents=[
            Document(text="first document.", doc_id="docA"),
            Document(text="second document.", doc_id="docB"),
        ]
    )
    vectors_before = dict(vector_store.data.embedding_dict)

    failing = IngestionPipeline(
        transformations=[SentenceSplitter(), RaisingTransform()],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    )
    # docB is missing from the input AND docA changed: the failed run must
    # neither purge docB nor drop docA's old version
    with pytest.raises(RuntimeError):
        failing.run(documents=[Document(text="changed first document.", doc_id="docA")])

    assert docstore.get_document_hash("docA") is not None
    assert docstore.get_document_hash("docB") is not None
    assert vector_store.data.embedding_dict == vectors_before


@pytest.mark.asyncio
async def test_async_upserts_and_delete_failed_run_does_not_purge() -> None:
    docstore = SimpleDocumentStore()
    vector_store = SimpleVectorStore()

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), MockEmbedding(embed_dim=8)],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    )
    await pipeline.arun(
        documents=[
            Document(text="first document.", doc_id="docA"),
            Document(text="second document.", doc_id="docB"),
        ]
    )
    vectors_before = dict(vector_store.data.embedding_dict)

    failing = IngestionPipeline(
        transformations=[SentenceSplitter(), RaisingTransform()],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    )
    # docB is missing from the input AND docA changed: the failed run must
    # neither purge docB nor drop docA's old version
    with pytest.raises(RuntimeError):
        await failing.arun(
            documents=[Document(text="changed first document.", doc_id="docA")]
        )

    assert docstore.get_document_hash("docA") is not None
    assert docstore.get_document_hash("docB") is not None
    assert vector_store.data.embedding_dict == vectors_before


@pytest.mark.asyncio
async def test_async_upserts_success_still_replaces_old_vectors() -> None:
    docstore = SimpleDocumentStore()
    vector_store = SimpleVectorStore()

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), MockEmbedding(embed_dim=8)],
        docstore=docstore,
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    await pipeline.arun(
        documents=[Document(text="old content. more text.", doc_id="1")]
    )
    old_ids = set(vector_store.data.embedding_dict)

    await pipeline.arun(documents=[Document(text="new content entirely.", doc_id="1")])
    new_ids = set(vector_store.data.embedding_dict)

    assert new_ids
    assert not (old_ids & new_ids)
