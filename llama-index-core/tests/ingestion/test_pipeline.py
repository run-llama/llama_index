from multiprocessing import cpu_count
from pathlib import Path
from typing import Sequence, Any

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.ingestion.pipeline import IngestionPipeline, DocstoreStrategy
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import SentenceSplitter, MarkdownElementNodeParser
from llama_index.core.readers import ReaderConfig, StringIterableReader
from llama_index.core.schema import Document, TransformComponent, BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore


def test_build_pipeline() -> None:
    pipeline = IngestionPipeline(
        readers=[
            ReaderConfig(
                reader=StringIterableReader(),
                reader_kwargs={"texts": ["This is a test."]},
            )
        ],
        documents=[Document.example()],
        transformations=[
            SentenceSplitter(),
            KeywordExtractor(llm=MockLLM()),
            MockEmbedding(embed_dim=8),
        ],
    )

    assert len(pipeline.transformations) == 3


def test_run_pipeline() -> None:
    pipeline = IngestionPipeline(
        readers=[
            ReaderConfig(
                reader=StringIterableReader(),
                reader_kwargs={"texts": ["This is a test."]},
            )
        ],
        documents=[Document.example()],
        transformations=[
            SentenceSplitter(),
            KeywordExtractor(llm=MockLLM()),
        ],
    )

    nodes = pipeline.run()

    assert len(nodes) == 2
    assert len(nodes[0].metadata) > 0


def test_run_pipeline_with_ref_doc_id():
    documents = [
        Document(text="one", doc_id="1"),
    ]
    pipeline = IngestionPipeline(
        documents=documents,
        transformations=[
            MarkdownElementNodeParser(),
            SentenceSplitter(),
            MockEmbedding(embed_dim=8),
        ],
    )

    nodes = pipeline.run()

    assert len(nodes) == 1
    assert nodes[0].ref_doc_id == "1"


def test_save_load_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    documents = [
        Document(text="one", doc_id="1"),
        Document(text="two", doc_id="2"),
        Document(text="one", doc_id="1"),
    ]

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    nodes = pipeline.run(documents=documents)
    assert len(nodes) == 2
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 2

    # dedup will catch the last node
    nodes = pipeline.run(documents=[documents[-1]])
    assert len(nodes) == 0
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 2

    # test save/load
    pipeline.persist("./test_pipeline")

    pipeline2 = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
    )

    pipeline2.load("./test_pipeline")

    # dedup will catch the last node
    nodes = pipeline.run(documents=[documents[-1]])
    assert len(nodes) == 0
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 2


def test_save_load_pipeline_without_docstore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    documents = [
        Document(text="one", doc_id="1"),
        Document(text="two", doc_id="2"),
        Document(text="one", doc_id="1"),
    ]

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
    )

    nodes = pipeline.run(documents=documents)
    assert len(nodes) == 3
    assert pipeline.docstore is None

    # dedup will not catch the last node if the document store is not set
    nodes = pipeline.run(documents=[documents[-1]])
    assert len(nodes) == 1
    assert pipeline.docstore is None

    # test save/load
    pipeline.persist("./test_pipeline")

    pipeline2 = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
    )

    pipeline2.load("./test_pipeline")

    # dedup will not catch the last node if the document store is not set
    nodes = pipeline.run(documents=[documents[-1]])
    assert len(nodes) == 1
    assert pipeline.docstore is None


def test_pipeline_update_text_content() -> None:
    document1 = Document.example()
    document1.id_ = "1"

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    nodes = pipeline.run(documents=[document1])
    assert len(nodes) == 19
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 1

    # adjust document content
    document1 = Document(text="test", doc_id="1")

    # run pipeline again
    nodes = pipeline.run(documents=[document1])

    assert len(nodes) == 1
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 1
    assert next(iter(pipeline.docstore.docs.values())).text == "test"  # type: ignore


def test_pipeline_update_metadata() -> None:
    """Test that IngestionPipeline updates document metadata, if it changed."""
    old_metadata = {"filename": "README.md", "category": "codebase"}
    document1 = Document.example()
    document1.metadata = old_metadata
    document1.id_ = "1"

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    nodes = pipeline.run(documents=[document1])
    assert len(nodes) >= 1
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 1
    for node in nodes:
        assert node.metadata == old_metadata

    # adjust document metadata
    new_metadata = {"filename": "README.md", "category": "documentation"}
    document1.metadata = new_metadata

    # run pipeline again
    nodes_new = pipeline.run(documents=[document1])

    assert len(nodes_new) == len(nodes)
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 1
    assert next(iter(pipeline.docstore.docs.values())).metadata == new_metadata  # type: ignore
    for node in nodes_new:
        assert node.metadata == new_metadata


def test_pipeline_does_not_re_embed_on_volatile_metadata_change() -> None:
    """Regression test: fields listed in `excluded_embed_metadata_keys` must
    NOT trigger re-embedding on upsert. This covers the churn scenario where
    volatile metadata such as file-stat timestamps or sizes would otherwise
    force redundant embedding API calls for byte-identical content.

    Companion to `test_pipeline_update_metadata`, which verifies the inverse:
    a change to a content-relevant (non-excluded) metadata field DOES trigger
    re-processing.
    """
    from unittest.mock import MagicMock

    embed_model = MockEmbedding(embed_dim=4)
    # Wrap the batch method so we can count invocations.
    embed_model.get_text_embedding_batch = MagicMock(  # type: ignore[method-assign]
        wraps=embed_model.get_text_embedding_batch
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=256, chunk_overlap=0),
            embed_model,
        ],
        docstore=SimpleDocumentStore(),
    )

    doc = Document(
        id_="0",
        text="hello world, this is a short test document for the pipeline",
        metadata={"tag": "v1", "last_modified_date": "2026-04-22"},
        excluded_embed_metadata_keys=["last_modified_date"],
    )

    # Phase 1: first ingest (should embed).
    pipeline.run(documents=[doc])
    embed_calls_after_phase1 = embed_model.get_text_embedding_batch.call_count
    assert embed_calls_after_phase1 >= 1

    # Phase 2: mutate only the volatile field, which is in
    # `excluded_embed_metadata_keys`. This must NOT re-embed.
    doc.metadata["last_modified_date"] = "2026-04-23"
    pipeline.run(documents=[doc])
    assert embed_model.get_text_embedding_batch.call_count == embed_calls_after_phase1, (
        "Re-embedding was triggered by a change to metadata listed in "
        "excluded_embed_metadata_keys. This re-introduces the churn bug."
    )

    # Phase 3: mutate a non-excluded (content-relevant) field. This SHOULD
    # re-embed, preserving the behavior added in #18303 for #17871.
    doc.metadata["tag"] = "v2"
    pipeline.run(documents=[doc])
    assert embed_model.get_text_embedding_batch.call_count > embed_calls_after_phase1, (
        "Content-relevant metadata change was not detected. The fix must still "
        "invalidate the cache when a non-excluded field changes."
    )


def test_pipeline_dedup_duplicates_only() -> None:
    documents = [
        Document(text="one", doc_id="1"),
        Document(text="two", doc_id="2"),
        Document(text="three", doc_id="3"),
    ]

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    nodes = pipeline.run(documents=documents)
    assert len(nodes) == 3

    nodes = pipeline.run(documents=documents)
    assert len(nodes) == 0


@pytest.mark.skipif(cpu_count() < 2, reason="requires at least 2 CPUs")
def test_pipeline_parallel_cache_populated() -> None:
    num_workers = 2
    docs = [
        Document(text=f"Sample document {i}." * 20, doc_id=str(i)) for i in range(4)
    ]
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=25, chunk_overlap=0)]
    )

    pipeline.run(documents=docs, num_workers=num_workers)

    cache_size = len(pipeline.cache.cache.get_all(collection=pipeline.cache.collection))
    assert cache_size == num_workers


@pytest.mark.skipif(cpu_count() < 2, reason="requires at least 2 CPUs")
def test_pipeline_parallel_cache_reused_on_second_run() -> None:
    num_workers = 2
    docs = [
        Document(text=f"Sample document {i}." * 20, doc_id=str(i)) for i in range(4)
    ]
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=25, chunk_overlap=0)]
    )

    pipeline.run(documents=docs, num_workers=num_workers)
    first_size = len(pipeline.cache.cache.get_all(collection=pipeline.cache.collection))

    pipeline.run(documents=docs, num_workers=num_workers)
    second_size = len(
        pipeline.cache.cache.get_all(collection=pipeline.cache.collection)
    )

    assert first_size == num_workers
    assert second_size == first_size


def test_pipeline_parallel() -> None:
    document1 = Document.example()
    document1.id_ = "1"
    document2 = Document(text="One\n\n\nTwo\n\n\nThree.", doc_id="2")

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    num_workers = min(2, cpu_count())
    nodes = pipeline.run(documents=[document1, document2], num_workers=num_workers)
    assert len(nodes) == 20
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 2


def test_pipeline_with_transform_error() -> None:
    class RaisingTransform(TransformComponent):
        def __call__(
            self, nodes: Sequence[BaseNode], **kwargs: Any
        ) -> Sequence[BaseNode]:
            raise RuntimeError

    document1 = Document.example()
    document1.id_ = "1"

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            RaisingTransform(),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    with pytest.raises(RuntimeError):
        pipeline.run(documents=[document1])

    assert pipeline.docstore.get_node("1", raise_error=False) is None


@pytest.mark.asyncio
async def test_arun_pipeline() -> None:
    pipeline = IngestionPipeline(
        readers=[
            ReaderConfig(
                reader=StringIterableReader(),
                reader_kwargs={"texts": ["This is a test."]},
            )
        ],
        documents=[Document.example()],
        transformations=[
            SentenceSplitter(),
            KeywordExtractor(llm=MockLLM()),
        ],
    )

    nodes = await pipeline.arun()

    assert len(nodes) == 2
    assert len(nodes[0].metadata) > 0


@pytest.mark.asyncio
async def test_arun_pipeline_with_ref_doc_id():
    documents = [
        Document(text="one", doc_id="1"),
    ]
    pipeline = IngestionPipeline(
        documents=documents,
        transformations=[
            MarkdownElementNodeParser(),
            SentenceSplitter(),
            MockEmbedding(embed_dim=8),
        ],
    )

    nodes = await pipeline.arun()

    assert len(nodes) == 1
    assert nodes[0].ref_doc_id == "1"


@pytest.mark.asyncio
async def test_async_pipeline_update_text_content() -> None:
    document1 = Document.example()
    document1.id_ = "1"

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    nodes = await pipeline.arun(documents=[document1])
    assert len(nodes) == 19
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 1

    # adjust document content
    document1 = Document(text="test", doc_id="1")

    # run pipeline again
    nodes = pipeline.run(documents=[document1])

    assert len(nodes) == 1
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 1
    assert next(iter(pipeline.docstore.docs.values())).text == "test"  # type: ignore


@pytest.mark.asyncio
async def test_async_pipeline_update_metadata() -> None:
    """Test that IngestionPipeline updates document metadata, if it changed."""
    old_metadata = {"filename": "README.md", "category": "codebase"}
    document1 = Document.example()
    document1.metadata = old_metadata
    document1.id_ = "1"

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    nodes = await pipeline.arun(documents=[document1])
    assert len(nodes) >= 1
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 1
    for node in nodes:
        assert node.metadata == old_metadata

    # adjust document metadata
    new_metadata = {"filename": "README.md", "category": "documentation"}
    document1.metadata = new_metadata

    # run pipeline again
    nodes_new = pipeline.run(documents=[document1])

    assert len(nodes_new) == len(nodes)
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 1
    assert next(iter(pipeline.docstore.docs.values())).metadata == new_metadata  # type: ignore
    for node in nodes_new:
        assert node.metadata == new_metadata


@pytest.mark.asyncio
async def test_async_pipeline_dedup_duplicates_only() -> None:
    documents = [
        Document(text="one", doc_id="1"),
        Document(text="two", doc_id="2"),
        Document(text="three", doc_id="3"),
    ]

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    nodes = await pipeline.arun(documents=documents)
    assert len(nodes) == 3

    nodes = await pipeline.arun(documents=documents)
    assert len(nodes) == 0


@pytest.mark.asyncio
@pytest.mark.skipif(cpu_count() < 2, reason="requires at least 2 CPUs")
async def test_async_pipeline_parallel_cache_populated() -> None:
    num_workers = 2
    docs = [
        Document(text=f"Sample document {i}." * 20, doc_id=str(i)) for i in range(4)
    ]
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=25, chunk_overlap=0)]
    )

    await pipeline.arun(documents=docs, num_workers=num_workers)

    cache_size = len(pipeline.cache.cache.get_all(collection=pipeline.cache.collection))
    assert cache_size == num_workers


@pytest.mark.asyncio
@pytest.mark.skipif(cpu_count() < 2, reason="requires at least 2 CPUs")
async def test_async_pipeline_parallel_cache_reused_on_second_run() -> None:
    num_workers = 2
    docs = [
        Document(text=f"Sample document {i}." * 20, doc_id=str(i)) for i in range(4)
    ]
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=25, chunk_overlap=0)]
    )

    await pipeline.arun(documents=docs, num_workers=num_workers)
    first_size = len(pipeline.cache.cache.get_all(collection=pipeline.cache.collection))

    await pipeline.arun(documents=docs, num_workers=num_workers)
    second_size = len(
        pipeline.cache.cache.get_all(collection=pipeline.cache.collection)
    )

    assert first_size == num_workers
    assert second_size == first_size


@pytest.mark.asyncio
async def test_async_pipeline_parallel() -> None:
    document1 = Document.example()
    document1.id_ = "1"
    document2 = Document(text="One\n\n\nTwo\n\n\nThree.", doc_id="2")

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    num_workers = min(2, cpu_count())
    nodes = await pipeline.arun(
        documents=[document1, document2], num_workers=num_workers
    )
    assert len(nodes) == 20
    assert pipeline.docstore is not None
    assert len(pipeline.docstore.docs) == 2


@pytest.mark.asyncio
async def test_async_pipeline_with_transform_error() -> None:
    class RaisingTransform(TransformComponent):
        def __call__(
            self, nodes: Sequence[BaseNode], **kwargs: Any
        ) -> Sequence[BaseNode]:
            raise RuntimeError

    document1 = Document.example()
    document1.id_ = "1"

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            RaisingTransform(),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    with pytest.raises(RuntimeError):
        await pipeline.arun(documents=[document1])

    assert pipeline.docstore.get_node("1", raise_error=False) is None


def test_docstore_strategy_not_mutated_on_run_without_vector_store() -> None:
    for strategy in (DocstoreStrategy.UPSERTS, DocstoreStrategy.UPSERTS_AND_DELETE):
        pipeline = IngestionPipeline(
            transformations=[],
            docstore=SimpleDocumentStore(),
            docstore_strategy=strategy,
        )
        with pytest.warns(UserWarning, match="requires a vector store"):
            pipeline.run(documents=[Document.example()])

        assert pipeline.docstore_strategy is strategy


@pytest.mark.asyncio
async def test_docstore_strategy_not_mutated_on_arun_without_vector_store() -> None:
    for strategy in (DocstoreStrategy.UPSERTS, DocstoreStrategy.UPSERTS_AND_DELETE):
        pipeline = IngestionPipeline(
            transformations=[],
            docstore=SimpleDocumentStore(),
            docstore_strategy=strategy,
        )
        with pytest.warns(UserWarning, match="requires a vector store"):
            await pipeline.arun(documents=[Document.example()])

        assert pipeline.docstore_strategy is strategy
