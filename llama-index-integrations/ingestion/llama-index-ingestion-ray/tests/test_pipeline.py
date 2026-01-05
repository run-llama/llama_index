from typing import Sequence, Any
from pathlib import Path
import pytest
import ray.exceptions
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import SentenceSplitter, MarkdownElementNodeParser
from llama_index.core.readers import ReaderConfig, StringIterableReader
from llama_index.core.schema import Document, BaseNode, TransformComponent
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.ingestion.ray import RayIngestionPipeline, RayTransformComponent


def test_build_pipeline() -> None:
    pipeline = RayIngestionPipeline(
        readers=[
            ReaderConfig(
                reader=StringIterableReader(),
                reader_kwargs={"texts": ["This is a test."]},
            )
        ],
        documents=[Document.example()],
        transformations=[
            RayTransformComponent(SentenceSplitter),
            RayTransformComponent(KeywordExtractor, llm=MockLLM()),
            RayTransformComponent(MockEmbedding, embed_dim=8),
        ],
    )

    assert len(pipeline.transformations) == 3


def test_run_pipeline() -> None:
    pipeline = RayIngestionPipeline(
        readers=[
            ReaderConfig(
                reader=StringIterableReader(),
                reader_kwargs={"texts": ["This is a test."]},
            )
        ],
        documents=[Document.example()],
        transformations=[
            RayTransformComponent(SentenceSplitter),
            RayTransformComponent(KeywordExtractor, llm=MockLLM()),
        ],
    )

    nodes = pipeline.run()

    assert len(nodes) == 2
    assert len(nodes[0].metadata) > 0


def test_run_pipeline_with_ref_doc_id():
    documents = [
        Document(text="one", doc_id="1"),
    ]
    pipeline = RayIngestionPipeline(
        documents=documents,
        transformations=[
            RayTransformComponent(MarkdownElementNodeParser, llm=MockLLM()),
            RayTransformComponent(SentenceSplitter),
            RayTransformComponent(MockEmbedding, embed_dim=8),
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

    pipeline = RayIngestionPipeline(
        transformations=[
            RayTransformComponent(SentenceSplitter, chunk_size=25, chunk_overlap=0),
        ],
        docstore=SimpleDocumentStore(),
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

    pipeline2 = RayIngestionPipeline(
        transformations=[
            RayTransformComponent(SentenceSplitter, chunk_size=25, chunk_overlap=0),
        ],
    )

    pipeline2.load("./test_pipeline")

    # dedup will catch the last node
    nodes = pipeline.run(documents=[documents[-1]])
    assert len(nodes) == 0
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

    pipeline = RayIngestionPipeline(
        transformations=[
            RayTransformComponent(SentenceSplitter, chunk_size=25, chunk_overlap=0),
            RayTransformComponent(RaisingTransform),
        ],
        docstore=SimpleDocumentStore(),
    )

    with pytest.raises(ray.exceptions.RayTaskError):
        pipeline.run(documents=[document1])

    assert pipeline.docstore.get_node("1", raise_error=False) is None


@pytest.mark.asyncio
async def test_arun_pipeline() -> None:
    pipeline = RayIngestionPipeline(
        readers=[
            ReaderConfig(
                reader=StringIterableReader(),
                reader_kwargs={"texts": ["This is a test."]},
            )
        ],
        documents=[Document.example()],
        transformations=[
            RayTransformComponent(SentenceSplitter),
            RayTransformComponent(KeywordExtractor, llm=MockLLM()),
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
    pipeline = RayIngestionPipeline(
        documents=documents,
        transformations=[
            RayTransformComponent(MarkdownElementNodeParser, llm=MockLLM()),
            RayTransformComponent(SentenceSplitter),
            RayTransformComponent(MockEmbedding, embed_dim=8),
        ],
    )

    nodes = await pipeline.arun()

    assert len(nodes) == 1
    assert nodes[0].ref_doc_id == "1"


@pytest.mark.asyncio
async def test_async_pipeline_with_transform_error() -> None:
    class RaisingTransform(TransformComponent):
        def __call__(
            self, nodes: Sequence[BaseNode], **kwargs: Any
        ) -> Sequence[BaseNode]:
            raise RuntimeError

    document1 = Document.example()
    document1.id_ = "1"

    pipeline = RayIngestionPipeline(
        transformations=[
            RayTransformComponent(SentenceSplitter, chunk_size=25, chunk_overlap=0),
            RayTransformComponent(RaisingTransform),
        ],
        docstore=SimpleDocumentStore(),
    )

    with pytest.raises(RuntimeError):
        await pipeline.arun(documents=[document1])

    assert pipeline.docstore.get_node("1", raise_error=False) is None
