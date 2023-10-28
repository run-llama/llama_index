import pytest
from llama_index.embeddings import OpenAIEmbedding
from llama_index.extractors import KeywordExtractor
from llama_index.ingestion.pipeline import IngestionPipeline
from llama_index.llms import MockLLM
from llama_index.node_parser import SentenceAwareNodeParser
from llama_index.readers import ReaderConfig, StringIterableReader
from llama_index.schema import Document


def test_build_pipeline() -> None:
    pipeline = IngestionPipeline(
        name="Test",
        reader=ReaderConfig(
            reader=StringIterableReader(), reader_kwargs={"texts": ["This is a test."]}
        ),
        documents=[Document.example()],
        transformations=[
            SentenceAwareNodeParser(),
            KeywordExtractor(llm=MockLLM()),
            OpenAIEmbedding(api_key="fake"),
        ],
    )

    assert len(pipeline.transformations) == 3
    assert len(pipeline.configured_transformations) == 3
    assert pipeline.name == "Test"


def test_run_local_pipeline() -> None:
    pipeline = IngestionPipeline(
        name="Test",
        reader=ReaderConfig(
            reader=StringIterableReader(), reader_kwargs={"texts": ["This is a test."]}
        ),
        documents=[Document.example()],
        transformations=[
            SentenceAwareNodeParser(),
            KeywordExtractor(llm=MockLLM()),
        ],
    )

    nodes = pipeline.run_local()

    assert len(nodes) == 2
    assert len(nodes[0].metadata) > 0


@pytest.mark.integration()
def test_register() -> None:
    pipeline = IngestionPipeline(
        name="Test",
        reader=ReaderConfig(
            reader=StringIterableReader(), reader_kwargs={"texts": ["This is a test."]}
        ),
        documents=[Document.example()],
        transformations=[
            SentenceAwareNodeParser(),
            KeywordExtractor(llm=MockLLM()),
        ],
    )

    pipeline_id = pipeline.register()

    # update pipeline
    updated_pipeline = IngestionPipeline(
        name="Test",
        reader=ReaderConfig(
            reader=StringIterableReader(),
            reader_kwargs={"texts": ["This is another test."]},
        ),
        documents=[Document.example()],
        transformations=[
            SentenceAwareNodeParser(),
            KeywordExtractor(llm=MockLLM()),
        ],
    )

    new_pipeline_id = updated_pipeline.register()

    # make sure we are updating the same pipeline instead of creating a new one
    assert pipeline_id == new_pipeline_id


def test_from_pipeline_name() -> None:
    pipeline = IngestionPipeline(
        name="Test",
        reader=ReaderConfig(
            reader=StringIterableReader(), reader_kwargs={"texts": ["This is a test."]}
        ),
        documents=[Document.example()],
        transformations=[
            SentenceAwareNodeParser(),
            KeywordExtractor(llm=MockLLM()),
        ],
    )

    pipeline.register()

    IngestionPipeline.from_pipeline_name("Test")
