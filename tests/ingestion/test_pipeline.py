from llama_index.embeddings import OpenAIEmbedding
from llama_index.extractors import KeywordExtractor
from llama_index.ingestion.pipeline import IngestionPipeline
from llama_index.llms import MockLLM
from llama_index.node_parser import SentenceAwareNodeParser
from llama_index.readers import ReaderConfig, StringIterableReader
from llama_index.schema import Document


def test_build_pipeline() -> None:
    pipeline = IngestionPipeline(
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


def test_run_pipeline() -> None:
    pipeline = IngestionPipeline(
        reader=ReaderConfig(
            reader=StringIterableReader(), reader_kwargs={"texts": ["This is a test."]}
        ),
        documents=[Document.example()],
        transformations=[
            SentenceAwareNodeParser(),
            KeywordExtractor(llm=MockLLM()),
        ],
    )

    nodes = pipeline.run()

    assert len(nodes) == 2
    assert len(nodes[0].metadata) > 0
