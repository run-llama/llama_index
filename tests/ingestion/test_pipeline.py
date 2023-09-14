from llama_index.token_counter.mock_embed_model import MockEmbedding
from llama_index.ingestion.pipeline import IngestionPipeline
from llama_index.llms import MockLLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import KeywordExtractor
from llama_index.readers import StringIterableReader, ReaderConfig
from llama_index.schema import Document


def test_build_pipeline() -> None:
    pipeline = IngestionPipeline(
        name="Test",
        reader=ReaderConfig(
            loader=StringIterableReader(), loader_kwargs={"texts": ["This is a test."]}
        ),
        documents=[Document.example()],
        transformations=[
            SimpleNodeParser.from_defaults(),
            KeywordExtractor(llm=MockLLM()),
        ],
        llm=MockLLM(),
        embed_model=MockEmbedding(embed_dim=10),
    )

    assert len(pipeline.transformations) == 2
    assert len(pipeline.configured_transformations) == 2
    assert pipeline.name == "Test"


def test_run_local_pipeline() -> None:
    pipeline = IngestionPipeline(
        name="Test",
        reader=ReaderConfig(
            loader=StringIterableReader(), loader_kwargs={"texts": ["This is a test."]}
        ),
        documents=[Document.example()],
        transformations=[
            SimpleNodeParser.from_defaults(),
            KeywordExtractor(llm=MockLLM()),
        ],
        llm=MockLLM(),
        embed_model=MockEmbedding(embed_dim=10),
    )

    nodes = pipeline.run_local()

    assert len(nodes) == 2
    assert nodes[0].embedding is not None and len(nodes[0].embedding) == 10
