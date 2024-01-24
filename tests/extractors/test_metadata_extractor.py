"""Test dataset generation."""

import os

from llama_index import SimpleDirectoryReader
from llama_index.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import MockLLM
from llama_index.text_splitter import TokenTextSplitter


def test_metadata_extractor() -> None:
    """Test metadata extraction."""
    llm = MockLLM()

    test_dir = os.path.dirname(os.path.abspath(__file__))
    uber_docs = SimpleDirectoryReader(
        input_files=[os.path.join(test_dir, "../test_data/uber.pdf")]
    ).load_data()
    uber_front_pages = uber_docs[0:3]
    uber_content = uber_docs[63:69]
    uber_docs = uber_front_pages + uber_content

    text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

    extractors = [
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
    ]

    transformations = [text_splitter, *extractors]

    pipeline = IngestionPipeline(transformations=transformations)

    uber_nodes = pipeline.run(documents=uber_docs)

    assert len(uber_nodes) == 21
    assert (
        uber_nodes[0].metadata["document_title"]
        != uber_nodes[-1].metadata["document_title"]
    )
