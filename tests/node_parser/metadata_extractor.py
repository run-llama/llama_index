from typing import List

from llama_index.ingestion import run_transformations
from llama_index.node_parser import SimpleNodeParser
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.indices.service_context import ServiceContext
from llama_index.schema import Document, TransformComponent


def test_metadata_extractor(mock_service_context: ServiceContext) -> None:
    extractors: List[TransformComponent] = [
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=3),
        SummaryExtractor(summaries=["prev", "self"]),
        KeywordExtractor(keywords=10),
    ]

    node_parser: TransformComponent = SimpleNodeParser.from_defaults()

    document = Document(
        text="sample text",
        metadata={"filename": "README.md", "category": "codebase"},
    )

    nodes = run_transformations([document], [node_parser, *extractors])

    assert "document_title" in nodes[0].metadata
    assert "questions_this_excerpt_can_answer" in nodes[0].metadata
    assert "section_summary" in nodes[0].metadata
    assert "excerpt_keywords" in nodes[0].metadata
