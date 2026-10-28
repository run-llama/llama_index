from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from llama_index.core.extractors import (
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import run_transformations
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    BaseNode,
    Document,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
    TransformComponent,
)


def test_metadata_extractor() -> None:
    extractors: List[TransformComponent] = [
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=3),
        SummaryExtractor(summaries=["prev", "self"]),
        KeywordExtractor(keywords=10),
    ]

    node_parser: TransformComponent = SentenceSplitter()

    document = Document(
        text="sample text",
        metadata={"filename": "README.md", "category": "codebase"},
    )

    nodes = run_transformations([document], [node_parser, *extractors])

    assert "document_title" in nodes[0].metadata
    assert "questions_this_excerpt_can_answer" in nodes[0].metadata
    assert "section_summary" in nodes[0].metadata
    assert "excerpt_keywords" in nodes[0].metadata


@pytest.mark.asyncio
async def test_title_extractor_standalone_nodes_not_grouped_under_none() -> None:
    """Standalone TextNodes with ref_doc_id=None should each get their own title."""
    extractor = TitleExtractor(nodes=1, llm=MockLLM(), show_progress=False)
    with patch.object(
        MockLLM,
        "apredict",
        new=AsyncMock(side_effect=lambda prompt, **kwargs: f"title:{kwargs.get('context_str', '')}"),
    ):
        node_a = TextNode(text="alpha content", id_="node-a")
        node_b = TextNode(text="beta content", id_="node-b")
        assert node_a.ref_doc_id is None
        assert node_b.ref_doc_id is None

        metadata_list = await extractor.aextract([node_a, node_b])

    assert len(metadata_list) == 2
    assert metadata_list[0]["document_title"] != metadata_list[1]["document_title"]
    assert "alpha content" in metadata_list[0]["document_title"]
    assert "beta content" in metadata_list[1]["document_title"]


@pytest.mark.asyncio
async def test_title_extractor_groups_nodes_sharing_ref_doc_id() -> None:
    """Nodes belonging to the same document (ref_doc_id) share a combined title."""
    extractor = TitleExtractor(nodes=2, llm=MockLLM(), show_progress=False)
    with patch.object(
        MockLLM,
        "apredict",
        new=AsyncMock(side_effect=lambda prompt, **kwargs: f"title({kwargs.get('context_str', '')})"),
    ):
        n1 = TextNode(
            text="part1",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1")},
        )
        n2 = TextNode(
            text="part2",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1")},
        )
        metadata_list = await extractor.aextract([n1, n2])

    assert len(metadata_list) == 2
    assert metadata_list[0]["document_title"] == metadata_list[1]["document_title"]
    assert "part1" in metadata_list[0]["document_title"]
    assert "part2" in metadata_list[0]["document_title"]


class _NonTextNode(BaseNode):
    @classmethod
    def class_name(cls) -> str:
        return "NonTextNode"

    @classmethod
    def get_type(cls) -> str:
        return "non_text"

    def get_content(self, metadata_mode=None) -> str:
        return "non-text content"

    def get_content_blocks(self, metadata_mode=None) -> list:
        return []

    def set_content(self, value: str) -> None:
        pass

    @property
    def hash(self) -> str:
        return "hash"


@pytest.mark.asyncio
async def test_title_extractor_skips_non_text_nodes_when_is_text_node_only() -> None:
    """Non-TextNodes should return empty metadata dict when is_text_node_only=True."""
    extractor = TitleExtractor(nodes=2, llm=MockLLM(), is_text_node_only=True, show_progress=False)
    text_node = TextNode(text="hello world", id_="t1")
    non_text_node = _NonTextNode(id_="nt1")

    metadata_list = await extractor.aextract([text_node, non_text_node])
    assert len(metadata_list) == 2
    assert "document_title" in metadata_list[0]
    assert metadata_list[1] == {}


@pytest.mark.asyncio
async def test_title_extractor_forwards_num_workers() -> None:
    """extract_titles and get_title_candidates should forward self.num_workers to run_jobs."""
    extractor = TitleExtractor(nodes=2, llm=MockLLM(), num_workers=7, show_progress=False)
    n1 = TextNode(
        text="chunk 1",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1")},
    )
    n2 = TextNode(
        text="chunk 2",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1")},
    )

    observed_workers: List[int] = []
    from llama_index.core.extractors import metadata_extractors as me_mod

    orig_run_jobs = me_mod.run_jobs

    async def spy_run_jobs(jobs, show_progress=False, workers=4, desc=None):
        observed_workers.append(workers)
        return await orig_run_jobs(jobs, show_progress=show_progress, workers=workers, desc=desc)

    with patch.object(me_mod, "run_jobs", side_effect=spy_run_jobs):
        await extractor.aextract([n1, n2])

    assert observed_workers == [7, 7]

