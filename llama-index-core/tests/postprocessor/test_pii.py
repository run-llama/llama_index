"""Test PII postprocessor."""

import asyncio
import json
from typing import Any, Dict

import pytest
from unittest.mock import patch

from llama_index.core.async_utils import DEFAULT_NUM_WORKERS
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor.pii import PIINodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode

PII_KEY = "__pii_node_info__"
NAMES = {"Zhang Wei": "[NAME1]", "John": "[NAME2]"}


def _fake_masked_response(text: str) -> str:
    """Build a response in the shape the postprocessor parses."""
    masked = text
    mapping: Dict[str, str] = {}
    for name, tag in NAMES.items():
        if name in masked:
            masked = masked.replace(name, tag)
            mapping[tag] = name
    return f"{masked}\nOutput Mapping:\n{json.dumps(mapping)}"


def mock_predict(self: Any, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
    """Patch llm.predict."""
    return _fake_masked_response(prompt_args["context_str"])


async def amock_predict(
    self: Any, prompt: BasePromptTemplate, **prompt_args: Any
) -> str:
    """Patch llm.apredict."""
    return mock_predict(self, prompt, **prompt_args)


def _nodes() -> list:
    return [
        NodeWithScore(node=TextNode(text="Hello Zhang Wei"), score=0.9),
        NodeWithScore(node=TextNode(text="I am John"), score=0.5),
        NodeWithScore(node=TextNode(text="nothing to mask here"), score=0.1),
    ]


def _content(node_with_score: NodeWithScore) -> str:
    return node_with_score.node.get_content(metadata_mode=MetadataMode.NONE)


@patch.object(MockLLM, "predict", mock_predict)
def test_pii_postprocessor() -> None:
    """Test PII masking, mapping metadata, and score preservation."""
    nodes = _nodes()
    pp = PIINodePostprocessor(llm=MockLLM())
    results = pp.postprocess_nodes(nodes)

    assert [_content(n) for n in results] == [
        "Hello [NAME1]",
        "I am [NAME2]",
        "nothing to mask here",
    ]
    assert results[0].node.metadata[PII_KEY] == {"[NAME1]": "Zhang Wei"}
    assert results[1].node.metadata[PII_KEY] == {"[NAME2]": "John"}
    assert results[2].node.metadata[PII_KEY] == {}
    # scores are carried through untouched
    assert [n.score for n in results] == [0.9, 0.5, 0.1]
    # the mapping must not leak back into the LLM or the embedding model
    for n in results:
        assert PII_KEY in n.node.excluded_llm_metadata_keys
        assert PII_KEY in n.node.excluded_embed_metadata_keys


@patch.object(MockLLM, "predict", mock_predict)
def test_pii_postprocessor_does_not_mutate_input() -> None:
    """The original nodes must be left untouched (the class deep-copies)."""
    nodes = _nodes()
    PIINodePostprocessor(llm=MockLLM()).postprocess_nodes(nodes)

    assert [_content(n) for n in nodes] == [
        "Hello Zhang Wei",
        "I am John",
        "nothing to mask here",
    ]
    assert all(PII_KEY not in n.node.metadata for n in nodes)


@patch.object(MockLLM, "apredict", amock_predict)
@pytest.mark.asyncio
async def test_apii_postprocessor() -> None:
    """Async masking must match the sync result exactly."""
    nodes = _nodes()
    pp = PIINodePostprocessor(llm=MockLLM())
    results = await pp.apostprocess_nodes(nodes)

    assert [_content(n) for n in results] == [
        "Hello [NAME1]",
        "I am [NAME2]",
        "nothing to mask here",
    ]
    assert results[0].node.metadata[PII_KEY] == {"[NAME1]": "Zhang Wei"}
    assert results[1].node.metadata[PII_KEY] == {"[NAME2]": "John"}
    assert [n.score for n in results] == [0.9, 0.5, 0.1]


@patch.object(MockLLM, "apredict", amock_predict)
@pytest.mark.asyncio
async def test_apii_postprocessor_empty_nodes() -> None:
    """No nodes means no LLM calls and an empty result."""
    pp = PIINodePostprocessor(llm=MockLLM())
    assert await pp.apostprocess_nodes([]) == []


@pytest.mark.asyncio
async def test_apii_postprocessor_runs_nodes_concurrently() -> None:
    """Every node should be masked in parallel, not one at a time."""
    # 3 nodes, deliberately BELOW async_utils.DEFAULT_NUM_WORKERS (4) so the
    # assertion measures this method's dispatch, not run_jobs' semaphore ceiling.
    nodes = _nodes()
    assert len(nodes) < DEFAULT_NUM_WORKERS

    in_flight = 0
    max_in_flight = 0

    async def tracking_predict(
        self: Any, prompt: BasePromptTemplate, **prompt_args: Any
    ) -> str:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            # yield control so a sequential implementation cannot overlap
            await asyncio.sleep(0)
            return mock_predict(self, prompt, **prompt_args)
        finally:
            in_flight -= 1

    pp = PIINodePostprocessor(llm=MockLLM())
    with patch.object(MockLLM, "apredict", tracking_predict):
        await pp.apostprocess_nodes(nodes)

    assert max_in_flight == len(nodes), (
        f"expected all {len(nodes)} nodes in flight concurrently, "
        f"saw at most {max_in_flight}"
    )
