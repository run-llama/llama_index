import sys
import types
import pytest
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.aimon_rerank import AIMonRerank

# --- Set up a fake "aimon" module if not already present ---
if "aimon" not in sys.modules:
    fake_aimon = types.ModuleType("aimon")

    class FakeAimonClient:
        def __init__(self, auth_header) -> None:
            self.auth_header = auth_header
            self.retrieval = self  # Simulate the retrieval attribute.

        def rerank(self, context_docs, queries, task_definition):
            # For each text, return a score equal to the word count multiplied by 10.
            return [[len(text.split()) * 10 for text in context_docs]]

    fake_aimon.Client = FakeAimonClient
    sys.modules["aimon"] = fake_aimon

# --- Unit Tests ---


def test_class_inheritance():
    # Verify that AIMonRerank inherits from BaseNodePostprocessor.
    names_of_base_classes = [b.__name__ for b in AIMonRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_class_inheritance_with_issubclass():
    # Alternatively, use issubclass for inheritance check.
    assert issubclass(AIMonRerank, BaseNodePostprocessor)


def test_init_with_api_key_provided():
    obj = AIMonRerank(api_key="testkey", top_n=2, model="test_model")
    assert obj._client.auth_header == "Bearer testkey"


def test_init_with_env_api_key(monkeypatch):
    monkeypatch.setenv("AIMON_API_KEY", "envkey")
    obj = AIMonRerank(top_n=2, model="test_model")
    assert obj._client.auth_header == "Bearer envkey"


def test_init_missing_api_key(monkeypatch):
    monkeypatch.delenv("AIMON_API_KEY", raising=False)
    with pytest.raises(KeyError):
        AIMonRerank(top_n=2, model="test_model")


## Tests for batch processing:

# Single batch

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, MetadataMode


def test_postprocess_nodes_single_batch():
    """
    Test processing nodes that all fit in one batch.
    Create three nodes with 10, 20, and 30 words respectively.
    The fake AIMon client will return a raw score equal to (word count * 10),
    which is then normalized (dividing by 100) in the processing.
    """
    nodes = [
        NodeWithScore(
            node=TextNode(text="word " * 10), score=0
        ),  # 10 words → raw score 100
        NodeWithScore(
            node=TextNode(text="word " * 20), score=0
        ),  # 20 words → raw score 200
        NodeWithScore(
            node=TextNode(text="word " * 30), score=0
        ),  # 30 words → raw score 300
    ]
    qb = QueryBundle(query_str="test query")
    obj = AIMonRerank(api_key="testkey", top_n=2)
    result = obj._postprocess_nodes(nodes, qb)

    scores = [node.score for node in result]
    assert scores == sorted(scores, reverse=True)

    highest_node_text = result[0].node.get_content(MetadataMode.EMBED).strip()
    assert highest_node_text == ("word " * 30).strip()


## Multiple batches

import llama_index.postprocessor.aimon_rerank.base as base_module


def test_postprocess_nodes_multiple_batches(monkeypatch):
    """
    Test the batching logic by lowering MAX_WORDS_PER_BATCH to force splitting.
    With MAX_WORDS_PER_BATCH set to 15:
      - A node with 10 words (raw score 100) is alone in its batch.
      - Two nodes (one with 10 words and one with 5 words) are split into another batch,
        yielding raw scores 100 and 50 (normalized 1.0 and 0.5).
    """
    original_max = base_module.MAX_WORDS_PER_BATCH
    monkeypatch.setattr(
        base_module, "MAX_WORDS_PER_BATCH", 15
    )  # Force small batch size

    try:
        nodes = [
            NodeWithScore(node=TextNode(text="word " * 10), score=0),
            NodeWithScore(node=TextNode(text="word " * 10), score=0),
            NodeWithScore(node=TextNode(text="word " * 5), score=0),
        ]
        qb = QueryBundle(query_str="test query")
        obj = AIMonRerank(api_key="testkey", top_n=3)
        result = obj._postprocess_nodes(nodes, qb)

        scores = [node.score for node in result]
        assert scores == sorted(scores, reverse=True)

        # Ensure that one of the nodes has a normalized score of 0.5 (i.e., raw score 50).
        assert 0.5 in scores
    finally:
        monkeypatch.setattr(
            base_module, "MAX_WORDS_PER_BATCH", original_max
        )  # Restore original batch size


## Testing a large batch

import llama_index.postprocessor.aimon_rerank.base as base_module


def test_postprocess_nodes_large_batch(monkeypatch):
    """
    Test processing nodes with a large batch size (greater than 10,000 words).
    """
    original_max = base_module.MAX_WORDS_PER_BATCH
    monkeypatch.setattr(
        base_module, "MAX_WORDS_PER_BATCH", 20000
    )  # Increase batch size for testing

    try:
        nodes = [
            NodeWithScore(node=TextNode(text="word " * 5000), score=0),  # 5000 words
            NodeWithScore(node=TextNode(text="word " * 6000), score=0),  # 6000 words
            NodeWithScore(node=TextNode(text="word " * 7000), score=0),  # 7000 words
        ]
        qb = QueryBundle(query_str="test query")
        obj = AIMonRerank(api_key="testkey", top_n=3)
        result = obj._postprocess_nodes(nodes, qb)

        scores = [node.score for node in result]
        assert scores == sorted(scores, reverse=True)

    finally:
        # Restore original value after test
        monkeypatch.setattr(base_module, "MAX_WORDS_PER_BATCH", original_max)
