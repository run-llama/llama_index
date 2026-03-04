"""Test LLM reranker."""

from typing import Any, List
from unittest.mock import patch
import pytest

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor.structured_llm_rerank import (
    StructuredLLMRerank,
    DocumentWithRelevance,
    DocumentRelevanceList,
)
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode


def mock_llmpredictor_structured_predict(
    self: Any, prompt: BasePromptTemplate, **prompt_args: Any
) -> DocumentRelevanceList:
    """Patch llm predictor predict."""
    context_str = prompt_args["context_str"]
    node_strs = context_str.split("\n")
    node_to_choice_and_score = {
        "Test": (True, "1"),
        "Test2": (False, "0"),
        "Test3": (True, "3"),
        "Test4": (False, "0"),
        "Test5": (True, "5"),
        "Test6": (False, "0"),
        "Test7": (True, "7"),
        "Test8": (False, "0"),
    }
    choices_and_scores = []
    for idx, node_str in enumerate(node_strs):
        choice, score = node_to_choice_and_score[node_str]
        if choice:
            choices_and_scores.append((idx + 1, score))

    doc_with_relvance = [
        DocumentWithRelevance(document_number=c, relevance=int(s))
        for c, s in choices_and_scores
    ]
    return DocumentRelevanceList(documents=doc_with_relvance)


def mock_format_node_batch_fn(nodes: List[BaseNode]) -> str:
    """Mock format node batch fn."""
    return "\n".join([node.get_content() for node in nodes])


class MockFunctionCallingLLM(MockLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return super().metadata.model_copy(update={"is_function_calling_model": True})


@patch.object(
    MockFunctionCallingLLM,
    "structured_predict",
    mock_llmpredictor_structured_predict,
)
def test_llm_rerank() -> None:
    """Test LLM rerank."""
    nodes = [
        TextNode(text="Test"),
        TextNode(text="Test2"),
        TextNode(text="Test3"),
        TextNode(text="Test4"),
        TextNode(text="Test5"),
        TextNode(text="Test6"),
        TextNode(text="Test7"),
        TextNode(text="Test8"),
    ]
    nodes_with_score = [NodeWithScore(node=n) for n in nodes]

    # choice batch size 4 (so two batches)
    # take top-3 across all data
    llm = MockFunctionCallingLLM()
    llm.metadata.is_function_calling_model = True
    llm_rerank = StructuredLLMRerank(
        llm=llm,
        format_node_batch_fn=mock_format_node_batch_fn,
        choice_batch_size=4,
        top_n=3,
    )
    query_str = "What is?"
    result_nodes = llm_rerank.postprocess_nodes(
        nodes_with_score, QueryBundle(query_str)
    )
    assert len(result_nodes) == 3
    assert result_nodes[0].node.get_content() == "Test7"
    assert result_nodes[1].node.get_content() == "Test5"
    assert result_nodes[2].node.get_content() == "Test3"


def mock_errored_structured_predict(
    self: Any, prompt: BasePromptTemplate, **prompt_args: Any
) -> str:
    return "fake error"


@patch.object(
    MockFunctionCallingLLM,
    "structured_predict",
    mock_errored_structured_predict,
)
@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_llm_rerank_errored_structured_predict(raise_on_failure: bool) -> None:
    """Test LLM rerank with errored structured predict."""
    nodes = [
        TextNode(text="Test"),
        TextNode(text="Test2"),
        TextNode(text="Test3"),
        TextNode(text="Test4"),
    ]
    nodes_with_score = [NodeWithScore(node=n) for n in nodes]

    llm = MockFunctionCallingLLM()
    llm.metadata.is_function_calling_model = True
    top_n = 3
    llm_rerank = StructuredLLMRerank(
        llm=llm,
        format_node_batch_fn=mock_format_node_batch_fn,
        choice_batch_size=4,
        top_n=top_n,
        raise_on_structured_prediction_failure=raise_on_failure,  # Set to False to test logging behavior
    )
    query_str = "What is?"
    if raise_on_failure:
        with pytest.raises(ValueError, match="Structured prediction failed for nodes"):
            llm_rerank.postprocess_nodes(nodes_with_score, QueryBundle(query_str))
    else:
        result_nodes = llm_rerank.postprocess_nodes(
            nodes_with_score, QueryBundle(query_str)
        )
        assert len(result_nodes) == top_n
        assert all(n.score == 0 for n in result_nodes)
