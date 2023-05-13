"""Test LLM reranker."""

from pathlib import Path
import pytest
from llama_index.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.readers import Document

from llama_index.indices.query.schema import QueryBundle
from llama_index.prompts.prompts import Prompt, SimpleInputPrompt
from llama_index.indices.service_context import ServiceContext
from llama_index.data_structs.node import Node, DocumentRelationship, NodeWithScore
from llama_index.indices.postprocessor.node import (
    PrevNextNodePostprocessor,
    KeywordNodePostprocessor,
)
from llama_index.indices.postprocessor.node_recency import (
    FixedRecencyPostprocessor,
    EmbeddingRecencyPostprocessor,
    TimeWeightedPostprocessor,
)
from llama_index.llm_predictor import LLMPredictor
from unittest.mock import patch
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import List, Any, Tuple, cast, Dict
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.indices.postprocessor.llm_rerank import LLMRerank


def mock_llmpredictor_predict(
    self: Any, prompt: Prompt, **prompt_args: Any
) -> Tuple[str, str]:
    """Patch llm predictor predict."""
    assert isinstance(prompt, QuestionAnswerPrompt)
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

    result_strs = [f"Doc: {str(c)}, Relevance: {s}" for c, s in choices_and_scores]
    return "\n".join(result_strs), ""


def mock_format_node_batch_fn(nodes: List[Node]) -> str:
    """Mock format node batch fn."""
    return "\n".join([node.text for node in nodes])


@patch.object(
    LLMPredictor,
    "predict",
    mock_llmpredictor_predict,
)
def test_llm_rerank() -> None:
    """Test LLM rerank."""
    nodes = [
        Node("Test"),
        Node("Test2"),
        Node("Test3"),
        Node("Test4"),
        Node("Test5"),
        Node("Test6"),
        Node("Test7"),
        Node("Test8"),
    ]
    nodes_with_score = [NodeWithScore(n) for n in nodes]

    # choice batch size 4 (so two batches)
    # take top-3 across all data
    llm_rerank = LLMRerank(
        format_node_batch_fn=mock_format_node_batch_fn,
        choice_batch_size=4,
        top_n=3,
    )
    query_str = "What is?"
    nodes = llm_rerank.postprocess_nodes(nodes_with_score, QueryBundle(query_str))
    assert len(nodes) == 3
    assert nodes[0].node.text == "Test7"
    assert nodes[1].node.text == "Test5"
    assert nodes[2].node.text == "Test3"
