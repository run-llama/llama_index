"""Test document summary retrievers."""
from typing import Any, List, Tuple
from unittest.mock import patch

from llama_index.indices.document_summary.base import DocumentSummaryIndex
from llama_index.indices.document_summary.retrievers import (
    DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.indices.query.response_synthesis import ResponseSynthesizer
from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.choice_select import ChoiceSelectPrompt
from llama_index.prompts.prompts import Prompt
from llama_index.readers.schema.base import Document
from tests.mock_utils.mock_prompts import MOCK_REFINE_PROMPT, MOCK_TEXT_QA_PROMPT


def mock_llmpredictor_predict_for_choice_select(
    self: Any, prompt: Prompt, **prompt_args: Any
) -> Tuple[str, str]:
    """Patch llm predictor predict for choice select."""
    assert isinstance(prompt, ChoiceSelectPrompt)
    # Return format: "answer_num: <int>, answer_relevance: <float>"
    # This matches the format expected by default_parse_choice_select_answer_fn
    return "answer_num: 1, answer_relevance: 0.9", ""


def _get_embeddings(
    self: Any, query_bundle: Any, nodes: List[Any]
) -> Tuple[List[float], List[List[float]]]:
    """Mock embeddings function for embedding retriever."""
    # Return query embedding
    query_embedding = [1.0, 0, 0, 0, 0]

    # Return node embeddings (one per node)
    node_embeddings = []
    for idx in range(len(nodes)):
        # Create a simple embedding that makes first node most similar
        if idx == 0:
            node_embeddings.append([1.0, 0, 0, 0, 0])
        else:
            node_embeddings.append([0, 0, 0, 0, 0])

    return query_embedding, node_embeddings


@patch.object(
    LLMPredictor,
    "predict",
    mock_llmpredictor_predict_for_choice_select,
)
def test_retrieve_default(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test retrieve default mode."""
    # Create documents with explicit doc_ids
    docs = [
        Document("This is a test v2.", doc_id="doc_1"),
        Document("This is another test.", doc_id="doc_2"),
        Document("This is a test.", doc_id="doc_3"),
        Document("Hello world.", doc_id="doc_4"),
    ]

    response_synthesizer = ResponseSynthesizer.from_args(
        text_qa_template=MOCK_TEXT_QA_PROMPT,
        refine_template=MOCK_REFINE_PROMPT,
        callback_manager=mock_service_context.callback_manager,
    )
    index = DocumentSummaryIndex.from_documents(
        docs,
        service_context=mock_service_context,
        response_synthesizer=response_synthesizer,
        summary_query="summary_query",
    )

    # Test default retriever mode
    query_str = "What is?"
    retriever = index.as_retriever(retriever_mode="default")
    nodes = retriever.retrieve(query_str)

    # Should return nodes from the selected summary
    # Mock returns "answer_num: 1, answer_relevance: 0.9" which selects first document
    assert len(nodes) > 0
    assert all(node.node is not None for node in nodes)


@patch.object(
    DocumentSummaryIndexEmbeddingRetriever,
    "_get_embeddings",
    side_effect=_get_embeddings,
)
def test_retrieve_embedding(
    _patch_get_embeddings: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test retrieve embedding mode."""
    docs = [
        Document("This is a test v2.", doc_id="doc_1"),
        Document("This is another test.", doc_id="doc_2"),
        Document("This is a test.", doc_id="doc_3"),
        Document("Hello world.", doc_id="doc_4"),
    ]

    response_synthesizer = ResponseSynthesizer.from_args(
        text_qa_template=MOCK_TEXT_QA_PROMPT,
        refine_template=MOCK_REFINE_PROMPT,
        callback_manager=mock_service_context.callback_manager,
    )
    index = DocumentSummaryIndex.from_documents(
        docs,
        service_context=mock_service_context,
        response_synthesizer=response_synthesizer,
        summary_query="summary_query",
    )

    # Test embedding retriever mode
    query_str = "What is?"
    retriever = index.as_retriever(retriever_mode="embedding", similarity_top_k=1)
    nodes = retriever.retrieve(query_str)

    # Should return nodes from the most similar summary
    assert len(nodes) > 0
    assert all(node.node is not None for node in nodes)
