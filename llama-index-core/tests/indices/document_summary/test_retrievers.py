"""Test document summary retrievers."""

from llama_index.core.indices.document_summary.base import (
    DocumentSummaryIndex,
    DocumentSummaryRetrieverMode,
)
from llama_index.core.indices.document_summary.retrievers import (
    DocumentSummaryIndexEmbeddingRetriever,
    DocumentSummaryIndexLLMRetriever,
)


def test_embedding_retriever(index: DocumentSummaryIndex) -> None:
    retriever = index.as_retriever()
    assert isinstance(retriever, DocumentSummaryIndexEmbeddingRetriever)
    results = retriever.retrieve("Test query")
    assert len(results) == 1
    assert results[0].node.ref_doc_id == "doc_4"

    retriever = index.as_retriever(similarity_top_k=2)
    assert isinstance(retriever, DocumentSummaryIndexEmbeddingRetriever)
    results = retriever.retrieve("Test query")
    assert len(results) == 2
    assert results[0].node.ref_doc_id == "doc_3"
    assert results[1].node.ref_doc_id == "doc_4"


def test_llm_retriever(
    index: DocumentSummaryIndex,
) -> None:
    retriever = index.as_retriever(retriever_mode=DocumentSummaryRetrieverMode.LLM)
    assert isinstance(retriever, DocumentSummaryIndexLLMRetriever)
    results = retriever.retrieve("Test query")
    assert len(results) == 1
