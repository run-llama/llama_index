from typing import Any, List
from unittest.mock import patch
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.list.retrievers import ListIndexEmbeddingRetriever
from gpt_index.readers.schema.base import Document
from tests.indices.list.test_index import _get_embeddings
from tests.mock_utils.mock_decorator import patch_common


@patch_common
def test_retrieve_default(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test list query."""
    index = GPTListIndex.from_documents(documents)

    query_str = "What is?"
    retriever = index.as_retriever(mode="default")
    nodes = retriever.retrieve(query_str)

    for node_with_score, line in zip(nodes, documents[0].text.split("\n")):
        assert node_with_score.node.text == line


@patch_common
@patch.object(
    ListIndexEmbeddingRetriever,
    "_get_embeddings",
    side_effect=_get_embeddings,
)
def test_embedding_query(
    _mock_similarity: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test embedding query."""
    index = GPTListIndex.from_documents(documents)

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(mode="embedding", similarity_top_k=1)
    nodes = retriever.retrieve(query_str)
    assert len(nodes) == 1

    assert nodes[0].node.text == "Hello world."
