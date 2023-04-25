"""Test vector store indexes."""

from typing import Any, List
from unittest.mock import patch

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.vector_store.base import GPTVectorStoreIndex

from gpt_index.readers.schema.base import Document
from gpt_index.storage.storage_context import StorageContext
from tests.indices.vector_store.test_simple import (
    mock_get_text_embedding,
    mock_get_text_embeddings,
)
from tests.mock_utils.mock_decorator import patch_common


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_build_faiss(
    _mock_embeds: Any,
    _mock_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    faiss_storage_context: StorageContext,
) -> None:
    """Test build GPTFaissIndex."""
    index = GPTVectorStoreIndex.from_documents(
        documents=documents, storage_context=faiss_storage_context
    )
    assert len(index.index_struct.nodes_dict) == 4

    node_ids = list(index.index_struct.nodes_dict.values())
    nodes = index.docstore.get_nodes(node_ids)
    node_texts = [node.text for node in nodes]
    assert "Hello world." in node_texts
    assert "This is a test." in node_texts
    assert "This is another test." in node_texts
    assert "This is a test v2." in node_texts


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_faiss_insert(
    _mock_embeds: Any,
    _mock_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    faiss_storage_context: StorageContext,
) -> None:
    """Test build GPTFaissIndex."""
    index = GPTVectorStoreIndex.from_documents(
        documents=documents, storage_context=faiss_storage_context
    )

    node_ids = index.index_struct.nodes_dict
    print(node_ids)

    # insert into index
    index.insert(Document(text="This is a test v3."))

    # check contents of nodes
    node_ids = index.index_struct.nodes_dict
    print(node_ids)

    node_ids = list(index.index_struct.nodes_dict.values())
    nodes = index.docstore.get_nodes(node_ids)
    print("nodes")
    print(nodes)
    node_texts = [node.text for node in nodes]
    assert "This is a test v2." in node_texts
    assert "This is a test v3." in node_texts
