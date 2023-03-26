"""Test Faiss index."""

import sys
from typing import Any, Dict, List, Tuple, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.vector_store.vector_indices import (
    GPTFaissIndex,
    GPTSimpleVectorIndex,
)
from gpt_index.readers.schema.base import Document
from gpt_index.vector_stores.simple import SimpleVectorStore
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_prompts import MOCK_REFINE_PROMPT, MOCK_TEXT_QA_PROMPT


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    index_kwargs = {
        "text_qa_template": MOCK_TEXT_QA_PROMPT,
    }
    query_kwargs = {
        "text_qa_template": MOCK_TEXT_QA_PROMPT,
        "refine_template": MOCK_REFINE_PROMPT,
        "similarity_top_k": 1,
    }
    return index_kwargs, query_kwargs


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text)]


class MockFaissIndex:
    """Mock Faiss index."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize params."""
        self._index: Dict[int, np.ndarray] = {}

    @property
    def ntotal(self) -> int:
        """Get ntotal."""
        return len(self._index)

    def add(self, vecs: np.ndarray) -> None:
        """Add vectors to index."""
        for vec in vecs:
            new_id = len(self._index)
            self._index[new_id] = vec

    def reset(self) -> None:
        """Reset index."""
        self._index = {}

    def search(self, vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search index."""
        # assume query vec is of the form 1 x k
        # index_mat is n x k
        index_mat = np.array(list(self._index.values()))
        # compute distances
        distances = np.linalg.norm(index_mat - vec, axis=1)

        indices = np.argsort(distances)[:k]
        sorted_distances = distances[indices][:k]

        # return distances and indices
        return sorted_distances[np.newaxis, :], indices[np.newaxis, :]


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "Hello world.":
        return [1, 0, 0, 0, 0]
    elif text == "This is a test.":
        return [0, 1, 0, 0, 0]
    elif text == "This is another test.":
        return [0, 0, 1, 0, 0]
    elif text == "This is a test v2.":
        return [0, 0, 0, 1, 0]
    elif text == "This is a test v3.":
        return [0, 0, 0, 0, 1]
    elif text == "This is bar test.":
        return [0, 0, 1, 0, 0]
    elif text == "Hello world backup.":
        # this is used when "Hello world." is deleted.
        return [1, 0, 0, 0, 0]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding`.")


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding(text) for text in texts]


async def mock_aget_text_embedding(text: str) -> List[float]:
    """Mock async get text embedding."""
    # assume dimensions are 5
    if text == "Hello world.":
        return [1, 0, 0, 0, 0]
    elif text == "This is a test.":
        return [0, 1, 0, 0, 0]
    elif text == "This is another test.":
        return [0, 0, 1, 0, 0]
    elif text == "This is a test v2.":
        return [0, 0, 0, 1, 0]
    elif text == "This is a test v3.":
        return [0, 0, 0, 0, 1]
    elif text == "This is bar test.":
        return [0, 0, 1, 0, 0]
    elif text == "Hello world backup.":
        # this is used when "Hello world." is deleted.
        return [1, 0, 0, 0, 0]
    else:
        raise ValueError("Invalid text for `mock_aget_text_embedding`.")


async def mock_aget_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock async get text embeddings."""
    return [await mock_aget_text_embedding(text) for text in texts]


def mock_get_query_embedding(query: str) -> List[float]:
    """Mock get query embedding."""
    return [0, 0, 1, 0, 0]


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
    struct_kwargs: Dict,
) -> None:
    """Test build GPTFaissIndex."""
    # NOTE: mock faiss import
    sys.modules["faiss"] = MagicMock()
    # NOTE: mock faiss index
    faiss_index = MockFaissIndex()

    index_kwargs, query_kwargs = struct_kwargs

    index = GPTFaissIndex(documents=documents, faiss_index=faiss_index, **index_kwargs)
    assert len(index.index_struct.nodes_dict) == 4
    # check contents of nodes
    assert index.index_struct.get_node("0").text == "Hello world."
    assert index.index_struct.get_node("1").text == "This is a test."
    assert index.index_struct.get_node("2").text == "This is another test."
    assert index.index_struct.get_node("3").text == "This is a test v2."


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
    struct_kwargs: Dict,
) -> None:
    """Test build GPTFaissIndex."""
    # NOTE: mock faiss import
    sys.modules["faiss"] = MagicMock()
    # NOTE: mock faiss index
    faiss_index = MockFaissIndex()

    index_kwargs, query_kwargs = struct_kwargs

    index = GPTFaissIndex(documents=documents, faiss_index=faiss_index, **index_kwargs)
    # insert into index
    index.insert(Document(text="This is a test v3."))

    # check contenst of nodes
    assert index.index_struct.get_node("3").text == "This is a test v2."
    assert index.index_struct.get_node("4").text == "This is a test v3."


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_faiss_query(
    _mock_query_embed: Any,
    _mock_texts_embed: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    # NOTE: mock faiss import
    sys.modules["faiss"] = MagicMock()
    # NOTE: mock faiss index
    faiss_index = MockFaissIndex()

    index_kwargs, query_kwargs = struct_kwargs
    index = GPTFaissIndex(documents=documents, faiss_index=faiss_index, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    response = index.query(query_str, **query_kwargs)
    assert str(response) == ("What is?:This is another test.")


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_build_simple(
    _mock_embeds: Any,
    _mock_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test build GPTSimpleVectorIndex."""
    index_kwargs, query_kwargs = struct_kwargs

    index = GPTSimpleVectorIndex(documents=documents, **index_kwargs)
    assert len(index.index_struct.nodes_dict) == 4
    # check contents of nodes
    actual_node_tups = [
        ("Hello world.", [1, 0, 0, 0, 0]),
        ("This is a test.", [0, 1, 0, 0, 0]),
        ("This is another test.", [0, 0, 1, 0, 0]),
        ("This is a test v2.", [0, 0, 0, 1, 0]),
    ]
    for text_id in index.index_struct.id_map.keys():
        node = index.index_struct.get_node(text_id)
        # NOTE: this test breaks abstraction
        assert isinstance(index._vector_store, SimpleVectorStore)
        embedding = index._vector_store.get(text_id)
        assert (node.text, embedding) in actual_node_tups


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_simple_insert(
    _mock_embeds: Any,
    _mock_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test insert GPTSimpleVectorIndex."""
    index_kwargs, query_kwargs = struct_kwargs

    index = GPTSimpleVectorIndex(documents=documents, **index_kwargs)
    # insert into index
    index.insert(Document(text="This is a test v3."))

    # check contenst of nodes
    actual_node_tups = [
        ("Hello world.", [1, 0, 0, 0, 0]),
        ("This is a test.", [0, 1, 0, 0, 0]),
        ("This is another test.", [0, 0, 1, 0, 0]),
        ("This is a test v2.", [0, 0, 0, 1, 0]),
        ("This is a test v3.", [0, 0, 0, 0, 1]),
    ]
    for text_id in index.index_struct.id_map.keys():
        node = index.index_struct.get_node(text_id)
        # NOTE: this test breaks abstraction
        assert isinstance(index._vector_store, SimpleVectorStore)
        embedding = index._vector_store.get(text_id)
        assert (node.text, embedding) in actual_node_tups


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_simple_delete(
    _mock_embeds: Any,
    _mock_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_splitter_overlap: Any,
    _mock_splitter: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test delete GPTSimpleVectorIndex."""
    index_kwargs, query_kwargs = struct_kwargs

    new_documents = [
        Document("Hello world.", doc_id="test_id_0"),
        Document("This is a test.", doc_id="test_id_1"),
        Document("This is another test.", doc_id="test_id_2"),
        Document("This is a test v2.", doc_id="test_id_3"),
    ]
    index = GPTSimpleVectorIndex(documents=new_documents, **index_kwargs)

    # test delete
    index.delete("test_id_0")
    assert len(index.index_struct.nodes_dict) == 3
    assert len(index.index_struct.id_map) == 3
    actual_node_tups = [
        ("This is a test.", [0, 1, 0, 0, 0], "test_id_1"),
        ("This is another test.", [0, 0, 1, 0, 0], "test_id_2"),
        ("This is a test v2.", [0, 0, 0, 1, 0], "test_id_3"),
    ]
    for text_id in index.index_struct.id_map.keys():
        node = index.index_struct.get_node(text_id)
        # NOTE: this test breaks abstraction
        assert isinstance(index._vector_store, SimpleVectorStore)
        embedding = index._vector_store.get(text_id)
        assert (node.text, embedding, node.ref_doc_id) in actual_node_tups

    # test insert
    index.insert(Document("Hello world backup.", doc_id="test_id_0"))
    assert len(index.index_struct.nodes_dict) == 4
    assert len(index.index_struct.id_map) == 4
    actual_node_tups = [
        ("Hello world backup.", [1, 0, 0, 0, 0], "test_id_0"),
        ("This is a test.", [0, 1, 0, 0, 0], "test_id_1"),
        ("This is another test.", [0, 0, 1, 0, 0], "test_id_2"),
        ("This is a test v2.", [0, 0, 0, 1, 0], "test_id_3"),
    ]
    for text_id in index.index_struct.id_map.keys():
        node = index.index_struct.get_node(text_id)
        # NOTE: this test breaks abstraction
        assert isinstance(index._vector_store, SimpleVectorStore)
        embedding = index._vector_store.get(text_id)
        assert (node.text, embedding, node.ref_doc_id) in actual_node_tups


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_simple_query(
    _mock_query_embed: Any,
    _mock_text_embeds: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTSimpleVectorIndex(documents, **index_kwargs)

    # test embedding query
    query_str = "What is?"
    response = index.query(query_str, **query_kwargs)
    assert str(response) == ("What is?:This is another test.")

    # test with keyword filter (required)
    query_kwargs_copy = query_kwargs.copy()
    query_kwargs_copy["similarity_top_k"] = 5
    response = index.query(query_str, **query_kwargs_copy, required_keywords=["Hello"])
    assert str(response) == ("What is?:Hello world.")

    # test with keyword filter (exclude)
    # insert into index
    index.insert(Document(text="This is bar test."))
    query_kwargs_copy = query_kwargs.copy()
    query_kwargs_copy["similarity_top_k"] = 2
    response = index.query(query_str, **query_kwargs_copy, exclude_keywords=["another"])
    assert str(response) == ("What is?:This is bar test.")


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "_get_query_embedding", side_effect=mock_get_query_embedding
)
def test_query_and_count_tokens(
    _mock_query_embed: Any,
    _mock_text_embeds: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Dict,
) -> None:
    """Test embedding query."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTSimpleVectorIndex([document], **index_kwargs)
    assert index.embed_model.total_tokens_used == 20

    # test embedding query
    query_str = "What is?"
    index.query(query_str, **query_kwargs)
    assert index.embed_model.last_token_usage == 3


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "_get_query_embedding", side_effect=mock_get_query_embedding
)
def test_query_and_similarity_scores(
    _mock_query_embed: Any,
    _mock_text_embeds: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Dict,
) -> None:
    """Test that sources nodes have similarity scores."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTSimpleVectorIndex([document], **index_kwargs)

    # test embedding query
    query_str = "What is?"
    response = index.query(query_str, **query_kwargs)
    assert len(response.source_nodes) > 0
    assert response.source_nodes[0].similarity is not None


@patch_common
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "_get_query_embedding", side_effect=mock_get_query_embedding
)
def test_query_and_similarity_scores_with_cutoff(
    _mock_query_embed: Any,
    _mock_text_embeds: Any,
    _mock_text_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Dict,
) -> None:
    """Test that sources nodes have similarity scores."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    index_kwargs, query_kwargs = struct_kwargs
    index = GPTSimpleVectorIndex([document], **index_kwargs)

    # test embedding query - no nodes
    query_str = "What is?"
    response = index.query(query_str, similarity_cutoff=1.1, **query_kwargs)
    assert len(response.source_nodes) == 0

    # test embedding query - 1 node
    query_str = "What is?"
    response = index.query(query_str, similarity_cutoff=0.9, **query_kwargs)
    assert len(response.source_nodes) == 1


@patch_common
@patch.object(
    OpenAIEmbedding, "_aget_text_embedding", side_effect=mock_aget_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_aget_text_embeddings", side_effect=mock_aget_text_embeddings
)
def test_simple_async(
    _mock_embeds: Any,
    _mock_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
    struct_kwargs: Dict,
) -> None:
    """Test simple vector index with use_async."""
    index_kwargs, query_kwargs = struct_kwargs

    index = GPTSimpleVectorIndex(documents=documents, use_async=True, **index_kwargs)
    assert len(index.index_struct.nodes_dict) == 4
    # check contents of nodes
    actual_node_tups = [
        ("Hello world.", [1, 0, 0, 0, 0]),
        ("This is a test.", [0, 1, 0, 0, 0]),
        ("This is another test.", [0, 0, 1, 0, 0]),
        ("This is a test v2.", [0, 0, 0, 1, 0]),
    ]
    for text_id in index.index_struct.id_map.keys():
        node = index.index_struct.get_node(text_id)
        vector_store = cast(SimpleVectorStore, index._vector_store)
        embedding = vector_store.get(text_id)
        assert (node.text, embedding) in actual_node_tups
