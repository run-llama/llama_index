"""Test vector store indexes."""

from typing import Any, List, cast
from unittest.mock import patch


from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.vector_store.base import GPTVectorStoreIndex

from gpt_index.readers.schema.base import Document
from gpt_index.vector_stores.simple import SimpleVectorStore
from tests.mock_utils.mock_decorator import patch_common


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
def test_build_simple(
    _mock_embeds: Any,
    _mock_embed: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    documents: List[Document],
) -> None:
    """Test build GPTVectorStoreIndex."""

    index = GPTVectorStoreIndex.from_documents(documents=documents)
    assert isinstance(index, GPTVectorStoreIndex)
    assert len(index.index_struct.nodes_dict) == 4
    # check contents of nodes
    actual_node_tups = [
        ("Hello world.", [1, 0, 0, 0, 0]),
        ("This is a test.", [0, 1, 0, 0, 0]),
        ("This is another test.", [0, 0, 1, 0, 0]),
        ("This is a test v2.", [0, 0, 0, 1, 0]),
    ]
    for text_id in index.index_struct.nodes_dict.keys():
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
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
) -> None:
    """Test insert GPTVectorStoreIndex."""
    index = GPTVectorStoreIndex.from_documents(documents=documents)
    assert isinstance(index, GPTVectorStoreIndex)
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
    for text_id in index.index_struct.nodes_dict.keys():
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
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
) -> None:
    """Test delete GPTVectorStoreIndex."""
    new_documents = [
        Document("Hello world.", doc_id="test_id_0"),
        Document("This is a test.", doc_id="test_id_1"),
        Document("This is another test.", doc_id="test_id_2"),
        Document("This is a test v2.", doc_id="test_id_3"),
    ]
    index = GPTVectorStoreIndex.from_documents(documents=new_documents)
    assert isinstance(index, GPTVectorStoreIndex)

    # test delete
    index.delete("test_id_0")
    assert len(index.index_struct.nodes_dict) == 3
    assert len(index.index_struct.doc_id_dict) == 3
    actual_node_tups = [
        ("This is a test.", [0, 1, 0, 0, 0], "test_id_1"),
        ("This is another test.", [0, 0, 1, 0, 0], "test_id_2"),
        ("This is a test v2.", [0, 0, 0, 1, 0], "test_id_3"),
    ]
    for text_id in index.index_struct.nodes_dict.keys():
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        # NOTE: this test breaks abstraction
        assert isinstance(index._vector_store, SimpleVectorStore)
        embedding = index._vector_store.get(text_id)
        assert (node.text, embedding, node.ref_doc_id) in actual_node_tups

    # test insert
    index.insert(Document("Hello world backup.", doc_id="test_id_0"))
    assert len(index.index_struct.nodes_dict) == 4
    actual_node_tups = [
        ("Hello world backup.", [1, 0, 0, 0, 0], "test_id_0"),
        ("This is a test.", [0, 1, 0, 0, 0], "test_id_1"),
        ("This is another test.", [0, 0, 1, 0, 0], "test_id_2"),
        ("This is a test v2.", [0, 0, 0, 1, 0], "test_id_3"),
    ]
    for text_id in index.index_struct.nodes_dict.keys():
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        # NOTE: this test breaks abstraction
        assert isinstance(index._vector_store, SimpleVectorStore)
        embedding = index._vector_store.get(text_id)
        assert (node.text, embedding, node.ref_doc_id) in actual_node_tups


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
) -> None:
    """Test simple vector index with use_async."""

    index = GPTVectorStoreIndex.from_documents(documents=documents, use_async=True)
    assert isinstance(index, GPTVectorStoreIndex)
    assert len(index.index_struct.nodes_dict) == 4
    # check contents of nodes
    actual_node_tups = [
        ("Hello world.", [1, 0, 0, 0, 0]),
        ("This is a test.", [0, 1, 0, 0, 0]),
        ("This is another test.", [0, 0, 1, 0, 0]),
        ("This is a test v2.", [0, 0, 0, 1, 0]),
    ]
    for text_id in index.index_struct.nodes_dict.keys():
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        vector_store = cast(SimpleVectorStore, index._vector_store)
        embedding = vector_store.get(text_id)
        assert (node.text, embedding) in actual_node_tups
