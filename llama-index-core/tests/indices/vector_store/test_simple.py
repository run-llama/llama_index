"""Test vector store indexes."""

import pickle
from typing import Any, List, cast

from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.simple import SimpleVectorStore


def test_build_simple(
    patch_llm_predictor,
    patch_token_text_splitter,
    mock_embed_model,
    documents: List[Document],
) -> None:
    """Test build VectorStoreIndex."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )
    assert isinstance(index, VectorStoreIndex)
    assert len(index.index_struct.nodes_dict) == 4
    # check contents of nodes
    actual_node_tups = [
        ("Hello world.", [1, 0, 0, 0, 0]),
        ("This is a test.", [0, 1, 0, 0, 0]),
        ("This is another test.", [0, 0, 1, 0, 0]),
        ("This is a test v2.", [0, 0, 0, 1, 0]),
    ]
    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        # NOTE: this test breaks abstraction
        assert isinstance(index._vector_store, SimpleVectorStore)
        embedding = index._vector_store.get(text_id)
        assert (node.get_content(), embedding) in actual_node_tups

    # test ref doc info
    all_ref_doc_info = index.ref_doc_info
    for idx, ref_doc_id in enumerate(all_ref_doc_info.keys()):
        assert documents[idx].node_id == ref_doc_id


def test_simple_insert(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    mock_embed_model,
) -> None:
    """Test insert VectorStoreIndex."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )
    assert isinstance(index, VectorStoreIndex)
    # insert into index
    index.insert(Document(text="This is a test v3."))

    # insert empty document to test empty document handling
    index.insert(Document(text=""))

    # check contenst of nodes
    actual_node_tups = [
        ("Hello world.", [1, 0, 0, 0, 0]),
        ("This is a test.", [0, 1, 0, 0, 0]),
        ("This is another test.", [0, 0, 1, 0, 0]),
        ("This is a test v2.", [0, 0, 0, 1, 0]),
        ("This is a test v3.", [0, 0, 0, 0, 1]),
    ]
    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        # NOTE: this test breaks abstraction
        assert isinstance(index._vector_store, SimpleVectorStore)
        embedding = index._vector_store.get(text_id)
        assert (node.get_content(), embedding) in actual_node_tups


def test_simple_delete(
    patch_llm_predictor, patch_token_text_splitter, mock_embed_model
) -> None:
    """Test delete VectorStoreIndex."""
    new_documents = [
        Document(text="Hello world.", id_="test_id_0"),
        Document(text="This is a test.", id_="test_id_1"),
        Document(text="This is another test.", id_="test_id_2"),
        Document(text="This is a test v2.", id_="test_id_3"),
    ]
    index = VectorStoreIndex.from_documents(
        documents=new_documents, embed_model=mock_embed_model
    )
    assert isinstance(index, VectorStoreIndex)

    # test delete
    index.delete_ref_doc("test_id_0")
    assert len(index.index_struct.nodes_dict) == 3
    actual_node_tups = [
        ("This is a test.", [0, 1, 0, 0, 0], "test_id_1"),
        ("This is another test.", [0, 0, 1, 0, 0], "test_id_2"),
        ("This is a test v2.", [0, 0, 0, 1, 0], "test_id_3"),
    ]
    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        # NOTE: this test breaks abstraction
        assert isinstance(index._vector_store, SimpleVectorStore)
        embedding = index._vector_store.get(text_id)
        assert (node.get_content(), embedding, node.ref_doc_id) in actual_node_tups

    # test insert
    index.insert(Document(text="Hello world backup.", id_="test_id_0"))
    assert len(index.index_struct.nodes_dict) == 4
    actual_node_tups = [
        ("Hello world backup.", [1, 0, 0, 0, 0], "test_id_0"),
        ("This is a test.", [0, 1, 0, 0, 0], "test_id_1"),
        ("This is another test.", [0, 0, 1, 0, 0], "test_id_2"),
        ("This is a test v2.", [0, 0, 0, 1, 0], "test_id_3"),
    ]
    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        # NOTE: this test breaks abstraction
        assert isinstance(index._vector_store, SimpleVectorStore)
        embedding = index._vector_store.get(text_id)
        assert (node.get_content(), embedding, node.ref_doc_id) in actual_node_tups


def test_simple_delete_ref_node_from_docstore(
    patch_llm_predictor, patch_token_text_splitter, mock_embed_model
) -> None:
    """Test delete VectorStoreIndex."""
    new_documents = [
        Document(text="This is a test.", id_="test_id_1"),
        Document(text="This is another test.", id_="test_id_2"),
    ]
    index = VectorStoreIndex.from_documents(
        documents=new_documents, embed_model=mock_embed_model
    )
    assert isinstance(index, VectorStoreIndex)

    docstore = index.docstore.get_ref_doc_info("test_id_1")

    assert docstore is not None

    # test delete
    index.delete_ref_doc("test_id_1", delete_from_docstore=True)

    docstore = index.docstore.get_ref_doc_info("test_id_1")

    assert docstore is None


def test_simple_async(
    allow_networking: Any,
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    mock_embed_model,
) -> None:
    """Test simple vector index with use_async."""
    index = VectorStoreIndex.from_documents(
        documents=documents, use_async=True, embed_model=mock_embed_model
    )
    assert isinstance(index, VectorStoreIndex)
    assert len(index.index_struct.nodes_dict) == 4
    # check contents of nodes
    actual_node_tups = [
        ("Hello world.", [1, 0, 0, 0, 0]),
        ("This is a test.", [0, 1, 0, 0, 0]),
        ("This is another test.", [0, 0, 1, 0, 0]),
        ("This is a test v2.", [0, 0, 0, 1, 0]),
    ]
    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        vector_store = cast(SimpleVectorStore, index._vector_store)
        embedding = vector_store.get(text_id)
        assert (node.get_content(), embedding) in actual_node_tups


def test_async_from_documents(mock_embed_model):
    documents = [Document(text=hex(i)[2:]) for i in range(16)]
    index = VectorStoreIndex.from_documents(
        documents=documents,
        use_async=True,
        embed_model=mock_embed_model,
        insert_batch_size=2,
        async_concurrency=2,
    )
    assert len(index.index_struct.nodes_dict) == 16
    # TODO assert embeds, check concurrency


def test_simple_insert_save(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    mock_embed_model,
) -> None:
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=mock_embed_model,
        storage_context=storage_context,
    )
    assert isinstance(index, VectorStoreIndex)

    loaded_index = load_index_from_storage(storage_context=storage_context)
    assert isinstance(loaded_index, VectorStoreIndex)
    assert index.index_struct == loaded_index.index_struct

    # insert into index
    index.insert(Document(text="This is a test v3."))

    loaded_index = load_index_from_storage(storage_context=storage_context)
    assert isinstance(loaded_index, VectorStoreIndex)
    assert index.index_struct == loaded_index.index_struct


def test_simple_pickle(
    patch_llm_predictor,
    patch_token_text_splitter,
    mock_embed_model,
    documents: List[Document],
) -> None:
    """Test build VectorStoreIndex."""
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )

    data = pickle.dumps(index)
    new_index = pickle.loads(data)

    assert isinstance(new_index, VectorStoreIndex)
    assert len(new_index.index_struct.nodes_dict) == 4
    # check contents of nodes
    actual_node_tups = [
        ("Hello world.", [1, 0, 0, 0, 0]),
        ("This is a test.", [0, 1, 0, 0, 0]),
        ("This is another test.", [0, 0, 1, 0, 0]),
        ("This is a test v2.", [0, 0, 0, 1, 0]),
    ]
    for text_id in new_index.index_struct.nodes_dict:
        node_id = new_index.index_struct.nodes_dict[text_id]
        node = new_index.docstore.get_node(node_id)
        # NOTE: this test breaks abstraction
        assert isinstance(new_index._vector_store, SimpleVectorStore)
        embedding = new_index._vector_store.get(text_id)
        assert (node.get_content(), embedding) in actual_node_tups

    # test ref doc info
    all_ref_doc_info = new_index.ref_doc_info
    for idx, ref_doc_id in enumerate(all_ref_doc_info.keys()):
        assert documents[idx].node_id == ref_doc_id
