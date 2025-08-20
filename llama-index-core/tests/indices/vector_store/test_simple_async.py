import pytest
from typing import List
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.vector_stores.simple import SimpleVectorStore


@pytest.mark.asyncio
async def test_simple_insertion(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    mock_embed_model,
):
    index = VectorStoreIndex.from_documents(
        documents=documents, embed_model=mock_embed_model
    )
    assert isinstance(index, VectorStoreIndex)
    # insert into index
    await index.ainsert(Document(text="This is a test v3."))

    # insert empty document to test empty document handling
    await index.ainsert(Document(text=""))

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


@pytest.mark.asyncio
async def test_simple_deletion(
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

    await index.adelete_ref_doc("test_id_0")
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
    await index.ainsert(Document(text="Hello world backup.", id_="test_id_0"))

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


@pytest.mark.asyncio
async def test_simple_update(
    patch_llm_predictor,
    patch_token_text_splitter,
    mock_embed_model,
):
    new_docs = [
        Document(id_="1", text="Hello World"),
        Document(id_="2", text="This is a test"),
    ]
    index = VectorStoreIndex.from_documents(
        documents=new_docs, embed_model=mock_embed_model
    )
    assert isinstance(index, VectorStoreIndex)
    actual_node_tups = [
        ("Hello World v1", "1"),
        ("This is a test", "2"),
    ]
    await index.aupdate_ref_doc(Document(id_="1", text="Hello World v1"))
    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        # NOTE: this test breaks abstraction
        assert (node.get_content(), node.ref_doc_id) in actual_node_tups


@pytest.mark.asyncio
async def test_simple_refresh(
    patch_llm_predictor,
    patch_token_text_splitter,
    mock_embed_model,
):
    new_docs = [
        Document(id_="1", text="Hello World"),
        Document(id_="2", text="This is a test"),
    ]
    index = VectorStoreIndex.from_documents(
        documents=new_docs, embed_model=mock_embed_model
    )
    assert isinstance(index, VectorStoreIndex)
    await index.arefresh_ref_docs(
        [
            Document(id_="1", text="Hello World v1"),
            Document(id_="2", text="This is a test v1"),
        ]
    )
    actual_node_tups = [
        ("Hello World v1", "1"),
        ("This is a test v1", "2"),
    ]
    for text_id in index.index_struct.nodes_dict:
        node_id = index.index_struct.nodes_dict[text_id]
        node = index.docstore.get_node(node_id)
        # NOTE: this test breaks abstraction
        assert (node.get_content(), node.ref_doc_id) in actual_node_tups
