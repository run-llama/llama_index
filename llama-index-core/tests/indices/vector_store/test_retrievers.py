from typing import List, cast
import pytest
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.schema import (
    Document,
    NodeRelationship,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
    ImageNode,
)


def test_simple_query(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    mock_embed_model,
) -> None:
    """Test embedding query."""
    index = VectorStoreIndex.from_documents(documents, embed_model=mock_embed_model)

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "This is another test."


def test_query_and_similarity_scores(
    patch_llm_predictor,
    patch_token_text_splitter,
) -> None:
    """Test that sources nodes have similarity scores."""
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    document = Document(text=doc_text)
    index = VectorStoreIndex.from_documents([document])

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert len(nodes) > 0
    assert nodes[0].score is not None


def test_simple_check_ids(
    patch_llm_predictor,
    patch_token_text_splitter,
) -> None:
    """Test build VectorStoreIndex."""
    ref_doc_id = "ref_doc_id_test"
    source_rel = {NodeRelationship.SOURCE: RelatedNodeInfo(node_id=ref_doc_id)}
    all_nodes = [
        TextNode(text="Hello world.", id_="node1", relationships=source_rel),
        TextNode(text="This is a test.", id_="node2", relationships=source_rel),
        TextNode(text="This is another test.", id_="node3", relationships=source_rel),
        TextNode(text="This is a test v2.", id_="node4", relationships=source_rel),
    ]
    index = VectorStoreIndex(all_nodes)

    # test query
    query_str = "What is?"
    retriever = index.as_retriever()
    nodes = retriever.retrieve(QueryBundle(query_str))
    assert nodes[0].node.get_content() == "This is another test."
    assert nodes[0].node.ref_doc_id == "ref_doc_id_test"
    assert nodes[0].node.node_id == "node3"
    vector_store = cast(SimpleVectorStore, index._vector_store)
    assert "node3" in vector_store._data.embedding_dict
    assert "node3" in vector_store._data.text_id_to_ref_doc_id


def test_query(
    patch_llm_predictor,
    patch_token_text_splitter,
) -> None:
    """Test embedding query."""
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    document = Document(text=doc_text)
    index = VectorStoreIndex.from_documents([document])

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    _ = retriever.retrieve(QueryBundle(query_str))


def test_query_image_node() -> None:
    """Test embedding query."""
    image_node = ImageNode(
        image="potato", embeddings=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    text_node = TextNode(
        text="potato", embeddings=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    index = VectorStoreIndex.from_documents([])
    index.insert_nodes([image_node, text_node])

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    results = retriever.retrieve(QueryBundle(query_str))

    assert len(results) == 2

    text_node = next(
        node
        for node in results
        if isinstance(node.node, TextNode) and not isinstance(node.node, ImageNode)
    )
    image_node = next(node for node in results if isinstance(node.node, ImageNode))

    assert image_node.node.node_id == image_node.node_id
    assert isinstance(image_node.node, ImageNode)
    assert image_node.node.image == "potato"
    assert text_node.node.node_id == text_node.node_id
    assert isinstance(text_node.node, TextNode)
    assert text_node.node.text == "potato"


def test_insert_fetched_nodes_handles_all_branches():
    """Test _insert_fetched_nodes_into_query_result for full branch coverage."""
    fetched_nodes = [
        TextNode(id_="0", text="doc 0"),
        TextNode(id_="1", text="doc 1"),
        TextNode(id_="two", text="doc two"),
    ]

    query_result = VectorStoreQueryResult(
        ids=[0, "1", "unknown"], similarities=[0.9, 0.8, 0.7], nodes=None
    )

    dummy_index = VectorStoreIndex([])

    retriever = VectorIndexRetriever(
        index=dummy_index, vector_store=None, docstore=None, embed_model=None
    )

    with pytest.raises(KeyError) as exc_info:
        retriever._insert_fetched_nodes_into_query_result(query_result, fetched_nodes)

    assert "Node ID 0 not found in index." in str(exc_info.value)


def test_insert_fetched_nodes_with_nodes_present():
    """Test _insert_fetched_nodes_into_query_result with `nodes` present instead of `ids`."""
    fetched_nodes = [TextNode(id_="abc", text="Updated text")]

    # This simulates query_result.nodes populated with old version of the same node
    old_node = TextNode(id_="abc", text="Old text")

    query_result = VectorStoreQueryResult(nodes=[old_node], similarities=[0.9])

    dummy_index = VectorStoreIndex([])

    retriever = VectorIndexRetriever(
        index=dummy_index, vector_store=None, docstore=None, embed_model=None
    )

    new_nodes = retriever._insert_fetched_nodes_into_query_result(
        query_result, fetched_nodes
    )

    # Should have replaced old node with the new one
    assert len(new_nodes) == 1
    assert new_nodes[0].text == "Updated text"
