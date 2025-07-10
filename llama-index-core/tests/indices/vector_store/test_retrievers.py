from typing import List, cast

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import (
    Document,
    NodeRelationship,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
    ImageNode,
)
from llama_index.core.vector_stores.simple import SimpleVectorStore


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
    image_node = ImageNode(image="potato")

    index = VectorStoreIndex.from_documents([])
    index.insert_nodes([image_node])

    # test embedding query
    query_str = "What is?"
    retriever = index.as_retriever()
    results = retriever.retrieve(QueryBundle(query_str))

    assert len(results) == 1

    assert results[0].node.node_id == image_node.node_id
    assert isinstance(results[0].node, ImageNode)
    assert results[0].node.image == "potato"
