"""Test vector store indexes."""

from typing import List


from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.sentence_window import SentenceWindowVectorIndex

from llama_index.schema import Document, MetadataMode
from llama_index.vector_stores.simple import SimpleVectorStore


def test_build_simple(
    mock_service_context: ServiceContext,
    documents: List[Document],
) -> None:
    """Test build VectorStoreIndex."""

    # merge documents
    document = Document(text=" ".join([d.text for d in documents]))

    index = SentenceWindowVectorIndex.from_documents(
        documents=[document], service_context=mock_service_context
    )
    assert isinstance(index, SentenceWindowVectorIndex)
    assert len(index.index_struct.nodes_dict) == 4
    # check contents of nodes, should be split into sentences
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
        assert (
            node.get_content(metadata_mode=MetadataMode.NONE),
            embedding,
        ) in actual_node_tups

        # test sentence metadata
        assert (
            node.metadata.get(SentenceWindowVectorIndex.window_metadata_key) is not None
        )


def test_simple_insert(
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test insert VectorStoreIndex."""
    # merge documents
    document = Document(text=" ".join([d.text for d in documents]))

    index = SentenceWindowVectorIndex.from_documents(
        documents=[document], service_context=mock_service_context
    )
    assert isinstance(index, SentenceWindowVectorIndex)
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
        assert (
            node.get_content(metadata_mode=MetadataMode.NONE),
            embedding,
        ) in actual_node_tups
