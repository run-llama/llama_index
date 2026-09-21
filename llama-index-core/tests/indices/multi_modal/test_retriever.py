from llama_index.core import MockEmbedding
from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import IndexNode


def test_retrieve_index_node_initializes_base_retriever_state() -> None:
    index = MultiModalVectorStoreIndex(
        nodes=[IndexNode(text="nested", index_id="nested")],
        embed_model=MockEmbedding(embed_dim=3),
        image_embed_model=MockMultiModalEmbedding(embed_dim=3),
        is_image_vector_store_empty=True,
    )
    retriever = index.as_retriever()

    assert retriever.object_map == {}
    nodes = retriever.retrieve("nested")

    assert len(nodes) == 1
    assert isinstance(nodes[0].node, IndexNode)
    assert nodes[0].node.index_id == "nested"
