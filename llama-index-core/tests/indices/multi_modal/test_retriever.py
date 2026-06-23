from llama_index.core import MockEmbedding
from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import Document, IndexNode, TextNode


def test_retriever_initializes_base_state_from_documents() -> None:
    index = MultiModalVectorStoreIndex.from_documents(
        [Document.example()],
        embed_model=MockEmbedding(embed_dim=3),
        image_embed_model=MockMultiModalEmbedding(embed_dim=3),
    )

    retriever = index.as_retriever()

    assert retriever.object_map == {}


def test_retriever_uses_index_object_map_for_recursive_retrieval() -> None:
    visible_node = TextNode(text="Visible node", id_="visible_node")
    index_node = IndexNode(
        text="Hidden node pointer",
        id_="index_node",
        index_id="hidden_node_index",
        obj=TextNode(text="Hidden node", id_="hidden_node"),
    )
    index = MultiModalVectorStoreIndex(
        nodes=[visible_node],
        objects=[index_node],
        embed_model=MockEmbedding(embed_dim=3),
        image_embed_model=MockMultiModalEmbedding(embed_dim=3),
        is_image_vector_store_empty=True,
    )

    nodes = index.as_retriever(similarity_top_k=2).retrieve("node")

    assert {node.node.id_ for node in nodes} == {"visible_node", "hidden_node"}
