import pytest

from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode, TextNode


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_mixed_insert_batches_keep_image_retrieval_available(tmp_path, use_async):
    image_path = tmp_path / "diagram.png"
    image_path.touch()
    embedding = MockMultiModalEmbedding(embed_dim=4)
    index = MultiModalVectorStoreIndex(
        [TextNode(text="Initial note")],
        embed_model=embedding,
        image_embed_model=embedding,
    )
    assert index.is_image_vector_store_empty

    async def insert(node):
        if use_async:
            await index.ainsert_nodes([node])
        else:
            index.insert_nodes([node])

    await insert(ImageNode(id_="diagram", image_path=str(image_path)))
    retriever = index.as_retriever()

    async def retrieve_images():
        if use_async:
            return await retriever.aimage_to_image_retrieve(str(image_path))
        return retriever.image_to_image_retrieve(str(image_path))

    assert [hit.node.node_id for hit in await retrieve_images()] == ["diagram"]
    assert not index.is_text_vector_store_empty

    await insert(TextNode(text="Follow-up note"))
    assert [hit.node.node_id for hit in await retrieve_images()] == ["diagram"]
