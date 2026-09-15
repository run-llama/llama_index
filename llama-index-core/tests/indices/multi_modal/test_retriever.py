import pytest
from typing import List
from llama_index.core import MockEmbedding
from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

BASE64_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="


def _get_mock_embed_models():
    embed_model_text = MockEmbedding(embed_dim=4)
    embed_model_image = MockMultiModalEmbedding(embed_dim=4)
    return embed_model_text, embed_model_image


def test_image_only_retriever_with_metadata_filters():
    """Test image-to-image and text-to-image retrieval on image-only index with metadata filters."""
    embed_model_text, embed_model_image = _get_mock_embed_models()

    nodes: List[ImageNode] = [
        ImageNode(
            image=BASE64_PNG,
            id_="img_1",
            metadata={"category": "cat", "tag": "cute"},
        ),
        ImageNode(
            image=BASE64_PNG,
            id_="img_2",
            metadata={"category": "dog", "tag": "cute"},
        ),
        ImageNode(
            image=BASE64_PNG,
            id_="img_3",
            metadata={"category": "cat", "tag": "funny"},
        ),
    ]

    index = MultiModalVectorStoreIndex(
        nodes=nodes,
        image_embed_model=embed_model_image,
        embed_model=embed_model_text,
    )

    # Filter for category == "cat"
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", operator=FilterOperator.EQ, value="cat")
        ]
    )

    retriever = index.as_retriever(filters=filters, image_similarity_top_k=5)

    # Image-to-image retrieval
    img_results = retriever.image_to_image_retrieve(BASE64_PNG)
    assert len(img_results) == 2
    for res in img_results:
        assert isinstance(res.node, ImageNode)
        assert res.node.metadata["category"] == "cat"
    assert {res.node.node_id for res in img_results} == {"img_1", "img_3"}

    # Text-to-image retrieval
    text_to_img_results = retriever.text_to_image_retrieve("a photo of a cat")
    assert len(text_to_img_results) == 2
    for res in text_to_img_results:
        assert isinstance(res.node, ImageNode)
        assert res.node.metadata["category"] == "cat"


def test_multimodal_retriever_mixed_with_filters():
    """Test retrieval on mixed text + image index with shared and modality-specific filters."""
    embed_model_text, embed_model_image = _get_mock_embed_models()

    text_nodes = [
        TextNode(
            text="Cat fact: cats sleep a lot.",
            id_="txt_1",
            metadata={"species": "feline"},
        ),
        TextNode(
            text="Dog fact: dogs bark.", id_="txt_2", metadata={"species": "canine"}
        ),
    ]
    image_nodes = [
        ImageNode(image=BASE64_PNG, id_="img_1", metadata={"species": "feline"}),
        ImageNode(image=BASE64_PNG, id_="img_2", metadata={"species": "canine"}),
    ]

    index = MultiModalVectorStoreIndex(
        nodes=text_nodes + image_nodes,
        image_embed_model=embed_model_image,
        embed_model=embed_model_text,
    )

    # Shared filter for species == "feline"
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="species", operator=FilterOperator.EQ, value="feline")
        ]
    )

    retriever = index.as_retriever(
        filters=filters,
        similarity_top_k=5,
        image_similarity_top_k=5,
    )

    # Combined retrieve
    results = retriever.retrieve("feline info")
    assert len(results) == 2
    node_ids = {r.node.node_id for r in results}
    assert node_ids == {"txt_1", "img_1"}
    for r in results:
        assert r.node.metadata["species"] == "feline"

    # Modality-specific filters (image_filters overriding filters for images)
    image_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="species", operator=FilterOperator.EQ, value="canine")
        ]
    )
    retriever_custom = index.as_retriever(
        filters=filters,
        image_filters=image_filters,
        similarity_top_k=5,
        image_similarity_top_k=5,
    )

    # Text retrieve should still use filters (feline), text-to-image should use image_filters (canine)
    txt_res = retriever_custom.text_retrieve("info")
    assert len(txt_res) == 1
    assert txt_res[0].node.node_id == "txt_1"

    img_res = retriever_custom.text_to_image_retrieve("info")
    assert len(img_res) == 1
    assert img_res[0].node.node_id == "img_2"


@pytest.mark.asyncio
async def test_async_multimodal_retriever_with_filters():
    """Test async retrieval methods on MultiModalVectorIndexRetriever with filters."""
    embed_model_text, embed_model_image = _get_mock_embed_models()

    text_nodes = [
        TextNode(text="Elephant text", id_="txt_1", metadata={"type": "wild"}),
        TextNode(text="Dog text", id_="txt_2", metadata={"type": "pet"}),
    ]
    image_nodes = [
        ImageNode(image=BASE64_PNG, id_="img_1", metadata={"type": "wild"}),
        ImageNode(image=BASE64_PNG, id_="img_2", metadata={"type": "pet"}),
    ]

    index = MultiModalVectorStoreIndex(
        nodes=text_nodes + image_nodes,
        image_embed_model=embed_model_image,
        embed_model=embed_model_text,
    )

    filters = MetadataFilters(
        filters=[MetadataFilter(key="type", operator=FilterOperator.EQ, value="wild")]
    )

    retriever = index.as_retriever(
        filters=filters,
        similarity_top_k=5,
        image_similarity_top_k=5,
    )

    # Async image-to-image
    i2i_res = await retriever.aimage_to_image_retrieve(BASE64_PNG)
    assert len(i2i_res) == 1
    assert i2i_res[0].node.node_id == "img_1"

    # Async text-to-image
    t2i_res = await retriever.atext_to_image_retrieve("wild elephant")
    assert len(t2i_res) == 1
    assert t2i_res[0].node.node_id == "img_1"

    # Async text
    txt_res = await retriever.atext_retrieve("wild elephant")
    assert len(txt_res) == 1
    assert txt_res[0].node.node_id == "txt_1"

    # Async combined
    all_res = await retriever.aretrieve("wild animals")
    assert len(all_res) == 2
    assert {r.node.node_id for r in all_res} == {"txt_1", "img_1"}
