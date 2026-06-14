import pytest

from llama_index.core import MockEmbedding, StorageContext
from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode, QueryBundle
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)


@pytest.fixture()
def image_retriever():
    matching_node = ImageNode(
        id_="matching-image",
        image_path="matching.png",
        metadata={"category": "keep"},
    )
    filtered_node = ImageNode(
        id_="filtered-image",
        image_path="filtered.png",
        metadata={"category": "skip"},
    )
    storage_context = StorageContext.from_defaults(image_store=SimpleVectorStore())
    index = MultiModalVectorStoreIndex(
        [matching_node, filtered_node],
        storage_context=storage_context,
        embed_model=MockEmbedding(embed_dim=3),
        image_embed_model=MockMultiModalEmbedding(embed_dim=3),
    )
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", operator=FilterOperator.EQ, value="keep")
        ]
    )
    return index.as_retriever(filters=filters, image_similarity_top_k=2)


def test_image_to_image_retrieve_applies_filters_for_image_path(image_retriever):
    results = image_retriever.image_to_image_retrieve("query.png")

    assert [node.node.node_id for node in results] == ["matching-image"]


def test_image_to_image_retrieve_applies_filters_for_query_bundle(image_retriever):
    results = image_retriever.image_to_image_retrieve(
        QueryBundle(query_str="", image_path="query.png")
    )

    assert [node.node.node_id for node in results] == ["matching-image"]


@pytest.mark.asyncio
async def test_aimage_to_image_retrieve_applies_filters_for_image_path(image_retriever):
    results = await image_retriever.aimage_to_image_retrieve("query.png")

    assert [node.node.node_id for node in results] == ["matching-image"]
