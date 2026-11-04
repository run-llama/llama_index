from typing import List, Tuple

import pytest
from pydantic import PrivateAttr

from llama_index.core.embeddings.mock_embed_model import (
    MockEmbedding,
    MockMultiModalEmbedding,
)
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.indices.multi_modal.retriever import (
    MultiModalVectorIndexRetriever,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import QueryBundle, TextNode


class TrackingEmbedding(MockEmbedding):
    _queries: List[str] = PrivateAttr(default_factory=list)

    def _get_vector(self) -> List[float]:
        return [0.0, 1.0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        self._queries.append(query)
        return self._get_vector()


@pytest.fixture()
def retriever_and_model() -> Tuple[MultiModalVectorIndexRetriever, TrackingEmbedding]:
    embed_model = TrackingEmbedding(embed_dim=2)
    index = MultiModalVectorStoreIndex(
        nodes=[
            TextNode(id_="supplied", text="Supplied vector", embedding=[1.0, 0.0]),
            TextNode(id_="generated", text="Generated vector", embedding=[0.0, 1.0]),
        ],
        embed_model=embed_model,
        image_embed_model=MockMultiModalEmbedding(embed_dim=2),
        is_image_vector_store_empty=True,
        transformations=[TokenTextSplitter(tokenizer=str.split)],
    )
    return index.as_retriever(similarity_top_k=1), embed_model


@pytest.mark.asyncio
@pytest.mark.parametrize("query_str", ["same query", ""])
async def test_atext_retrieve_preserves_supplied_embedding(
    retriever_and_model: Tuple[MultiModalVectorIndexRetriever, TrackingEmbedding],
    query_str: str,
) -> None:
    retriever, embed_model = retriever_and_model
    sync_query = QueryBundle(query_str=query_str, embedding=[1.0, 0.0])
    async_query = QueryBundle(query_str=query_str, embedding=[1.0, 0.0])

    sync_nodes = retriever.text_retrieve(sync_query)
    async_nodes = await retriever.atext_retrieve(async_query)

    assert (
        [node.node.node_id for node in async_nodes]
        == [node.node.node_id for node in sync_nodes]
        == ["supplied"]
    )
    assert async_query.embedding == [1.0, 0.0]
    assert embed_model._queries == []


@pytest.mark.asyncio
async def test_atext_retrieve_embeds_text_when_embedding_is_missing(
    retriever_and_model: Tuple[MultiModalVectorIndexRetriever, TrackingEmbedding],
) -> None:
    retriever, embed_model = retriever_and_model

    nodes = await retriever.atext_retrieve("same query")

    assert [node.node.node_id for node in nodes] == ["generated"]
    assert embed_model._queries == ["same query"]


@pytest.mark.asyncio
async def test_atext_retrieve_uses_custom_embedding_strings(
    retriever_and_model: Tuple[MultiModalVectorIndexRetriever, TrackingEmbedding],
) -> None:
    retriever, embed_model = retriever_and_model
    query = QueryBundle(
        query_str="display query", custom_embedding_strs=["first", "second"]
    )

    nodes = await retriever.atext_retrieve(query)

    assert [node.node.node_id for node in nodes] == ["generated"]
    assert query.embedding == [0.0, 1.0]
    assert embed_model._queries == ["first", "second"]


@pytest.mark.asyncio
async def test_atext_retrieve_skips_embedding_for_empty_text_store() -> None:
    embed_model = TrackingEmbedding(embed_dim=2)
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=embed_model,
        image_embed_model=MockMultiModalEmbedding(embed_dim=2),
        is_text_vector_store_empty=True,
        transformations=[TokenTextSplitter(tokenizer=str.split)],
    )

    nodes = await index.as_retriever().atext_retrieve("same query")

    assert nodes == []
    assert embed_model._queries == []
