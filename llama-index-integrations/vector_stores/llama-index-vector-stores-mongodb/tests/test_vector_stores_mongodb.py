from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.vector_stores.mongodb.pipelines import (
    combine_pipelines,
    final_hybrid_stage,
    fulltext_search_stage,
    reciprocal_rank_stage,
    vector_search_stage,
)


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in MongoDBAtlasVectorSearch.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def make_store(**kwargs: Any) -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        mongodb_client=MagicMock(),
        async_mongodb_client=MagicMock(),
        **kwargs,
    )


def test_vector_pipeline_unchanged() -> None:
    store = make_store(embedding_key="vector", oversampling_factor=5)
    query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=3)
    assert store.is_embedding_query
    assert store._create_query_pipeline(query) == [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "vector",
                "queryVector": [0.1, 0.2],
                "numCandidates": 15,
                "limit": 3,
                "filter": {},
            }
        },
        {"$set": {"score": {"$meta": "vectorSearchScore"}}},
    ]
    with pytest.raises(ValueError, match="query_embedding"):
        store._create_query_pipeline(VectorStoreQuery(query_str="hello"))


@pytest.mark.parametrize("embedding", [None, [0.1, 0.2]])
def test_auto_embed_pipeline(embedding: Optional[list]) -> None:
    store = make_store(
        auto_embed=True,
        text_key="content",
        vector_index_name="auto_index",
        metadata_key="attributes",
        oversampling_factor=5,
    )
    query = VectorStoreQuery(
        query_str="find a document",
        query_embedding=embedding,
        similarity_top_k=3,
        filters=MetadataFilters(filters=[MetadataFilter(key="year", value=2026)]),
    )
    assert not store.is_embedding_query
    assert store._create_query_pipeline(query) == [
        {
            "$vectorSearch": {
                "index": "auto_index",
                "path": "content",
                "query": {"text": "find a document"},
                "numCandidates": 15,
                "limit": 3,
                "filter": {"attributes.year": {"$eq": 2026}},
            }
        },
        {"$set": {"score": {"$meta": "vectorSearchScore"}}},
    ]


@pytest.mark.parametrize("query_str", [None, "", " \t", 123])
@pytest.mark.parametrize(
    "options", [{"auto_embed": True}, {"rerank_model": "rerank-2.5"}]
)
def test_query_text_required(query_str: Any, options: dict) -> None:
    store = make_store(**options)
    with pytest.raises(ValueError, match="requires a non-empty query_str"):
        store._create_query_pipeline(
            VectorStoreQuery(query_str=query_str, query_embedding=[0.1, 0.2])
        )


@pytest.mark.parametrize("auto_embed", [False, True])
@pytest.mark.parametrize("rerank_top_n", [None, 20])
def test_rerank_pipeline(auto_embed: bool, rerank_top_n: Optional[int]) -> None:
    store = make_store(
        auto_embed=auto_embed,
        text_key="content",
        rerank_model="rerank-2.5",
        rerank_top_n=rerank_top_n,
    )
    query = VectorStoreQuery(
        query_str="find a document",
        similarity_top_k=3,
        query_embedding=None if auto_embed else [0.1, 0.2],
    )
    pipeline = store._create_query_pipeline(query)
    retrieval_count = rerank_top_n or 3
    assert pipeline[0] == {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "content" if auto_embed else "embedding",
            **(
                {"query": {"text": query.query_str}}
                if auto_embed
                else {"queryVector": [0.1, 0.2]}
            ),
            "limit": retrieval_count,
            "numCandidates": retrieval_count * 10,
            "filter": {},
        }
    }
    assert pipeline[1:] == [
        {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        {
            "$rerank": {
                "query": {"text": "find a document"},
                "path": "content",
                "model": "rerank-2.5",
                "numDocsToRerank": retrieval_count,
            }
        },
        {"$set": {"score": {"$meta": "score"}}},
        {"$limit": 3},
    ]


@pytest.mark.parametrize(
    "mode", [VectorStoreQueryMode.TEXT_SEARCH, VectorStoreQueryMode.HYBRID]
)
def test_rerank_rejects_other_modes(mode: VectorStoreQueryMode) -> None:
    with pytest.raises(ValueError, match="only supports DEFAULT"):
        make_store(rerank_model="rerank-2.5")._create_query_pipeline(
            VectorStoreQuery(mode=mode, query_str="hello", query_embedding=[0.1])
        )


@pytest.mark.parametrize("value", [0, -1, 1001, 1.5, True])
def test_invalid_rerank_top_n(value: Any) -> None:
    with pytest.raises(ValueError, match="integer between 1 and 1000"):
        make_store(rerank_model="rerank-2.5", rerank_top_n=value)


def test_rerank_configuration() -> None:
    with pytest.raises(ValueError, match="requires rerank_model"):
        make_store(rerank_top_n=20)
    with pytest.raises(ValueError, match="non-empty model name"):
        make_store(rerank_model=" ")
    with pytest.raises(ValueError, match="at least similarity_top_k"):
        make_store(rerank_model="rerank-2.5", rerank_top_n=2)._create_query_pipeline(
            VectorStoreQuery(
                query_str="hello", query_embedding=[0.1], similarity_top_k=3
            )
        )
    with pytest.raises(ValueError, match="between 1 and 1000"):
        make_store(rerank_model="rerank-2.5")._create_query_pipeline(
            VectorStoreQuery(
                query_str="hello", query_embedding=[0.1], similarity_top_k=1001
            )
        )


@pytest.mark.parametrize("auto_embed", [False, True])
def test_insert_embeddings(auto_embed: bool) -> None:
    store = make_store(auto_embed=auto_embed, text_key="content")
    node = TextNode(text="hello", embedding=None if auto_embed else [0.1, 0.2])
    ids, documents = store._create_data_to_insert([node])
    assert ids == [node.node_id]
    assert documents[0]["content"] == "hello"
    if auto_embed:
        assert "embedding" not in documents[0]
    else:
        assert documents[0]["embedding"] == [0.1, 0.2]


def test_text_pipeline_unchanged() -> None:
    query = VectorStoreQuery(
        mode=VectorStoreQueryMode.TEXT_SEARCH, query_str="hello", similarity_top_k=3
    )
    assert make_store()._create_query_pipeline(query) == [
        {
            "$search": {
                "index": "fulltext_index",
                "text": {"query": "hello", "path": "text"},
            }
        },
        {"$limit": 3},
        {"$set": {"score": {"$meta": "searchScore"}}},
    ]


@pytest.mark.parametrize("auto_embed", [False, True])
def test_hybrid_pipeline(auto_embed: bool) -> None:
    store = make_store(auto_embed=auto_embed)
    query = VectorStoreQuery(
        mode=VectorStoreQueryMode.HYBRID,
        query_str="hello",
        similarity_top_k=3,
        query_embedding=None if auto_embed else [0.1, 0.2],
    )
    expected = [
        vector_search_stage(
            query_vector=query.query_embedding,
            query_text="hello" if auto_embed else None,
            search_field="text" if auto_embed else "embedding",
            index_name="vector_index",
            limit=3,
        )
    ]
    expected.extend(reciprocal_rank_stage("vector_score"))
    text_pipeline = fulltext_search_stage("hello", "text", "fulltext_index", limit=3)
    text_pipeline.extend(reciprocal_rank_stage("fulltext_score"))
    combine_pipelines(expected, text_pipeline, store.collection.name)
    expected.extend(
        final_hybrid_stage(["vector_score", "fulltext_score"], limit=3, alpha=0.5)
    )
    expected.append({"$project": {"embedding": 0}})
    assert store._create_query_pipeline(query) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_auto_embed_retriever_with_reranking(use_async: bool) -> None:
    store = make_store(auto_embed=True, rerank_model="rerank-2.5")
    result = {"_id": "node-id", "text": "hello", "metadata": {}, "score": 0.9}
    store.collection.aggregate.return_value = [result]
    cursor = MagicMock()
    cursor.__aiter__.return_value = [result]
    store.async_collection.aggregate = AsyncMock(return_value=cursor)
    index = VectorStoreIndex.from_vector_store(
        store, embed_model=MockEmbedding(embed_dim=2)
    )
    retriever = index.as_retriever(similarity_top_k=1)
    with (
        patch.object(
            MockEmbedding,
            "_get_query_embedding",
            side_effect=AssertionError("Client embedding called"),
        ),
        patch.object(
            MockEmbedding,
            "_aget_query_embedding",
            side_effect=AssertionError("Client embedding called"),
        ),
    ):
        nodes = (
            await retriever.aretrieve("hello")
            if use_async
            else retriever.retrieve("hello")
        )
    assert len(nodes) == 1
    assert nodes[0].node.node_id == "node-id"
    assert nodes[0].node.get_content() == "hello"
    assert nodes[0].score == 0.9
    aggregate = (
        store.async_collection.aggregate if use_async else store.collection.aggregate
    )
    aggregate.assert_called_once_with(
        store._create_query_pipeline(
            VectorStoreQuery(query_str="hello", similarity_top_k=1)
        )
    )
