import aiohttp  # noqa
import logging
import os
import re
import uuid
from typing import AsyncGenerator, List, Generator

from elasticsearch import AsyncElasticsearch, ConnectionError

import pandas as pd
import pytest
import pytest_asyncio

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.elasticsearch import (
    ElasticsearchStore,
    AsyncBM25Strategy,
    AsyncDenseVectorStrategy,
    AsyncSparseVectorStrategy,
)
from llama_index.vector_stores.elasticsearch.base import (
    _mode_must_match_retrieval_strategy,
)

from llama_index.vector_stores.elasticsearch.utils import get_elasticsearch_client
from llama_index.vector_stores.elasticsearch.base import _to_elasticsearch_filter

##
# Start Elasticsearch locally
# cd tests
# docker-compose up elasticsearch
#
# Run tests
# cd tests
# pytest test_vector_stores_elasticsearch.py


logging.basicConfig(level=logging.DEBUG)


@pytest.fixture()
def index_name() -> str:
    """Return the index name."""
    return f"test_{uuid.uuid4().hex}"


@pytest_asyncio.fixture(scope="function")
async def es_client() -> AsyncGenerator[AsyncElasticsearch, None]:
    es_client = None

    try:
        # Create client and test connection
        es_client = get_elasticsearch_client(
            url=os.environ.get("ES_URL", "http://localhost:9200"),
            cloud_id=os.environ.get("ES_CLOUD_ID"),
            api_key=os.environ.get("ES_API_KEY"),
            username=os.environ.get("ES_USERNAME", "elastic"),
            password=os.environ.get("ES_PASSWORD", "changeme"),
        )

        yield es_client

        # Clear all indexes
        index_response = await es_client.indices.get(index="_all")
        index_names = index_response.keys()
        for index_name in index_names:
            if index_name.startswith("test_"):
                await es_client.indices.delete(index=index_name)
        await es_client.indices.refresh(index="_all")

    except ConnectionError as err:
        pytest.skip(f"Could not connect to Elasticsearch: {err}")

    finally:
        if es_client:
            await es_client.close()


@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={
                "author": "Stephen King",
                "theme": "Friendship",
            },
            embedding=[1.0, 0.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
            embedding=[0.0, 1.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "director": "Christopher Nolan",
            },
            embedding=[0.0, 0.5, 1.0],
        ),
        TextNode(
            text="I was taught that the way of progress was neither swift nor easy.",
            id_="0b31ae71-b797-4e88-8495-031371a7752e",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="text-3")},
            metadata={
                "author": "Marie Curie",
            },
            embedding=[0.0, 0.0, 0.9],
        ),
        TextNode(
            text=(
                "The important thing is not to stop questioning."
                + " Curiosity has its own reason for existing."
            ),
            id_="bd2e080b-159a-4030-acc3-d98afd2ba49b",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="text-4")},
            metadata={
                "author": "Albert Einstein",
            },
            embedding=[0.0, 0.0, 0.5],
        ),
        TextNode(
            text=(
                "I am no bird; and no net ensnares me;"
                + " I am a free human being with an independent will."
            ),
            id_="f658de3b-8cef-4d1c-8bed-9a263c907251",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="text-5")},
            metadata={
                "author": "Charlotte Bronte",
            },
            embedding=[0.0, 0.0, 0.3],
        ),
    ]


def test_instance_creation(index_name: str, es_client: AsyncElasticsearch) -> None:
    url = os.environ.get("ES_URL", "http://localhost:9200")
    cloud_id = os.environ.get("ES_CLOUD_ID")
    api_key = os.environ.get("ES_API_KEY")
    user = os.environ.get("ES_USERNAME")
    password = os.environ.get("ES_PASSWORD")

    es_store = ElasticsearchStore(
        index_name=index_name,
        es_url=url,
        es_cloud_id=cloud_id,
        es_api_key=api_key,
        es_user=user,
        es_password=password,
    )
    assert isinstance(es_store, ElasticsearchStore)
    es_store.close()


def test_mode_must_match_retrieval_strategy() -> None:
    # DEFAULT mode should never raise any exception
    mode = VectorStoreQueryMode.DEFAULT
    retrieval_strategy = AsyncBM25Strategy()
    _mode_must_match_retrieval_strategy(mode, retrieval_strategy)

    # AsyncSparseVectorStrategy with mode SPARSE should not raise any exception
    mode = VectorStoreQueryMode.SPARSE
    retrieval_strategy = AsyncSparseVectorStrategy()
    _mode_must_match_retrieval_strategy(mode, retrieval_strategy)

    # AsyncBM25Strategy with TEXT_SEARCH should not raise any exception
    mode = VectorStoreQueryMode.TEXT_SEARCH
    retrieval_strategy = AsyncBM25Strategy()
    _mode_must_match_retrieval_strategy(mode, retrieval_strategy)

    # AsyncDenseVectorStrategy(hybrid=True) with mode HYBRID should not raise any exception
    mode = VectorStoreQueryMode.HYBRID
    retrieval_strategy = AsyncDenseVectorStrategy(hybrid=True)
    _mode_must_match_retrieval_strategy(mode, retrieval_strategy)

    # unknown mode should raise NotImplementedError
    for mode in [
        VectorStoreQueryMode.SEMANTIC_HYBRID,
        VectorStoreQueryMode.SVM,
        VectorStoreQueryMode.LOGISTIC_REGRESSION,
        VectorStoreQueryMode.LINEAR_REGRESSION,
        VectorStoreQueryMode.MMR,
    ]:
        retrieval_strategy = AsyncDenseVectorStrategy()
        with pytest.raises(NotImplementedError):
            _mode_must_match_retrieval_strategy(mode, retrieval_strategy)

    # if mode is SPARSE and strategy is not AsyncSparseVectorStrategy, should raise ValueError
    mode = VectorStoreQueryMode.SPARSE
    retrieval_strategy = AsyncDenseVectorStrategy()
    with pytest.raises(ValueError):
        _mode_must_match_retrieval_strategy(mode, retrieval_strategy)

    # if mode is HYBRID and strategy is not AsyncDenseVectorStrategy, should raise ValueError
    mode = VectorStoreQueryMode.HYBRID
    retrieval_strategy = AsyncSparseVectorStrategy()
    with pytest.raises(ValueError):
        _mode_must_match_retrieval_strategy(mode, retrieval_strategy)

    # if mode is HYBRID and strategy is AsyncDenseVectorStrategy but hybrid is not enabled, should raise ValueError
    mode = VectorStoreQueryMode.HYBRID
    retrieval_strategy = AsyncDenseVectorStrategy(hybrid=False)
    with pytest.raises(ValueError):
        _mode_must_match_retrieval_strategy(mode, retrieval_strategy)


@pytest.fixture()
def es_store(
    index_name: str, es_client: AsyncElasticsearch
) -> Generator[ElasticsearchStore, None, None]:
    store = ElasticsearchStore(
        es_client=es_client,
        index_name=index_name,
        distance_strategy="EUCLIDEAN_DISTANCE",
    )
    try:
        yield store
    finally:
        store.close()


@pytest.fixture()
def es_hybrid_store(
    index_name: str, es_client: AsyncElasticsearch
) -> Generator[ElasticsearchStore, None, None]:
    store = ElasticsearchStore(
        es_client=es_client,
        index_name=index_name,
        distance_strategy="EUCLIDEAN_DISTANCE",
        retrieval_strategy=AsyncDenseVectorStrategy(hybrid=True),
    )
    try:
        yield store
    finally:
        store.close()


@pytest.fixture()
def es_bm25_store(
    index_name: str, es_client: AsyncElasticsearch
) -> Generator[ElasticsearchStore, None, None]:
    store = ElasticsearchStore(
        es_client=es_client,
        index_name=index_name,
        retrieval_strategy=AsyncBM25Strategy(),
    )
    try:
        yield store
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_query(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await es_store.async_add(node_embeddings)
        res = await es_store.aquery(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )
    else:
        es_store.add(node_embeddings)
        res = es_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_text_query(
    es_bm25_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await es_bm25_store.async_add(node_embeddings)
        res = await es_bm25_store.aquery(
            VectorStoreQuery(
                query_str="lorem",
                mode=VectorStoreQueryMode.TEXT_SEARCH,
                similarity_top_k=1,
            )
        )
    else:
        es_bm25_store.add(node_embeddings)
        res = es_bm25_store.query(
            VectorStoreQuery(
                query_str="lorem",
                mode=VectorStoreQueryMode.TEXT_SEARCH,
                similarity_top_k=1,
            )
        )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_hybrid_query(
    es_client: AsyncElasticsearch,
    es_hybrid_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if (await es_client.license.get())["license"]["type"] == "basic":
        pytest.skip("This test requires a non-basic license.")

    if use_async:
        await es_hybrid_store.async_add(node_embeddings)
        res = await es_hybrid_store.aquery(
            VectorStoreQuery(
                query_str="lorem",
                query_embedding=[1.0, 0.0, 0.0],
                mode=VectorStoreQueryMode.HYBRID,
                similarity_top_k=1,
            )
        )
    else:
        es_hybrid_store.add(node_embeddings)
        res = es_hybrid_store.query(
            VectorStoreQuery(
                query_str="lorem",
                query_embedding=[1.0, 0.0, 0.0],
                mode=VectorStoreQueryMode.HYBRID,
                similarity_top_k=1,
            )
        )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_query_with_filters(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="author", value="Stephen King")]
    )
    q = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0], similarity_top_k=10, filters=filters
    )
    if use_async:
        await es_store.async_add(node_embeddings)
        res = await es_store.aquery(q)
    else:
        es_store.add(node_embeddings)
        res = es_store.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_query_with_es_filters(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    q = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=10)
    if use_async:
        await es_store.async_add(node_embeddings)
        res = await es_store.aquery(
            q, es_filter=[{"wildcard": {"metadata.author": "stephe*"}}]
        )
    else:
        es_store.add(node_embeddings)
        res = es_store.query(
            q, es_filter=[{"wildcard": {"metadata.author": "stephe*"}}]
        )
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_query_and_delete(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    q = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)

    if use_async:
        await es_store.async_add(node_embeddings)
        res = await es_store.aquery(q)
    else:
        es_store.add(node_embeddings)
        res = es_store.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"

    if use_async:
        await es_store.adelete("test-0")
        res = await es_store.aquery(q)
    else:
        es_store.delete("test-0")
        res = es_store.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "f658de3b-8cef-4d1c-8bed-9a263c907251"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_embed_query_ranked(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    einstein_bronte_curie = [
        "bd2e080b-159a-4030-acc3-d98afd2ba49b",
        "f658de3b-8cef-4d1c-8bed-9a263c907251",
        "0b31ae71-b797-4e88-8495-031371a7752e",
    ]
    query_get_1_first = VectorStoreQuery(
        query_embedding=[0.0, 0.0, 0.5], similarity_top_k=3
    )
    await check_top_match(
        es_store, node_embeddings, use_async, query_get_1_first, *einstein_bronte_curie
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_text_query_ranked(
    es_bm25_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    node1 = "0b31ae71-b797-4e88-8495-031371a7752e"
    node2 = "f658de3b-8cef-4d1c-8bed-9a263c907251"

    query_get_1_first = VectorStoreQuery(
        query_str="I was", mode=VectorStoreQueryMode.TEXT_SEARCH, similarity_top_k=2
    )
    await check_top_match(
        es_bm25_store, node_embeddings, use_async, query_get_1_first, node1, node2
    )

    query_get_2_first = VectorStoreQuery(
        query_str="I am", mode=VectorStoreQueryMode.TEXT_SEARCH, similarity_top_k=2
    )
    await check_top_match(
        es_bm25_store, node_embeddings, use_async, query_get_2_first, node2, node1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_text_query_ranked_hybrid(
    es_hybrid_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    node1 = "f658de3b-8cef-4d1c-8bed-9a263c907251"
    node2 = "0b31ae71-b797-4e88-8495-031371a7752e"

    query_get_1_first = VectorStoreQuery(
        query_str="human",
        query_embedding=[0.0, 0.0, 0.5],
        mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=3,
    )
    await check_top_match(
        es_hybrid_store, node_embeddings, use_async, query_get_1_first, node1, node2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_text_query_ranked_hybrid_large_top_k(
    es_hybrid_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    node1 = "f658de3b-8cef-4d1c-8bed-9a263c907251"
    node2 = "0b31ae71-b797-4e88-8495-031371a7752e"

    query_get_1_first = VectorStoreQuery(
        query_str="human",
        query_embedding=[0.0, 0.0, 0.5],
        mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=10,
    )
    await check_top_match(
        es_hybrid_store, node_embeddings, use_async, query_get_1_first, node1, node2
    )


def test_check_user_agent(es_store: ElasticsearchStore) -> None:
    user_agent = es_store._store.client._headers["User-Agent"]
    pattern = r"^llama_index-py-vs/\d+\.\d+\.\d+(\.post\d+)?$"
    assert re.match(pattern, user_agent) is not None, (
        f"The string '{user_agent}' does not match the expected user-agent."
    )


async def check_top_match(
    store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
    query: VectorStoreQuery,
    *expected_nodes: str,
) -> None:
    if use_async:
        await store.async_add(node_embeddings)
        res = await store.aquery(query)
    else:
        store.add(node_embeddings)
        res = store.query(query)
    assert res.nodes

    # test the nodes are return in the expected order
    for i, node in enumerate(expected_nodes):
        assert res.nodes[i].node_id == node
    # test the returned order is in descending order w.r.t. similarities
    # test similarities are normalized (0, 1)
    df = pd.DataFrame({"node": res.nodes, "sim": res.similarities, "id": res.ids})
    sorted_by_sim = df.sort_values(by="sim", ascending=False)
    for idx, item in enumerate(sorted_by_sim.itertuples()):
        res_node = res.nodes[idx]
        assert res_node.node_id == item.id
        assert 0 <= item.sim <= 1


def test_metadata_filter_to_es_filter() -> None:
    metadata_filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="k1", value="v1"),
            ExactMatchFilter(key="k2", value="v2"),
        ]
    )
    es_filter_default = _to_elasticsearch_filter(standard_filters=metadata_filters)
    assert es_filter_default == {
        "bool": {
            "must": [
                {"term": {"metadata.k1.keyword": {"value": "v1"}}},
                {"term": {"metadata.k2.keyword": {"value": "v2"}}},
            ]
        }
    }
    es_filter_enum = _to_elasticsearch_filter(
        standard_filters=metadata_filters, metadata_keyword_suffix=".enum"
    )
    assert es_filter_enum == {
        "bool": {
            "must": [
                {"term": {"metadata.k1.enum": {"value": "v1"}}},
                {"term": {"metadata.k2.enum": {"value": "v2"}}},
            ]
        }
    }
    es_filter_empty = _to_elasticsearch_filter(
        standard_filters=metadata_filters, metadata_keyword_suffix=""
    )
    assert es_filter_empty == {
        "bool": {
            "must": [
                {"term": {"metadata.k1": {"value": "v1"}}},
                {"term": {"metadata.k2": {"value": "v2"}}},
            ]
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_delete_nodes(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await es_store.async_add(node_embeddings)
    else:
        es_store.add(node_embeddings)

    node_ids = [node_embeddings[0].node_id, node_embeddings[1].node_id]
    if use_async:
        await es_store.adelete_nodes(node_ids=node_ids)
    else:
        es_store.delete_nodes(node_ids=node_ids)

    res = es_store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=5)
    )
    assert len(res.nodes) == 4
    assert all(node.node_id not in node_ids for node in res.nodes)

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="author", value="Marie Curie")]
    )
    if use_async:
        await es_store.adelete_nodes(filters=filters)
    else:
        es_store.delete_nodes(filters=filters)

    res = es_store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=5)
    )
    assert len(res.nodes) == 3
    assert all(node.metadata.get("author") != "Marie Curie" for node in res.nodes)

    remaining_node_ids = [node.node_id for node in res.nodes[:2]]
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="author", value="Albert Einstein")]
    )
    if use_async:
        await es_store.adelete_nodes(node_ids=remaining_node_ids, filters=filters)
    else:
        es_store.delete_nodes(node_ids=remaining_node_ids, filters=filters)

    res = es_store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=5)
    )
    assert len(res.nodes) == 2
    assert any(node.metadata.get("author") == "Charlotte Bronte" for node in res.nodes)
    assert any(
        node.metadata.get("director") == "Christopher Nolan" for node in res.nodes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_get_nodes(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    """Test the get_nodes method with node_ids and filters."""
    if use_async:
        await es_store.async_add(node_embeddings)
    else:
        es_store.add(node_embeddings)

    node_ids = [node_embeddings[0].node_id, node_embeddings[1].node_id]
    if use_async:
        nodes = await es_store.aget_nodes(node_ids=node_ids)
    else:
        nodes = es_store.get_nodes(node_ids=node_ids)

    assert len(nodes) == 2
    retrieved_node_ids = [node.node_id for node in nodes]
    assert all(node_id in retrieved_node_ids for node_id in node_ids)

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="author", value="Stephen King")]
    )
    if use_async:
        nodes = await es_store.aget_nodes(filters=filters)
    else:
        nodes = es_store.get_nodes(filters=filters)

    assert len(nodes) == 1
    assert nodes[0].metadata["author"] == "Stephen King"

    assert nodes[0].get_content() == "lorem ipsum"

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="author", value="Non-existent Author")]
    )
    if use_async:
        nodes = await es_store.aget_nodes(filters=filters)
    else:
        nodes = es_store.get_nodes(filters=filters)

    assert len(nodes) == 0

    with pytest.raises(ValueError):
        if use_async:
            await es_store.aget_nodes()
        else:
            es_store.get_nodes()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_clear(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    """Test that clear/aclear methods properly delete all data from the index."""
    if use_async:
        await es_store.async_add(node_embeddings)
    else:
        es_store.add(node_embeddings)

    q = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=10)
    if use_async:
        res = await es_store.aquery(q)
    else:
        res = es_store.query(q)
    assert len(res.nodes) > 0

    if use_async:
        await es_store.aclear()
    else:
        es_store.clear()

    if use_async:
        await es_store.async_add([node_embeddings[0]])
        res = await es_store.aquery(q)
    else:
        es_store.add([node_embeddings[0]])
        res = es_store.query(q)

    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == node_embeddings[0].node_id
