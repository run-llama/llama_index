import logging
import os
import re
import uuid
from typing import Any, Dict, Generator, List, Union

import pandas as pd
import pytest
from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores import ElasticsearchStore
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

##
# Start Elasticsearch locally
# cd tests/vector_stores/docker-compose
# docker-compose -f elasticsearch.yml up
#
# Run tests
# cd tests/vector_stores
# pytest test_elasticsearch.py


logging.basicConfig(level=logging.DEBUG)

try:
    import elasticsearch

    es_client = elasticsearch.Elasticsearch("http://localhost:9200")
    es_client.info()

    elasticsearch_not_available = False

    es_license = es_client.license.get()
    basic_license: bool = es_license["license"]["type"] == "basic"
except (ImportError, Exception):
    elasticsearch_not_available = True
    basic_license = True


@pytest.fixture(scope="session")
def conn() -> Any:
    import elasticsearch

    return elasticsearch.Elasticsearch("http://localhost:9200")


@pytest.fixture()
def index_name() -> str:
    """Return the index name."""
    return f"test_{uuid.uuid4().hex}"


@pytest.fixture(scope="session")
def elasticsearch_connection() -> Union[dict, Generator[dict, None, None]]:
    # Running this integration test with Elastic Cloud
    # Required for in-stack inference testing (ELSER + model_id)
    from elasticsearch import Elasticsearch

    es_url = os.environ.get("ES_URL", "http://localhost:9200")
    cloud_id = os.environ.get("ES_CLOUD_ID")
    es_username = os.environ.get("ES_USERNAME", "elastic")
    es_password = os.environ.get("ES_PASSWORD", "changeme")

    if cloud_id:
        yield {
            "es_cloud_id": cloud_id,
            "es_user": es_username,
            "es_password": es_password,
        }
        es = Elasticsearch(cloud_id=cloud_id, basic_auth=(es_username, es_password))

    else:
        # Running this integration test with local docker instance
        yield {
            "es_url": es_url,
        }
        es = Elasticsearch(hosts=es_url)

    # Clear all indexes
    index_names = es.indices.get(index="_all").keys()
    for index_name in index_names:
        if index_name.startswith("test_"):
            es.indices.delete(index=index_name)
    es.indices.refresh(index="_all")


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
            embedding=[0.0, 0.0, 1.0],
        ),
        TextNode(
            text="I was taught that the way of progress was neither swift nor easy.",
            id_="0b31ae71-b797-4e88-8495-031371a7752e",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="text-3")},
            metadate={
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
            metadate={
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
            metadate={
                "author": "Charlotte Bronte",
            },
            embedding=[0.0, 0.0, 0.3],
        ),
    ]


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
def test_instance_creation(index_name: str, elasticsearch_connection: Dict) -> None:
    es_store = ElasticsearchStore(
        **elasticsearch_connection,
        index_name=index_name,
    )
    assert isinstance(es_store, ElasticsearchStore)


@pytest.fixture()
def es_store(index_name: str, elasticsearch_connection: Dict) -> ElasticsearchStore:
    return ElasticsearchStore(
        **elasticsearch_connection,
        index_name=index_name,
        distance_strategy="EUCLIDEAN_DISTANCE",
    )


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
@pytest.mark.asyncio()
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


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_text_query(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await es_store.async_add(node_embeddings)
        res = await es_store.aquery(
            VectorStoreQuery(
                query_str="lorem",
                mode=VectorStoreQueryMode.TEXT_SEARCH,
                similarity_top_k=1,
            )
        )
    else:
        es_store.add(node_embeddings)
        res = es_store.query(
            VectorStoreQuery(
                query_str="lorem",
                mode=VectorStoreQueryMode.TEXT_SEARCH,
                similarity_top_k=1,
            )
        )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.skipif(
    elasticsearch_not_available,
    basic_license,
    reason="elasticsearch is not available or license is basic",
)
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_hybrid_query(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await es_store.async_add(node_embeddings)
        res = await es_store.aquery(
            VectorStoreQuery(
                query_str="lorem",
                query_embedding=[1.0, 0.0, 0.0],
                mode=VectorStoreQueryMode.HYBRID,
                similarity_top_k=1,
            )
        )
    else:
        es_store.add(node_embeddings)
        res = es_store.query(
            VectorStoreQuery(
                query_str="lorem",
                query_embedding=[1.0, 0.0, 0.0],
                mode=VectorStoreQueryMode.HYBRID,
                similarity_top_k=1,
            )
        )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
@pytest.mark.asyncio()
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


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
@pytest.mark.asyncio()
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


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
@pytest.mark.asyncio()
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


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
@pytest.mark.asyncio()
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


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_text_query_ranked(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    node1 = "0b31ae71-b797-4e88-8495-031371a7752e"
    node2 = "f658de3b-8cef-4d1c-8bed-9a263c907251"

    query_get_1_first = VectorStoreQuery(
        query_str="I was", mode=VectorStoreQueryMode.TEXT_SEARCH, similarity_top_k=2
    )
    await check_top_match(
        es_store, node_embeddings, use_async, query_get_1_first, node1, node2
    )

    query_get_2_first = VectorStoreQuery(
        query_str="I am", mode=VectorStoreQueryMode.TEXT_SEARCH, similarity_top_k=2
    )
    await check_top_match(
        es_store, node_embeddings, use_async, query_get_2_first, node2, node1
    )


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_es_and_text_query_ranked_hybrid(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    node1 = "f658de3b-8cef-4d1c-8bed-9a263c907251"
    node2 = "0b31ae71-b797-4e88-8495-031371a7752e"

    query_get_1_first = VectorStoreQuery(
        query_str="I was",
        query_embedding=[0.0, 0.0, 0.5],
        mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=2,
    )
    await check_top_match(
        es_store, node_embeddings, use_async, query_get_1_first, node1, node2
    )


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
def test_check_user_agent(
    index_name: str,
    node_embeddings: List[TextNode],
) -> None:
    from elastic_transport import AsyncTransport
    from elasticsearch import AsyncElasticsearch

    class CustomTransport(AsyncTransport):
        requests = []

        async def perform_request(self, *args, **kwargs):  # type: ignore
            self.requests.append(kwargs)
            return await super().perform_request(*args, **kwargs)

    es_client_instance = AsyncElasticsearch(
        "http://localhost:9200",
        transport_class=CustomTransport,
    )

    es_store = ElasticsearchStore(
        es_client=es_client_instance,
        index_name=index_name,
        distance_strategy="EUCLIDEAN_DISTANCE",
    )

    es_store.add(node_embeddings)

    user_agent = es_client_instance.transport.requests[0]["headers"][  # type: ignore
        "user-agent"
    ]
    pattern = r"^llama_index-py-vs/\d+\.\d+\.\d+$"
    match = re.match(pattern, user_agent)

    assert (
        match is not None
    ), f"The string '{user_agent}' does not match the expected user-agent."


async def check_top_match(
    es_store: ElasticsearchStore,
    node_embeddings: List[TextNode],
    use_async: bool,
    query: VectorStoreQuery,
    *expected_nodes: str,
) -> None:
    if use_async:
        await es_store.async_add(node_embeddings)
        res = await es_store.aquery(query)
    else:
        es_store.add(node_embeddings)
        res = es_store.query(query)
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
