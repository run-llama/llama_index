import logging
import os
import uuid
from typing import Any, Dict, Generator, List, Union

import pytest

from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores import ElasticsearchStore
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    NodeWithEmbedding,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

# Install Elasticsearch via https://github.com/elastic/elasticsearch-labs/blob/main/developer-guide.md#running-elasticsearch

logging.basicConfig(level=logging.DEBUG)

try:
    import elasticsearch  # noqa: F401

    es_client = elasticsearch.Elasticsearch("http://localhost:9200")
    es_client.info()

    elasticsearch_not_available = False
except (ImportError, Exception):
    elasticsearch_not_available = True


@pytest.fixture(scope="session")
def conn() -> Any:
    import elasticsearch  # noqa: F401

    es_client = elasticsearch.Elasticsearch("http://localhost:9200")

    return es_client


@pytest.fixture(scope="function")
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
def node_embeddings() -> List[NodeWithEmbedding]:
    return [
        NodeWithEmbedding(
            embedding=[1.0, 0.0, 0.0],
            node=TextNode(
                text="lorem ipsum",
                id_="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")
                },
                metadata={
                    "author": "Stephen King",
                    "theme": "Friendship",
                },
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0, 1.0, 0.0],
            node=TextNode(
                text="lorem ipsum",
                id_="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")
                },
                metadata={
                    "director": "Francis Ford Coppola",
                    "theme": "Mafia",
                },
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0, 0.0, 1.0],
            node=TextNode(
                text="lorem ipsum",
                id_="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")
                },
                metadata={
                    "director": "Christopher Nolan",
                },
            ),
        ),
    ]


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
def test_instance_creation(index_name, elasticsearch_connection) -> None:
    es_store = ElasticsearchStore(
        **elasticsearch_connection,
        index_name=index_name,
    )
    assert isinstance(es_store, ElasticsearchStore)


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
def test_add_to_es_and_query(
    index_name, elasticsearch_connection, node_embeddings: List[NodeWithEmbedding]
) -> None:
    es_store = ElasticsearchStore(
        **elasticsearch_connection,
        index_name=index_name,
        distance_strategy="COSINE",
    )
    es_store.add(node_embeddings)
    res = es_store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
    )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
def test_add_to_es_and_text_query(
    index_name, elasticsearch_connection, node_embeddings: List[NodeWithEmbedding]
) -> None:
    es_store = ElasticsearchStore(
        **elasticsearch_connection,
        index_name=index_name,
        distance_strategy="COSINE",
    )
    es_store.add(node_embeddings)
    res = es_store.query(
        VectorStoreQuery(
            query_str="lorem", mode=VectorStoreQueryMode.TEXT_SEARCH, similarity_top_k=1
        )
    )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
def test_add_to_es_and_hybrid_query(
    index_name, elasticsearch_connection, node_embeddings: List[NodeWithEmbedding]
) -> None:
    es_store = ElasticsearchStore(
        **elasticsearch_connection,
        index_name=index_name,
        distance_strategy="COSINE",
    )
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
def test_add_to_es_query_with_filters(
    index_name, elasticsearch_connection, node_embeddings: List[NodeWithEmbedding]
) -> None:
    es_store = ElasticsearchStore(
        **elasticsearch_connection,
        index_name=index_name,
        distance_strategy="COSINE",
    )

    es_store.add(node_embeddings)

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="author", value="Stephen King")]
    )
    q = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0], similarity_top_k=10, filters=filters
    )

    res = es_store.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"


def test_add_to_es_query_with_es_filters(
    index_name, elasticsearch_connection, node_embeddings: List[NodeWithEmbedding]
) -> None:
    es_store = ElasticsearchStore(
        **elasticsearch_connection,
        index_name=index_name,
        distance_strategy="COSINE",
    )

    es_store.add(node_embeddings)

    q = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=10)

    res = es_store.query(q, es_filter={"wildcard": {"metadata.author": "stephe*"}})
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"


@pytest.mark.skipif(
    elasticsearch_not_available, reason="elasticsearch is not available"
)
def test_add_to_es_query_and_delete(
    index_name, elasticsearch_connection, node_embeddings: List[NodeWithEmbedding]
) -> None:
    es_store = ElasticsearchStore(
        **elasticsearch_connection,
        index_name=index_name,
        distance_strategy="COSINE",
    )

    q = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)

    es_store.add(node_embeddings)
    res = es_store.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"
    es_store.delete("test-0")
    res = es_store.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d"
