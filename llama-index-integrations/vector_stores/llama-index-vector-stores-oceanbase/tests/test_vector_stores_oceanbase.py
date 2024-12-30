import logging
import os
from typing import List

import pytest
from llama_index.vector_stores.oceanbase import OceanBaseVectorStore

try:
    from pyobvector import ObVecClient

    OB_URI = os.getenv("OB_URI", "127.0.0.1:2881")
    OB_USER = os.getenv("OB_USER", "root@test")
    OB_PWD = os.getenv("OB_PWD", "")
    OB_DBNAME = os.getenv("OB_DBNAME", "test")

    CONN_ARGS = {
        "uri": OB_URI,
        "user": OB_USER,
        "password": OB_PWD,
        "db_name": OB_DBNAME,
    }

    # test client
    client = ObVecClient(**CONN_ARGS)

    oceanbase_available = True
except Exception as e:
    oceanbase_available = False

from llama_index.core.schema import (
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

ADA_TOKEN_COUNT = 1536

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def text_to_embedding(text: str) -> List[float]:
    """Convert text to a unique embedding using ASCII values."""
    ascii_values = [float(ord(char)) for char in text]
    # Pad or trim the list to make it of length ADA_TOKEN_COUNT
    return ascii_values[:ADA_TOKEN_COUNT] + [0.0] * (
        ADA_TOKEN_COUNT - len(ascii_values)
    )


@pytest.fixture(scope="session")
def node_embeddings() -> list[TextNode]:
    """Return a list of TextNodes with embeddings."""
    return [
        TextNode(
            text="foo",
            id_="ffddfb6b-2cad-48ec-917e-6a7dfaba3c9e",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "theme": "FOO",
                "location": 1,
            },
            embedding=text_to_embedding("foo"),
        ),
        TextNode(
            text="bar",
            id_="6f74bc29-8d84-4c3c-b458-e6a082cf1938",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "theme": "BAR",
                "location": 2,
            },
            embedding=text_to_embedding("bar"),
        ),
        TextNode(
            text="baz",
            id_="e99b4f5f-5bc5-4cff-8c4e-ee1d66479a11",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "theme": "BAZ",
                "location": 3,
            },
            embedding=text_to_embedding("baz"),
        ),
    ]


def test_class():
    names_of_base_classes = [b.__name__ for b in OceanBaseVectorStore.__mro__]
    assert OceanBaseVectorStore.__name__ in names_of_base_classes


@pytest.mark.skipif(not oceanbase_available, reason="oceanbase is not available")
def test_init_client():
    client = ObVecClient(**CONN_ARGS)
    client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")

    # Initialize OceanBaseVectorStore
    oceanbase = OceanBaseVectorStore(
        client=client,
        dim=1024,
    )


@pytest.mark.skipif(not oceanbase_available, reason="oceanbase is not available")
def test_add_node(node_embeddings: List[TextNode]):
    client = ObVecClient(**CONN_ARGS)
    client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")

    # Initialize OceanBaseVectorStore
    oceanbase = OceanBaseVectorStore(
        client=client,
        dim=1536,
        drop_old=True,
        normalize=True,
    )

    oceanbase.add(node_embeddings)


@pytest.mark.skipif(not oceanbase_available, reason="oceanbase is not available")
def test_search_with_l2_distance(node_embeddings: List[TextNode]):
    client = ObVecClient(**CONN_ARGS)
    client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")

    # Initialize OceanBaseVectorStore
    oceanbase = OceanBaseVectorStore(
        client=client,
        dim=1536,
        drop_old=True,
        normalize=True,
    )

    oceanbase.add(node_embeddings)

    q = VectorStoreQuery(query_embedding=text_to_embedding("foo"), similarity_top_k=1)

    result = oceanbase.query(q)
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
    )
    assert result.similarities is not None and result.similarities[0] == 1.0
    assert result.ids is not None and result.ids[0] == node_embeddings[0].node_id


@pytest.mark.skipif(not oceanbase_available, reason="oceanbase is not available")
def test_search_with_neg_ip_distance(node_embeddings: List[TextNode]):
    client = ObVecClient(**CONN_ARGS)
    client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")

    # Initialize OceanBaseVectorStore
    oceanbase = OceanBaseVectorStore(
        client=client,
        dim=1536,
        drop_old=True,
        normalize=True,
        vidx_metric_type="inner_product",
    )

    oceanbase.add(node_embeddings)

    q = VectorStoreQuery(query_embedding=text_to_embedding("foo"), similarity_top_k=1)

    result = oceanbase.query(q)
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
    )
    assert (
        result.similarities is not None and result.similarities[0] == 0.9999999701976776
    )
    assert result.ids is not None and result.ids[0] == node_embeddings[0].node_id


@pytest.mark.skipif(not oceanbase_available, reason="oceanbase is not available")
def test_delete_doc(node_embeddings: List[TextNode]):
    client = ObVecClient(
        **CONN_ARGS,
    )
    client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")

    # Initialize OceanBaseVectorStore
    oceanbase = OceanBaseVectorStore(
        client=client,
        dim=1536,
        drop_old=True,
        normalize=True,
    )

    oceanbase.add(node_embeddings)

    q = VectorStoreQuery(query_embedding=text_to_embedding("foo"), similarity_top_k=3)

    result = oceanbase.query(q)
    assert result.nodes is not None and len(result.nodes) == 3
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
    )
    assert result.similarities is not None and result.similarities[0] == 1.0
    assert result.ids is not None and result.ids[0] == node_embeddings[0].node_id

    oceanbase.delete(ref_doc_id="test-1")

    q = VectorStoreQuery(query_embedding=text_to_embedding("foo"), similarity_top_k=3)

    result = oceanbase.query(q)
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[2].text
    )
    assert (
        result.similarities is not None and result.similarities[0] == 0.9315029757824523
    )
    assert result.ids is not None and result.ids[0] == node_embeddings[2].node_id


@pytest.mark.skipif(not oceanbase_available, reason="oceanbase is not available")
def test_delete_nodes_and_get_nodes(node_embeddings: List[TextNode]):
    client = ObVecClient(
        **CONN_ARGS,
    )
    client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")

    # Initialize OceanBaseVectorStore
    oceanbase = OceanBaseVectorStore(
        client=client,
        dim=1536,
        drop_old=True,
        normalize=True,
    )

    oceanbase.add(node_embeddings)

    result = oceanbase.get_nodes()
    assert len(result) == 3

    result = oceanbase.get_nodes(
        node_ids=[
            node_embeddings[1].id_,
            node_embeddings[2].id_,
        ],
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=2, operator=">"),
            ]
        ),
    )
    assert len(result) == 1

    oceanbase.delete_nodes(
        node_ids=[
            node_embeddings[1].id_,
            node_embeddings[2].id_,
        ],
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=2, operator=">"),
            ]
        ),
    )
    result = oceanbase.get_nodes(
        node_ids=[
            node_embeddings[0].id_,
            node_embeddings[1].id_,
            node_embeddings[2].id_,
        ],
    )
    assert len(result) == 2
    assert (
        result[0].id_ == node_embeddings[1].id_
        and result[1].id_ == node_embeddings[0].id_
    )


@pytest.mark.skipif(not oceanbase_available, reason="oceanbase is not available")
def test_clear(node_embeddings: List[TextNode]):
    client = ObVecClient(
        **CONN_ARGS,
    )
    client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")

    # Initialize OceanBaseVectorStore
    oceanbase = OceanBaseVectorStore(
        client=client,
        dim=1536,
        drop_old=True,
        normalize=True,
    )

    oceanbase.add(node_embeddings)
    oceanbase.clear()


@pytest.mark.skipif(not oceanbase_available, reason="oceanbase is not available")
def test_search_with_filter(node_embeddings: List[TextNode]):
    client = ObVecClient(
        **CONN_ARGS,
    )
    client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")

    # Initialize OceanBaseVectorStore
    oceanbase = OceanBaseVectorStore(
        client=client,
        dim=1536,
        drop_old=True,
        normalize=True,
    )

    oceanbase.add(node_embeddings)

    q = VectorStoreQuery(
        query_embedding=text_to_embedding("foo"),
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=2, operator=">="),
            ]
        ),
    )

    result = oceanbase.query(q)
    assert result.nodes is not None and len(result.nodes) == 2
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[1].text
        and result.nodes[1].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[2].text
    )

    q = VectorStoreQuery(
        query_embedding=text_to_embedding("foo"),
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=2, operator=">="),
                MetadataFilter(key="theme", value="BAZ", operator="=="),
            ],
            condition="and",
        ),
    )

    result = oceanbase.query(q)
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[2].text
    )

    q = VectorStoreQuery(
        query_embedding=text_to_embedding("foo"),
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=2, operator=">="),
                MetadataFilter(key="theme", value="FOO", operator="=="),
            ],
            condition="or",
        ),
    )

    result = oceanbase.query(q)
    assert result.nodes is not None and len(result.nodes) == 3
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
        and result.nodes[1].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[1].text
        and result.nodes[2].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[2].text
    )
