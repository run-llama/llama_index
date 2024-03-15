import logging
import sys
import uuid
from typing import Any, Generator, List
import clickhouse_connect
import pytest
from llama_index.core.schema import (
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    BasePydanticVectorStore,
)
from llama_index.vector_stores.clickhouse import ClickHouseVectorStore

##
# Start ClickHouse locally
# cd llama-index-integrations/vector_stores/llama-index-vector-stores-clickhouse
# docker-compose up
#
# Run tests
# cd tests
# pytest test_clickhouse.py

logging.basicConfig(level=logging.DEBUG)

TEST_DB = "test_vector_db"
clickhouse_not_available = True

try:
    client = clickhouse_connect.get_client(
        host="localhost", port=8123, username="default", password=""
    )
    client.ping()
    clickhouse_not_available = False
except Exception:
    clickhouse_not_available = True


@pytest.fixture()
def table_name() -> str:
    """Return the table name."""
    return f"test_{uuid.uuid4().hex}"


def test_class():
    names_of_base_classes = [b.__name__ for b in ClickHouseVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.fixture(scope="session")
def clickhouse_client() -> Generator:
    import clickhouse_connect

    clickhouse_client = clickhouse_connect.get_client(
        host="localhost", port=8123, username="default", password=""
    )
    clickhouse_client.command(f"DROP DATABASE IF EXISTS {TEST_DB}")
    clickhouse_client.command(f"CREATE DATABASE {TEST_DB}")
    yield clickhouse_client
    clickhouse_client.command(f"DROP DATABASE {TEST_DB}")


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse not available")
@pytest.fixture()
def clickhouse_store(table_name: str, clickhouse_client: Any) -> ClickHouseVectorStore:
    return ClickHouseVectorStore(
        clickhouse_client,
        database=TEST_DB,
        table=table_name,
        metric="l2",
    )


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


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_instance_creation(table_name: str, clickhouse_client: Any) -> None:
    ch_store = ClickHouseVectorStore(
        clickhouse_client,
        database=TEST_DB,
        table=table_name,
    )
    assert isinstance(ch_store, ClickHouseVectorStore)


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_table_creation(table_name: str, clickhouse_client: Any) -> None:
    ch_store = ClickHouseVectorStore(
        clickhouse_client,
        database=TEST_DB,
        table=table_name,
    )
    ch_store.create_table(3)
    ch_store.drop()


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_add_to_ch_and_query(
    clickhouse_store: ClickHouseVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    clickhouse_store.add(node_embeddings)
    res = clickhouse_store.query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
    )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_add_to_ch_and_text_query(
    clickhouse_store: ClickHouseVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    clickhouse_store.add(node_embeddings)
    res = clickhouse_store.query(
        VectorStoreQuery(
            query_str="lorem",
            mode=VectorStoreQueryMode.TEXT_SEARCH,
            similarity_top_k=1,
        )
    )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
@pytest.mark.skipif(sys.platform == "darwin", reason="annoy not supported on osx")
def test_add_to_ch_and_text_query_annoy(
    clickhouse_client: Any,
    table_name: str,
    node_embeddings: List[TextNode],
) -> None:
    clickhouse_store = ClickHouseVectorStore(
        clickhouse_client,
        database=TEST_DB,
        table=table_name,
        metric="l2",
        index_type="ANNOY",
        index_params={"NumTrees": 100},
    )
    clickhouse_store.add(node_embeddings)
    res = clickhouse_store.query(
        VectorStoreQuery(
            query_str="lorem",
            mode=VectorStoreQueryMode.TEXT_SEARCH,
            similarity_top_k=1,
        )
    )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_add_to_ch_and_text_query_hnsw(
    clickhouse_client: Any,
    table_name: str,
    node_embeddings: List[TextNode],
) -> None:
    clickhouse_store = ClickHouseVectorStore(
        clickhouse_client,
        database=TEST_DB,
        table=table_name,
        metric="l2",
        index_type="HNSW",
        index_params={"ScalarKind": "f16"},
    )
    clickhouse_store.add(node_embeddings)
    res = clickhouse_store.query(
        VectorStoreQuery(
            query_str="lorem",
            mode=VectorStoreQueryMode.TEXT_SEARCH,
            similarity_top_k=1,
        )
    )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_add_to_ch_and_hybrid_query(
    clickhouse_store: ClickHouseVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    clickhouse_store.add(node_embeddings)
    res = clickhouse_store.query(
        VectorStoreQuery(
            query_str="lorem",
            query_embedding=[1.0, 0.0, 0.0],
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=1,
        )
    )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_add_to_ch_query_with_filters(
    clickhouse_store: ClickHouseVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="author", value="Stephen King")]
    )
    q = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0], similarity_top_k=10, filters=filters
    )
    clickhouse_store.add(node_embeddings)
    res = clickhouse_store.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_add_to_ch_query_with_where_filters(
    clickhouse_store: ClickHouseVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="author", value="Stephen King")]
    )
    q = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0], similarity_top_k=10, filters=filters
    )
    clickhouse_store.add(node_embeddings)
    res = clickhouse_store.query(
        q, where="JSONExtractString(metadata, 'theme')='Friendship'"
    )
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_add_to_ch_query_and_delete(
    clickhouse_store: ClickHouseVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    q = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)

    clickhouse_store.add(node_embeddings)
    res = clickhouse_store.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"

    clickhouse_store.delete("test-0")
    res = clickhouse_store.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "f658de3b-8cef-4d1c-8bed-9a263c907251"


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_add_to_ch_and_embed_query_ranked(
    clickhouse_store: ClickHouseVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    einstein_bronte_curie = [
        "bd2e080b-159a-4030-acc3-d98afd2ba49b",
        "f658de3b-8cef-4d1c-8bed-9a263c907251",
        "0b31ae71-b797-4e88-8495-031371a7752e",
    ]
    query_get_1_first = VectorStoreQuery(
        query_embedding=[0.0, 0.0, 0.5], similarity_top_k=3
    )
    clickhouse_store.add(node_embeddings)
    check_top_match(clickhouse_store, query_get_1_first, *einstein_bronte_curie)


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_add_to_ch_and_text_query_ranked(
    clickhouse_store: ClickHouseVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    node1 = "0b31ae71-b797-4e88-8495-031371a7752e"
    node2 = "f658de3b-8cef-4d1c-8bed-9a263c907251"
    clickhouse_store.add(node_embeddings)
    query_get_1_first = VectorStoreQuery(
        query_str="I was", mode=VectorStoreQueryMode.TEXT_SEARCH, similarity_top_k=2
    )
    check_top_match(clickhouse_store, query_get_1_first, node1, node2)

    query_get_2_first = VectorStoreQuery(
        query_str="I am", mode=VectorStoreQueryMode.TEXT_SEARCH, similarity_top_k=2
    )
    check_top_match(clickhouse_store, query_get_2_first, node2, node1)


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_add_to_ch_and_text_query_ranked_hybrid(
    clickhouse_store: ClickHouseVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    node1 = "0b31ae71-b797-4e88-8495-031371a7752e"
    node2 = "f658de3b-8cef-4d1c-8bed-9a263c907251"

    query_get_1_first = VectorStoreQuery(
        query_str="I was",
        query_embedding=[0.0, 0.0, 0.5],
        mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=2,
    )
    clickhouse_store.add(node_embeddings)
    check_top_match(clickhouse_store, query_get_1_first, node1, node2)


@pytest.mark.skipif(clickhouse_not_available, reason="clickhouse is not available")
def test_invalid_query_modes(
    clickhouse_store: ClickHouseVectorStore,
) -> None:
    query_sparse = VectorStoreQuery(
        query_str="I was",
        query_embedding=[0.0, 0.0, 0.5],
        mode=VectorStoreQueryMode.SPARSE,
        similarity_top_k=2,
    )
    clickhouse_store.add([])
    with pytest.raises(ValueError) as exc:
        clickhouse_store.query(query_sparse)
    assert str(exc.value) == "query mode VectorStoreQueryMode.SPARSE not supported"


def check_top_match(
    clickhouse_store: ClickHouseVectorStore,
    query: VectorStoreQuery,
    *expected_nodes: str,
) -> None:
    res = clickhouse_store.query(query)
    assert res.nodes
    # test the nodes are return in the expected order
    for i, node in enumerate(expected_nodes):
        assert res.nodes[i].node_id == node
