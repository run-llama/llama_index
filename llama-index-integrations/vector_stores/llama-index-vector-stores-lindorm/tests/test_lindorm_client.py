import pytest
import random
import string
import logging
from llama_index.core.schema import (
    TextNode,
    RelatedNodeInfo,
    NodeRelationship,
)
from llama_index.vector_stores.lindorm import (
    LindormVectorStore,
    LindormVectorClient,
)
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)

logger = logging.getLogger(__name__)


def _get_lindorm_vector_store():
    # Lindorm instance info, please replace with your own
    host = "<ld-bp******jm*******-proxy-search-pub.lindorm.aliyuncs.com>"
    port = 30070
    username = "<your username>"
    password = "<your password>"
    index_name = "<lindorm_pytest_index>"
    nprobe = "2"
    reorder_factor = "10"

    # Check if placeholder values exist, skip if they do
    if "<" in host or "<" in username or "<" in password or "<" in index_name:
        return None

    # Create a client and vector store instance
    client = LindormVectorClient(
        host=host,
        port=port,
        username=username,
        password=password,
        index=index_name,
        dimension=5,
        nprobe=nprobe,
        reorder_factor=reorder_factor,
    )
    return LindormVectorStore(client)


@pytest.fixture(scope="module")
def vector_store():
    store = _get_lindorm_vector_store()
    if not store:
        pytest.skip("No Lindorm config, skipping test case!")
    return store


@pytest.fixture(scope="session")
def nodes():
    nodes = []
    for i in range(1000):
        vector = [random.random() for _ in range(5)]
        characters = string.ascii_letters + string.digits
        random_string = "".join(random.choices(characters, k=5))
        new_node = TextNode(
            embedding=vector,
            text=random_string + " " + str(i),
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-" + str(i))
            },
            metadata={
                "author": "test " + random_string,
                "mark_id": i,
            },
        )
        nodes.append(new_node)
    return nodes


def test_add_nodes(vector_store, nodes):
    added_ids = vector_store.add(nodes)
    assert len(added_ids) == len(nodes)
    assert all(id_ for id_ in added_ids)


def test_simple_query(vector_store):
    query_embedding = [1.0, 1.0, 1.0, 1.0, 1.0]
    simple_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5)
    result = vector_store.query(simple_query)
    assert len(result.nodes) > 0


def test_query_with_metadata_filter(vector_store):
    query_embedding = [1.0, 1.0, 1.0, 1.0, 1.0]
    filter1 = MetadataFilter(key="mark_id", value=0, operator=FilterOperator.GTE)
    filter2 = MetadataFilter(key="mark_id", value=500, operator=FilterOperator.LTE)
    filters = MetadataFilters(filters=[filter1, filter2], condition=FilterCondition.AND)
    filter_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=5, filters=filters
    )
    result = vector_store.query(filter_query)
    assert len(result.nodes) > 0


def test_lexical_query(vector_store):
    query_embedding = [1.0, 1.0, 1.0, 1.0, 1.0]
    lexical_query = VectorStoreQuery(
        mode=VectorStoreQueryMode.TEXT_SEARCH,
        query_embedding=query_embedding,
        similarity_top_k=5,
        # your query str match the field "content"(text you stored in),
        # and note the minimum search granularity of query str is one token.
        query_str="your query str",
    )
    result = vector_store.query(lexical_query)
    assert len(result.nodes) > 0


def test_hybrid_query(vector_store):
    query_embedding = [1.0, 1.0, 1.0, 1.0, 1.0]
    hybrid_query = VectorStoreQuery(
        mode=VectorStoreQueryMode.HYBRID,
        query_embedding=query_embedding,
        similarity_top_k=5,
        # your query str match the field "content"(text you stored in),
        # and note the minimum search granularity of query str is one token.
        query_str="your query str",
    )
    result = vector_store.query(hybrid_query)
    assert len(result.nodes) > 0


def test_delete_node(vector_store):
    vector_store.delete(ref_doc_id="test-0")
    query_embedding = [1.0, 1.0, 1.0, 1.0, 1.0]
    filter_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="relationships.SOURCE.node_id",
                    value="test-0",
                    operator=FilterOperator.EQ,
                )
            ],
            condition=FilterCondition.AND,
        ),
    )
    result = vector_store.query(filter_query)
    assert len(result.nodes) == 0
