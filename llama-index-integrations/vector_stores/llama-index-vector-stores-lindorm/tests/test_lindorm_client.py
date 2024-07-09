import pytest
import random
import string
from llama_index.core.schema import (
    TextNode,
    RelatedNodeInfo,
    NodeRelationship,
)
from llama_index.vector_stores.lindorm import (
    LindormSearchVectorStore,
    LindormSearchVectorClient,
)
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)

@pytest.fixture(scope="module")
def vector_store():
    # Lindorm instance info
    # how to obtain an lindorm instance:
    # https://alibabacloud.com/help/en/lindorm/latest/create-an-instance?spm=a2c63.l28256.0.0.4cc0f53cUfKOxI 

    # how to access your lindorm instance:
    # https://www.alibabacloud.com/help/en/lindorm/latest/view-endpoints?spm=a2c63.p38356.0.0.37121bcdxsDvbN

    # run curl commands to connect to and use LindormSearch:
    # https://www.alibabacloud.com/help/en/lindorm/latest/connect-and-use-the-search-engine-with-the-curl-command
    host = "ld-bp******jm*******-proxy-search-pub.lindorm.aliyuncs.com"
    port = 30070
    username = 'your username'
    password = 'your password'
    index_name = "lindorm_pytest_index"
    
    # Create a client and vector store instance
    client = LindormSearchVectorClient(
        host=host,
        port=port,
        username=username,
        password=password,
        index=index_name,
        dimension=5,
    )
    vector_store = LindormSearchVectorStore(client)
    
    yield vector_store
    
    # Teardown: delete index and close client
    client._os_client.indices.delete(index=index_name)
    client._os_client.close()

@pytest.fixture(scope="session")
def nodes():
    nodes = []
    for i in range(0, 1000):
        vector = [random.random() for _ in range(5)]
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k=5))
        new_node = TextNode(
            embedding=vector,
            text=random_string + " " + str(i),
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-" + str(i))},
            metadata={
                "author": "test " + random_string,
                "mark_id": i,
            }
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
    filter1 = MetadataFilter(key="metadata.mark_id", value=0, operator=FilterOperator.GTE)
    filter2 = MetadataFilter(key="metadata.mark_id", value=500, operator=FilterOperator.LTE)
    filters = MetadataFilters(filters=[filter1, filter2], condition=FilterCondition.AND)
    filter_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5, filters=filters)
    result = vector_store.query(filter_query)
    assert len(result.nodes) > 0

def test_lexical_query(vector_store):
    query_embedding = [1.0, 1.0, 1.0, 1.0, 1.0]
    lexical_query = VectorStoreQuery(
        mode=VectorStoreQueryMode.TEXT_SEARCH,
        query_embedding=query_embedding,
        similarity_top_k=5,
        # your query str match the field "content"(text you stored in),
        # and note the the minimum search granularity of query str is one token.
        query_str="your query str"
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
        # and note the the minimum search granularity of query str is one token.
        query_str="your query str"
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
            filters=[MetadataFilter(key="relationships.SOURCE.node_id", value="test-0", operator=FilterOperator.EQ)],
            condition=FilterCondition.AND 
        )
    )
    result = vector_store.query(filter_query)
    assert len(result.nodes) == 0
