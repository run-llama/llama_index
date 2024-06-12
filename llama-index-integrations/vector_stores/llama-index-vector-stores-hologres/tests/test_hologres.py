from llama_index.vector_stores.hologres import HologresVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
import logging

logger = logging.getLogger(__name__)


def _get_hologres_vector_store():
    test_hologres_config = {
        "host": "<host>",
        "port": 80,
        "user": "<user>",
        "password": "<password>",
        "database": "<database>",
        "table_name": "<table_name>",
    }
    if (
        "<" in test_hologres_config["host"]
        or "<" in test_hologres_config["database"]
        or "<" in test_hologres_config["table_name"]
    ):
        return None

    return HologresVectorStore.from_param(
        host=test_hologres_config["host"],
        port=test_hologres_config["port"],
        user=test_hologres_config["user"],
        password=test_hologres_config["password"],
        database=test_hologres_config["database"],
        embedding_dimension=5,
        table_name=test_hologres_config["table_name"],
        pre_delete_table=True,
    )


def test_add_and_query_and_delete():
    vector_store = _get_hologres_vector_store()
    if not vector_store:
        logger.info("No hologres config, skipping test case!")
        return

    vectors = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]]
    texts = ["Blue", "Yellow", "Green"]

    ids = []
    nodes = []
    for i in range(len(texts)):
        node = TextNode()
        node.set_content(texts[i])
        node.embedding = vectors[i]
        node.metadata = {"id": node.node_id, "number": i, "doc_id": f"{i % 2}"}
        nodes.append(node)
        ids.append(node.node_id)

    return_ids = vector_store.add(nodes)
    assert ids == return_ids

    # Test query result numbers
    query = VectorStoreQuery(
        query_embedding=[0.1, 0.1, 0.1, 0.1, 0.1], similarity_top_k=2
    )
    response = vector_store.query(query)
    assert len(response.nodes) == 2

    # Test delete
    vector_store.delete("0")
    query = VectorStoreQuery(
        query_embedding=[0.1, 0.1, 0.1, 0.1, 0.1], similarity_top_k=2
    )
    response = vector_store.query(query)
    assert len(response.nodes) == 1
    node = response.nodes[0]
    assert node.metadata["doc_id"] == "1"
