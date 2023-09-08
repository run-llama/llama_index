from importlib.util import find_spec
from typing import Generator, List

import pytest

try:
    find_spec("pymilvus")
    from milvus import default_server

    milvus_libs = 1
except ImportError:
    milvus_libs = None  # type: ignore

from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores import MilvusVectorStore
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
)


@pytest.fixture
def embedded_milvus() -> Generator:
    default_server.cleanup()
    default_server.start()
    yield "http://" + str(default_server.server_address) + ":" + str(
        default_server.listen_port
    )
    default_server.stop()
    default_server.cleanup()


@pytest.fixture
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
            embedding=[1.0, 1.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
            embedding=[2.0, 2.0],
        ),
    ]


@pytest.mark.skipif(milvus_libs is None, reason="Missing milvus packages")
def test_add_stores_data(node_embeddings: List[TextNode], embedded_milvus: str) -> None:
    milvus_store = MilvusVectorStore(dim=2, uri=embedded_milvus, collection_name="test")

    milvus_store.add(node_embeddings)
    milvus_store.milvusclient.flush("test")
    assert milvus_store.client.num_entities("test") == 2


@pytest.mark.skipif(milvus_libs is None, reason="Missing milvus packages")
def test_search_data(node_embeddings: List[TextNode], embedded_milvus: str) -> None:
    milvus_store = MilvusVectorStore(dim=2, uri=embedded_milvus, collection_name="test")
    milvus_store.add(node_embeddings)

    res = milvus_store.query(
        VectorStoreQuery(query_embedding=[3, 3], similarity_top_k=1)
    )
    assert res.ids is not None and res.ids[0] == "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d"
    assert res.nodes is not None and res.nodes[0].metadata["theme"] == "Mafia"


@pytest.mark.skipif(milvus_libs is None, reason="Missing milvus packages")
def test_search_data_filter(
    node_embeddings: List[TextNode], embedded_milvus: str
) -> None:
    milvus_store = MilvusVectorStore(dim=2, uri=embedded_milvus, collection_name="test")
    milvus_store.add(node_embeddings)

    res = milvus_store.query(
        VectorStoreQuery(
            query_embedding=[3, 3],
            similarity_top_k=1,
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key="theme", value="Friendship")]
            ),
        )
    )

    assert res.ids is not None and res.ids[0] == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"
    assert res.nodes is not None and res.nodes[0].metadata["theme"] == "Friendship"

    print(node_embeddings[0].node_id)
    res = milvus_store.query(
        VectorStoreQuery(
            query_embedding=[3, 3],
            node_ids=["c330d77f-90bd-4c51-9ed2-57d8d693b3b0"],
            similarity_top_k=1,
        )
    )
    assert res.ids is not None and res.ids[0] == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"
    assert res.nodes is not None and res.nodes[0].metadata["theme"] == "Friendship"

    res = milvus_store.query(
        VectorStoreQuery(
            query_embedding=[3, 3],
            doc_ids=["test-0"],
            similarity_top_k=1,
        )
    )
    assert res.ids is not None and res.ids[0] == "c330d77f-90bd-4c51-9ed2-57d8d693b3b0"
    assert res.nodes is not None and res.nodes[0].metadata["theme"] == "Friendship"
