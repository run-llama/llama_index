import unittest

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.astra import AstraDBVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery

import astrapy

print(f"astrapy detected: {astrapy.__version__}")


def get_astra_db_store() -> AstraDBVectorStore:
    return AstraDBVectorStore(
        token="AstraCS:<...>",
        api_endpoint="https://<...>",
        collection_name="test_collection",
        embedding_dimension=2,
        namespace="default_keyspace",
        ttl_seconds=123,
    )


class TestAstraDBVectorStore(unittest.TestCase):
    def test_astra_db_create_and_crud(self) -> None:
        vector_store = get_astra_db_store()

        vector_store.add(
            [
                TextNode(
                    text="test node text",
                    id_="test node id",
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc id")
                    },
                    embedding=[0.5, 0.5],
                )
            ]
        )

        vector_store.delete("test node id")

        vector_store.client

    def test_astra_db_queries(self) -> None:
        vector_store = get_astra_db_store()

        query = VectorStoreQuery(query_embedding=[1, 1], similarity_top_k=3)

        vector_store.query(
            query,
        )
