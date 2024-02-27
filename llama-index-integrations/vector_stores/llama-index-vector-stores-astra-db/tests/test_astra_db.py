import os
import pytest
from typing import Iterable

import astrapy
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.astra_db import AstraDBVectorStore

print(f"astrapy detected: {astrapy.__version__}")


# env variables
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT", "")


@pytest.fixture(scope="module")
def astra_db_store() -> Iterable[AstraDBVectorStore]:
    store = AstraDBVectorStore(
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name="test_collection",
        embedding_dimension=2,
    )
    yield store

    store._astra_db.delete_collection("test_collection")


@pytest.mark.skipif(
    ASTRA_DB_APPLICATION_TOKEN == "" or ASTRA_DB_API_ENDPOINT == "",
    reason="missing Astra DB credentials",
)
def test_astra_db_create_and_crud(astra_db_store: AstraDBVectorStore) -> None:
    astra_db_store.add(
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

    astra_db_store.delete("test node id")


@pytest.mark.skipif(
    ASTRA_DB_APPLICATION_TOKEN == "" or ASTRA_DB_API_ENDPOINT == "",
    reason="missing Astra DB credentials",
)
def test_astra_db_queries(astra_db_store: AstraDBVectorStore) -> None:
    query = VectorStoreQuery(query_embedding=[1, 1], similarity_top_k=3)

    astra_db_store.query(
        query,
    )
