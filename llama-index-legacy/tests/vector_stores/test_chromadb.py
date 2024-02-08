import os
from typing import Dict, List

import pytest
from llama_index.legacy.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.legacy.vector_stores import ChromaVectorStore
from llama_index.legacy.vector_stores.types import VectorStoreQuery

##
# Start chromadb locally
# cd tests
# docker-compose up
#
# Run tests
# cd tests/vector_stores
# pytest test_chromadb.py


PARAMS: Dict[str, str] = {
    "host": os.environ.get("CHROMADB_HOST", "localhost"),
    "port": os.environ.get("CHROMADB_PORT", "8000"),
}
COLLECTION_NAME = "llama_collection"

try:
    import chromadb

    # connection check
    conn__ = chromadb.HttpClient(**PARAMS)  # type: ignore
    conn__.get_or_create_collection(COLLECTION_NAME)

    chromadb_not_available = False
except (ImportError, Exception):
    chromadb_not_available = True


@pytest.mark.skipif(chromadb_not_available, reason="chromadb is not available")
def test_instance_creation_from_collection() -> None:
    connection = chromadb.HttpClient(**PARAMS)
    collection = connection.get_collection(COLLECTION_NAME)
    store = ChromaVectorStore.from_collection(collection)
    assert isinstance(store, ChromaVectorStore)


@pytest.mark.skipif(chromadb_not_available, reason="chromadb is not available")
def test_instance_creation_from_http_params() -> None:
    store = ChromaVectorStore.from_params(
        host=PARAMS["host"],
        port=PARAMS["port"],
        collection_name=COLLECTION_NAME,
        collection_kwargs={},
    )
    assert isinstance(store, ChromaVectorStore)


@pytest.mark.skipif(chromadb_not_available, reason="chromadb is not available")
def test_instance_creation_from_persist_dir() -> None:
    store = ChromaVectorStore.from_params(
        persist_dir="./data",
        collection_name=COLLECTION_NAME,
        collection_kwargs={},
    )
    assert isinstance(store, ChromaVectorStore)


@pytest.fixture()
def vector_store() -> ChromaVectorStore:
    connection = chromadb.HttpClient(**PARAMS)
    collection = connection.get_collection(COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=collection)


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


@pytest.mark.skipif(chromadb_not_available, reason="chromadb is not available")
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_chromadb_and_query(
    vector_store: ChromaVectorStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await vector_store.async_add(node_embeddings)
        res = await vector_store.aquery(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )
    else:
        vector_store.add(node_embeddings)
        res = vector_store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
        )
    assert res.nodes
    assert res.nodes[0].get_content() == "lorem ipsum"
