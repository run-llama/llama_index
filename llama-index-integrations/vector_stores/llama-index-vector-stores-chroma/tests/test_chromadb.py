import os
from typing import Dict, List, Generator

import pytest
from unittest import mock
from llama_index.vector_stores.chroma.base import MAX_CHUNK_SIZE
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)


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

    http_client_chromadb_mode = True
except (ImportError, Exception):
    http_client_chromadb_mode = False


# To test chromadb http-client functionality do:
# cd tests
# docker-compose up
#
@pytest.mark.skipif(
    http_client_chromadb_mode is False,
    reason="chromadb is not running in http client mode",
)
def test_instance_creation_from_http_params() -> None:
    store = ChromaVectorStore.from_params(
        host=PARAMS["host"],
        port=PARAMS["port"],
        collection_name=COLLECTION_NAME,
        collection_kwargs={},
    )
    assert isinstance(store, ChromaVectorStore)


def test_instance_creation_from_collection() -> None:
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    store = ChromaVectorStore.from_collection(collection)
    assert isinstance(store, ChromaVectorStore)


def test_instance_creation_from_persist_dir() -> None:
    store = ChromaVectorStore.from_params(
        persist_dir="./data",
        collection_name=COLLECTION_NAME,
        collection_kwargs={},
    )
    assert isinstance(store, ChromaVectorStore)


@pytest.fixture()
def vector_store() -> Generator[ChromaVectorStore, None, None]:
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    yield ChromaVectorStore(chroma_collection=collection)
    chroma_client.delete_collection(name=COLLECTION_NAME)


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
            metadata={
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
            metadata={
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
            metadata={
                "author": "Charlotte Bronte",
            },
            embedding=[0.0, 0.0, 0.3],
        ),
    ]


@pytest.mark.asyncio
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


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_chromadb_and_query_by_metafilters_only(
    vector_store: ChromaVectorStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="author", value="Marie Curie", operator=FilterOperator.EQ
            )
        ],
        condition=FilterCondition.AND,
    )

    if use_async:
        await vector_store.async_add(node_embeddings)
        res = await vector_store.aquery(
            VectorStoreQuery(filters=filters, similarity_top_k=1)
        )
    else:
        vector_store.add(node_embeddings)
        res = vector_store.query(VectorStoreQuery(filters=filters, similarity_top_k=1))

    assert (
        res.nodes[0].get_content()
        == "I was taught that the way of progress was neither swift nor easy."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_chromadb_and_query_by_nested_metadata_filters(
    vector_store: ChromaVectorStore,
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="author", value="Stephen King", operator=FilterOperator.EQ
            ),
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="theme", value="Mafia", operator=FilterOperator.EQ
                    ),
                    MetadataFilter(
                        key="theme", value="Friendship", operator=FilterOperator.EQ
                    ),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )

    if use_async:
        await vector_store.async_add(node_embeddings)
        res = await vector_store.aquery(
            VectorStoreQuery(filters=filters, similarity_top_k=1)
        )
    else:
        vector_store.add(node_embeddings)
        res = vector_store.query(VectorStoreQuery(filters=filters, similarity_top_k=1))

    assert res.nodes[0].get_content() == "lorem ipsum"


def test_get_nodes(
    vector_store: ChromaVectorStore, node_embeddings: List[TextNode]
) -> None:
    vector_store.add(node_embeddings)
    res = vector_store.get_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        ]
    )
    assert len(res) == 3
    assert res[0].get_content() == "lorem ipsum"
    assert res[1].get_content() == "lorem ipsum"
    assert res[2].get_content() == "lorem ipsum"


def test_delete_nodes(
    vector_store: ChromaVectorStore, node_embeddings: List[TextNode]
) -> None:
    vector_store.add(node_embeddings)
    vector_store.delete_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
        ]
    )
    res = vector_store.get_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        ]
    )
    assert len(res) == 1
    assert res[0].get_content() == "lorem ipsum"
    assert res[0].id_ == "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d"


def test_clear(
    vector_store: ChromaVectorStore, node_embeddings: List[TextNode]
) -> None:
    vector_store.add(node_embeddings)
    vector_store.clear()
    res = vector_store.get_nodes(
        node_ids=[
            "c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            "c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            "c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
        ]
    )
    assert len(res) == 0


class SimpleCollection:
    """Minimal collection stub without _client attribute for testing fallback behavior."""


@mock.patch("llama_index.vector_stores.chroma.base.chromadb.HttpClient")
def test_max_chunk_size_new_client(mock_client_cls: mock.Mock) -> None:
    # Case 1: New client created (mocked)
    mock_client = mock.Mock()
    mock_client.get_max_batch_size.return_value = 100
    mock_client.get_or_create_collection.return_value = mock.Mock()
    mock_client_cls.return_value = mock_client

    store = ChromaVectorStore(
        chroma_collection=None,
        host="localhost",
        port=8000,
        collection_name="test_collection",
    )
    assert store._collection is not None
    assert store.max_chunk_size == 100


def test_max_chunk_size_collection_with_client() -> None:
    # Case 2: Collection passed, has _client
    mock_collection = mock.Mock()
    mock_collection._client.get_max_batch_size.return_value = 200
    store = ChromaVectorStore(chroma_collection=mock_collection)
    assert store.max_chunk_size == 200


def test_max_chunk_size_collection_no_client() -> None:
    # Case 3: Collection passed, no _client (fallback)
    store = ChromaVectorStore(chroma_collection=SimpleCollection())
    assert store.max_chunk_size == MAX_CHUNK_SIZE


def test_max_chunk_size_exception() -> None:
    # Case 4: Exception during get_max_batch_size (fallback)
    mock_collection_error = mock.Mock()
    mock_collection_error._client.get_max_batch_size.side_effect = Exception("Error")
    store = ChromaVectorStore(chroma_collection=mock_collection_error)
    assert store.max_chunk_size == MAX_CHUNK_SIZE
