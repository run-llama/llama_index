"""Integration tests for the Actian Vector AI vector store."""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator

import pytest

from llama_index.core import MockEmbedding, StorageContext, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.schema import Document
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.actian_vectorai import ActianVectorAIVectorStore


VECTORAI_SERVER_URL = os.getenv("VECTORAI_SERVER_URL", "localhost:6574")
EMBED_DIM = 128


@contextmanager
def _managed_vector_store() -> Iterator[ActianVectorAIVectorStore]:
    """Create an isolated collection and ensure it is cleaned up."""
    collection_name = f"test_integration_collection_{uuid.uuid4().hex}"
    with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL,
        collection_name=collection_name,
    ) as vector_store:
        try:
            yield vector_store
        finally:
            if vector_store.client.collections.exists(collection_name):
                vector_store.client.collections.delete(collection_name)


@asynccontextmanager
async def _amanaged_vector_store() -> AsyncIterator[ActianVectorAIVectorStore]:
    """Async variant of vector store setup with collection cleanup."""
    collection_name = f"test_integration_collection_{uuid.uuid4().hex}"
    async with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL,
        collection_name=collection_name,
    ) as vector_store:
        try:
            yield vector_store
        finally:
            if await vector_store.async_client.collections.exists(collection_name):
                await vector_store.async_client.collections.delete(collection_name)


@pytest.fixture()
def documents() -> list[Document]:
    return [
        Document(
            text="LlamaIndex powers retrieval-augmented generation pipelines.",
            metadata={"topic": "rag", "doc_key": "doc_1"},
            doc_id="doc_1",
        ),
        Document(
            text="Actian Vector AI can store and search dense embeddings.",
            metadata={"topic": "vector-db", "doc_key": "doc_2"},
            doc_id="doc_2",
        ),
        Document(
            text="Metadata filters help constrain vector search results.",
            metadata={"topic": "filters", "doc_key": "doc_3"},
            doc_id="doc_3",
        ),
    ]


def _assert_retrieval_with_retries(
    retriever,
    query_text: str,
    expected_topic: str,
    similarity_top_k: int = 3,
    max_retries: int = 5,
) -> None:
    for attempt in range(max_retries):
        result_nodes = retriever.retrieve(query_text)
        top_nodes = result_nodes[:similarity_top_k]
        if any(node.metadata.get("topic") == expected_topic for node in top_nodes):
            return
        if attempt < max_retries - 1:
            time.sleep(1)

    raise AssertionError(
        f"Expected top retrieval topic '{expected_topic}' for query '{query_text}' "
        f"after {max_retries} attempts."
    )


async def _aassert_retrieval_with_retries(
    retriever,
    query_text: str,
    expected_topic: str,
    similarity_top_k: int = 3,
    max_retries: int = 5,
) -> None:
    for attempt in range(max_retries):
        result_nodes = await retriever.aretrieve(query_text)
        top_nodes = result_nodes[:similarity_top_k]
        if any(node.metadata.get("topic") == expected_topic for node in top_nodes):
            return
        if attempt < max_retries - 1:
            await asyncio.sleep(1)

    raise AssertionError(
        f"Expected top retrieval topic '{expected_topic}' for query '{query_text}' "
        f"after {max_retries} attempts."
    )


def test_actian_connection() -> None:
    with _managed_vector_store() as vector_store:
        info = vector_store.client.health_check()
        assert info["title"] == "Actian VectorAI DB"


def test_index_retriever_and_query_engine(documents: list[Document]) -> None:
    with _managed_vector_store() as vector_store:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=MockEmbedding(embed_dim=EMBED_DIM),
            storage_context=storage_context,
        )

        retriever = index.as_retriever(similarity_top_k=3)
        query_engine = index.as_query_engine(llm=MockLLM(), similarity_top_k=3)
        query_text = documents[1].text

        _assert_retrieval_with_retries(
            retriever,
            query_text,
            expected_topic="vector-db",
        )

        response = query_engine.query(query_text)
        assert any(
            node.metadata.get("topic") == "vector-db" for node in response.source_nodes
        )


def test_metadata_filtered_retrieval_and_query_engine(
    documents: list[Document],
) -> None:
    with _managed_vector_store() as vector_store:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=MockEmbedding(embed_dim=EMBED_DIM),
            storage_context=storage_context,
        )

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="topic",
                    operator=FilterOperator.EQ,
                    value="vector-db",
                )
            ]
        )

        retriever = index.as_retriever(
            similarity_top_k=3,
            filters=filters,
        )
        query_engine = index.as_query_engine(
            llm=MockLLM(),
            similarity_top_k=3,
            filters=filters,
        )
        query_text = "Tell me about dense vector databases."

        retrieved_nodes = retriever.retrieve(query_text)
        assert len(retrieved_nodes) == 1
        assert retrieved_nodes[0].metadata["topic"] == "vector-db"

        response = query_engine.query(query_text)
        assert len(response.source_nodes) == 1
        assert response.source_nodes[0].metadata["topic"] == "vector-db"


@pytest.mark.asyncio
async def test_async_retriever_and_query_engine(documents: list[Document]) -> None:
    async with _amanaged_vector_store() as vector_store:
        vector_store.connect()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=MockEmbedding(embed_dim=EMBED_DIM),
            storage_context=storage_context,
        )

        retriever = index.as_retriever(similarity_top_k=3)
        query_engine = index.as_query_engine(llm=MockLLM(), similarity_top_k=3)
        query_text = documents[2].text

        await _aassert_retrieval_with_retries(
            retriever,
            query_text,
            expected_topic="filters",
        )

        response = await query_engine.aquery(query_text)
        vector_store.close()
        assert any(
            node.metadata.get("topic") == "filters" for node in response.source_nodes
        )
