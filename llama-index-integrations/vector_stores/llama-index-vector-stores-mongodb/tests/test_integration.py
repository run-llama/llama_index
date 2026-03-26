"""
Integration Tests of llama-index-vector-stores-mongodb
with MongoDB Atlas Vector Datastore and OPENAI Embedding model.

As described in docs/providers/mongodb/setup.md, to run this, one must
have a running MongoDB Atlas Cluster, and
provide a valid OPENAI_API_KEY.
"""

import asyncio
import pytest
from typing import List

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from .conftest import lock


def test_mongodb_connection(atlas_client: MongoClient) -> None:
    """Confirm that the connection to the datastore works."""
    assert atlas_client.admin.command("ping")["ok"]


@pytest.mark.asyncio
async def test_index(
    documents: List[Document], vector_store: MongoDBAtlasVectorSearch
) -> None:
    """
    End-to-end example from essay and query to response.

    via NodeParser, LLM Embedding, VectorStore, and Synthesizer.
    """
    with lock:
        vector_store.collection.delete_many({})
        await asyncio.sleep(2)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        query_engine = index.as_query_engine()

        question = "What are LLMs useful for?"
        no_response = True
        response = None
        retries = 5
        search_limit = query_engine.retriever.similarity_top_k
        while no_response and retries:
            response = query_engine.query(question)
            async_response = await query_engine.aquery(question)
            if (
                len(response.source_nodes) == search_limit
                and len(async_response.source_nodes) == search_limit
            ):
                no_response = False
            else:
                retries -= 1
                await asyncio.sleep(5)

        assert retries
        assert "knowledge generation and reasoning" in response.response
        assert "knowledge generation and reasoning" in async_response.response
