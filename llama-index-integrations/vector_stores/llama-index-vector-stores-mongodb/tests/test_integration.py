"""Integration Tests of llama-index-vector-stores-mongodb
with MongoDB Atlas Vector Datastore and OPENAI Embedding model.

As described in docs/providers/mongodb/setup.md, to run this, one must
have a running MongoDB Atlas Cluster, and
provide a valid OPENAI_API_KEY.
"""

import os
from time import sleep

import pytest
from llama_index.core import StorageContext, VectorStoreIndex

from .conftest import lock


def test_required_vars():
    """Confirm that the environment has all it needs"""
    required_vars = ['OPENAI_API_KEY', 'MONGO_URI']
    for var in required_vars:
        try:
            os.environ[var]
        except KeyError:
            pytest.fail(f"Required var '{var}' not in os.environ")


def test_mongodb_connection(atlas_client):
    """Confirm that the connection to the datastore works."""
    assert atlas_client.admin.command('ping')['ok']


def test_index(documents, vector_store):
    """End-to-end example from essay and query to response

    via NodeParser, LLM Embedding, VectorStore, and Synthesizer."""
    with lock:
        vector_store._collection.delete_many({})
        sleep(2)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        query_engine = index.as_query_engine()

        question = "Who is the author of this essay?"
        no_response = True
        response = None
        while no_response:
            response = query_engine.query(question)
            if response.response == "Empty Response":
                sleep(2)
            else:
                no_response = False
        assert "Paul Graham" == response.response
