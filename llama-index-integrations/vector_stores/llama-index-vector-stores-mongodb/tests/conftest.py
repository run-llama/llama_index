import os
from time import sleep

import openai
import pytest
from llama_index.core import (SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import NodeParser, SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

openai.api_key = os.environ["OPENAI_API_KEY"]

import threading
from pathlib import Path

lock = threading.Lock()


@pytest.fixture(scope="session")
def documents(tmp_path_factory):
    """ List of documents represents data to be embedded in the datastore.
    Minimum requirements fpr Documents in the /upsert endpoint's UpsertRequest.
    """
    data_dir = Path(__file__).parents[4] / "docs/docs/examples/data/paul_graham"
    return SimpleDirectoryReader(data_dir).load_data()


@pytest.fixture(scope="session")
def nodes(documents):
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=200),
            OpenAIEmbedding(),
        ],
    )

    nodes = pipeline.run(documents=documents)
    return nodes


db_name = os.environ.get("MONGODB_DATABASE", "llama_index_test_db")
collection_name = os.environ.get("MONGODB_COLLECTION", "llama_index_test_vectorstore")
index_name = os.environ.get("MONGODB_INDEX", "vector_index")
cluster_uri = os.environ["MONGO_URI"]


@pytest.fixture(scope="session")
def atlas_client():

    client = MongoClient(cluster_uri)

    assert db_name in client.list_database_names()
    assert collection_name in client[db_name].list_collection_names()
    assert index_name in [idx['name'] for idx in client[db_name][collection_name].list_search_indexes()]

    # Clear the collection for the tests
    client[db_name][collection_name].delete_many({})

    return client


@pytest.fixture(scope="session")
def vector_store(atlas_client):

    return MongoDBAtlasVectorSearch(
        mongodb_client=atlas_client,
        db_name=db_name,
        collection_name=collection_name,
        index_name=index_name,
    )