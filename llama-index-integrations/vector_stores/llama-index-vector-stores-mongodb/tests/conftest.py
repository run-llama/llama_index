import os
from typing import List
import pytest
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

import threading

lock = threading.Lock()


@pytest.fixture(scope="session")
def documents() -> List[Document]:
    """List of documents represents data to be embedded in the datastore.
    Minimum requirements for Documents in the /upsert endpoint's UpsertRequest.
    """
    text = Document.example().text
    metadata = Document.example().metadata
    texts = text.split("\n")
    return [Document(text=text, metadata={"text": text}) for text in texts]


@pytest.fixture(scope="session")
def nodes(documents) -> List[TextNode]:
    if OPENAI_API_KEY is None:
        return None

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=200),
            OpenAIEmbedding(),
        ],
    )

    return pipeline.run(documents=documents)


db_name = os.environ.get("MONGODB_DATABASE", "llama_index_test_db")
collection_name = os.environ.get("MONGODB_COLLECTION", "llama_index_test_vectorstore")
vector_index_name = os.environ.get("MONGODB_INDEX", "vector_index")
MONGODB_URI = os.environ.get("MONGODB_URI")


@pytest.fixture(scope="session")
def atlas_client() -> MongoClient:
    if MONGODB_URI is None:
        return None

    client = MongoClient(MONGODB_URI)

    assert db_name in client.list_database_names()
    assert collection_name in client[db_name].list_collection_names()

    # TODO error: $listSearchIndexes is not allowed or the syntax is incorrect
    # assert vector_index_name in [
    #    idx["name"] for idx in client[db_name][collection_name].list_search_indexes()
    # ]

    # Clear the collection for the tests
    client[db_name][collection_name].delete_many({})

    return client


@pytest.fixture(scope="session")
def vector_store(atlas_client: MongoClient) -> MongoDBAtlasVectorSearch:
    if MONGODB_URI is None:
        return None

    return MongoDBAtlasVectorSearch(
        mongodb_client=atlas_client,
        db_name=db_name,
        collection_name=collection_name,
        vector_index_name=vector_index_name,
        fulltext_index_name="fulltext_index",
    )
