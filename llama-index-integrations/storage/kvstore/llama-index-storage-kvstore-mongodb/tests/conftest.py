import os
import pytest
from pymongo import MongoClient

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

import threading

lock = threading.Lock()

db_name = os.environ.get("MONGODB_DATABASE", "llama_index_test_db")
collection_name = os.environ.get("MONGODB_COLLECTION", "llama_index_test_kvstore")
MONGODB_URI = os.environ.get("MONGODB_URI")


@pytest.fixture(scope="session")
def atlas_client() -> MongoClient:
    if MONGODB_URI is None:
        return None

    client = MongoClient(MONGODB_URI)

    assert db_name in client.list_database_names()
    assert collection_name in client[db_name].list_collection_names()

    # Clear the collection for the tests
    client[db_name][collection_name].delete_many({})

    return client
