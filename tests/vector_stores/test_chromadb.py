from typing import Dict

import pytest
from llama_index.vector_stores import ChromaVectorStore

##
# Start chromadb locally
# cd tests
# docker-compose up
#
# Run tests
# cd tests/vector_stores
# pytest test_chromadb.py


PARAMS: Dict[str, str] = {
    "host": "localhost",
    "port": "8000",
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
def test_instance_creation() -> None:
    connection = chromadb.HttpClient(**PARAMS)
    collection = connection.get_collection(COLLECTION_NAME)
    store = ChromaVectorStore(chroma_collection=collection)
    assert isinstance(store, ChromaVectorStore)
