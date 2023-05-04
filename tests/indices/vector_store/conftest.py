import os
import pytest
import pathlib
import sys
from unittest.mock import MagicMock


from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from tests.indices.vector_store.mock_faiss import MockFaissIndex


@pytest.fixture()
def faiss_vector_store(tmp_path: pathlib.Path) -> FaissVectorStore:
    # NOTE: mock faiss import for CI
    if "CI" in os.environ:
        sys.modules["faiss"] = MagicMock()

    # NOTE: mock faiss index
    faiss_index = MockFaissIndex()

    return FaissVectorStore(faiss_index=faiss_index)


@pytest.fixture()
def faiss_storage_context(faiss_vector_store: FaissVectorStore) -> StorageContext:
    return StorageContext.from_defaults(vector_store=faiss_vector_store)
