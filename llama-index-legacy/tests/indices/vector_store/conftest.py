import os
import pathlib
import sys
from unittest.mock import MagicMock

import pytest
from llama_index.legacy.storage.storage_context import StorageContext
from llama_index.legacy.vector_stores.faiss import FaissVectorStore
from llama_index.legacy.vector_stores.txtai import TxtaiVectorStore

from tests.indices.vector_store.mock_faiss import MockFaissIndex
from tests.indices.vector_store.mock_txtai import MockTxtaiIndex


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


@pytest.fixture()
def txtai_vector_store(tmp_path: pathlib.Path) -> TxtaiVectorStore:
    # NOTE: mock txtai import for CI
    if "CI" in os.environ:
        sys.modules["txtai"] = MagicMock()

    # NOTE: mock txtai index
    txtai_index = MockTxtaiIndex()

    return TxtaiVectorStore(txtai_index=txtai_index)


@pytest.fixture()
def txtai_storage_context(txtai_vector_store: TxtaiVectorStore) -> StorageContext:
    return StorageContext.from_defaults(vector_store=txtai_vector_store)
