import pytest
import pathlib
import sys
from unittest.mock import MagicMock

import pytest


from gpt_index.storage.storage_context import StorageContext
from gpt_index.vector_stores.faiss import FaissVectorStore
from tests.indices.vector_store.mock_faiss import MockFaissIndex


@pytest.fixture()
def faiss_vector_store(tmp_path: pathlib) -> FaissVectorStore:
    # NOTE: use temp file
    file_path = tmp_path / "test_file.txt"

    # NOTE: mock faiss import
    sys.modules["faiss"] = MagicMock()

    # NOTE: mock faiss index
    faiss_index = MockFaissIndex()

    return FaissVectorStore(faiss_index=faiss_index, persist_path=file_path)


@pytest.fixture()
def faiss_storage_context(
    faiss_vector_store: FaissVectorStore, tmp_path: pathlib.Path
) -> StorageContext:
    return StorageContext.from_defaults(
        vector_store=faiss_vector_store, persist_dir=tmp_path
    )
