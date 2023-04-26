from typing import List
import pytest
import pathlib
import sys
from typing import List
from unittest.mock import MagicMock

import pytest
from gpt_index.indices.service_context import ServiceContext
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.llm_predictor.base import LLMPredictor


from gpt_index.readers.schema.base import Document
from gpt_index.storage.storage_context import StorageContext
from gpt_index.vector_stores.faiss import FaissVectorStore
from tests.indices.vector_store.mock_services import MockEmbedding
from tests.indices.vector_store.mock_faiss import MockFaissIndex
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_predict import mock_llmpredictor_predict
from tests.mock_utils.mock_text_splitter import (
    mock_token_splitter_newline,
    mock_token_splitter_newline_with_overlaps,
)


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text)]


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


@pytest.fixture
def patch_token_text_splitter(monkeypatch: pytest.MonkeyPatch) -> TokenTextSplitter:
    monkeypatch.setattr(TokenTextSplitter, "split_text", mock_token_splitter_newline)
    monkeypatch.setattr(
        TokenTextSplitter,
        "split_text_with_overlaps",
        mock_token_splitter_newline_with_overlaps,
    )


@pytest.fixture
def patch_llm_predictor(monkeypatch: pytest.MonkeyPatch) -> TokenTextSplitter:
    monkeypatch.setattr(
        LLMPredictor,
        "total_tokens_used",
        0,
    )
    monkeypatch.setattr(
        LLMPredictor,
        "predict",
        mock_llmpredictor_predict,
    )
    monkeypatch.setattr(
        LLMPredictor,
        "__init__",
        lambda x: None,
    )


@pytest.fixture()
def mock_service_context(
    patch_token_text_splitter, patch_llm_predictor
) -> ServiceContext:
    return ServiceContext.from_defaults(embed_model=MockEmbedding())
