from typing import List
import pytest

from gpt_index.readers.schema.base import Document

import pytest

import pytest
from gpt_index.indices.service_context import ServiceContext
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.llm_predictor.base import LLMPredictor


from tests.indices.vector_store.mock_services import MockEmbedding
from tests.mock_utils.mock_predict import mock_llmpredictor_predict, patch_llmpredictor_apredict, patch_llmpredictor_predict
from tests.mock_utils.mock_text_splitter import (
    patch_token_splitter_newline,
    patch_token_splitter_newline_with_overlaps,
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


@pytest.fixture
def patch_token_text_splitter(monkeypatch: pytest.MonkeyPatch) -> TokenTextSplitter:
    monkeypatch.setattr(TokenTextSplitter, "split_text", patch_token_splitter_newline)
    monkeypatch.setattr(
        TokenTextSplitter,
        "split_text_with_overlaps",
        patch_token_splitter_newline_with_overlaps,
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
        patch_llmpredictor_predict,
    )
    monkeypatch.setattr(
        LLMPredictor,
        "apredict",
        patch_llmpredictor_apredict,
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
