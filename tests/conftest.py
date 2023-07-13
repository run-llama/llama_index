# import os
# import socket
from typing import Any

import pytest
from llama_index.indices.service_context import ServiceContext
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.llms.base import LLMMetadata


from tests.indices.vector_store.mock_services import MockEmbedding
from llama_index.llms.mock import MockLLM
from tests.mock_utils.mock_predict import (
    patch_llmpredictor_apredict,
    patch_llmpredictor_predict,
)
from tests.mock_utils.mock_text_splitter import (
    patch_token_splitter_newline,
    patch_token_splitter_newline_with_overlaps,
)


# @pytest.fixture(autouse=True)
# def no_networking(monkeypatch: pytest.MonkeyPatch) -> None:
#     def deny_network(*args: Any, **kwargs: Any) -> None:
#         raise RuntimeError("Network access denied for test")

#     monkeypatch.setattr(socket, "socket", deny_network)


@pytest.fixture
def allow_networking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()


@pytest.fixture
def patch_token_text_splitter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TokenTextSplitter, "split_text", patch_token_splitter_newline)
    monkeypatch.setattr(
        TokenTextSplitter,
        "split_text_with_overlaps",
        patch_token_splitter_newline_with_overlaps,
    )


@pytest.fixture
def patch_llm_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
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
        "llm",
        MockLLM(),
    )
    monkeypatch.setattr(
        LLMPredictor,
        "__init__",
        lambda x: None,
    )
    monkeypatch.setattr(
        LLMPredictor,
        "metadata",
        LLMMetadata(),
    )


@pytest.fixture()
def mock_service_context(
    patch_token_text_splitter: Any, patch_llm_predictor: Any
) -> ServiceContext:
    return ServiceContext.from_defaults(embed_model=MockEmbedding())


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM()
