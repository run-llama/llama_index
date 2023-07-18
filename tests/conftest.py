import os
# import socket
from typing import Any, Optional

import openai
import pytest

from llama_index.indices.service_context import ServiceContext
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.llms.base import LLMMetadata
from llama_index.llms.mock import MockLLM
from tests.indices.vector_store.mock_services import MockEmbedding
from tests.mock_utils.mock_predict import (patch_llmpredictor_apredict,
                                           patch_llmpredictor_predict)
from tests.mock_utils.mock_text_splitter import (
    patch_token_splitter_newline, patch_token_splitter_newline_with_overlaps)

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

class CachedOpenAIApiKeys:
    """
    Saves the users' OpenAI API key either in the environment variable
    or set to the library itself.
    This allows us to run tests by setting it without plowing over
    the local environment.
    """

    def __init__(
        self,
        set_env_key_to: Optional[str] = "",
        set_library_key_to: Optional[str] = None,
        set_fake_key: bool = False,
    ):
        self.set_env_key_to = set_env_key_to
        self.set_library_key_to = set_library_key_to
        self.set_fake_key = set_fake_key

    def __enter__(self) -> None:
        self.api_env_variable_was = os.environ.get("OPENAI_API_KEY", "")
        self.openai_api_key_was = openai.api_key

        os.environ["OPENAI_API_KEY"] = str(self.set_env_key_to)
        openai.api_key = self.set_library_key_to

        if self.set_fake_key:
            openai.api_key = "sk-" + "a" * 48

    # No matter what, set the environment variable back to what it was
    def __exit__(self, *exc: Any) -> None:
        os.environ["OPENAI_API_KEY"] = str(self.api_env_variable_was)
        openai.api_key = self.openai_api_key_was