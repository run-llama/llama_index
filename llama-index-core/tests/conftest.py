# pants requires this import to recognize the dep
import pytest_asyncio  # noqa: F401

import os

# import socket
from typing import List, Optional

import openai
import pytest
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser.text import SentenceSplitter, TokenTextSplitter
from llama_index.core.service_context_elements.llm_predictor import LLMPredictor
from llama_index.core.settings import _Settings

from tests.indices.vector_store.mock_services import MockEmbedding
from tests.mock_utils.mock_predict import (
    patch_llmpredictor_apredict,
    patch_llmpredictor_predict,
)
from tests.mock_utils.mock_text_splitter import patch_token_splitter_newline

# @pytest.fixture(autouse=True)
# def no_networking(monkeypatch: pytest.MonkeyPatch) -> None:
#     def deny_network(*args: Any, **kwargs: Any) -> None:
#         raise RuntimeError("Network access denied for test")

#     monkeypatch.setattr(socket, "socket", deny_network)


@pytest.fixture(autouse=True)
def set_env_vars():
    # set env var for testing
    os.environ["IS_TESTING"] = "1"
    yield
    # test clean up, unset env var
    del os.environ["IS_TESTING"]


@pytest.fixture()
def allow_networking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()


@pytest.fixture()
def patch_token_text_splitter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SentenceSplitter, "split_text", patch_token_splitter_newline)
    monkeypatch.setattr(
        SentenceSplitter,
        "split_text_metadata_aware",
        patch_token_splitter_newline,
    )
    monkeypatch.setattr(TokenTextSplitter, "split_text", patch_token_splitter_newline)
    monkeypatch.setattr(
        TokenTextSplitter, "split_text_metadata_aware", patch_token_splitter_newline
    )


@pytest.fixture()
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
        "metadata",
        LLMMetadata(),
    )

    monkeypatch.setattr(
        MockLLM,
        "predict",
        patch_llmpredictor_predict,
    )
    monkeypatch.setattr(
        MockLLM,
        "apredict",
        patch_llmpredictor_apredict,
    )
    monkeypatch.setattr(
        MockLLM,
        "metadata",
        LLMMetadata(),
    )


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture()
def mock_embed_model():
    return MockEmbedding()


@pytest.fixture()
def mock_settings():
    from llama_index.core import Settings

    old = Settings
    Settings = _Settings()
    Settings.embed_model = MockEmbedding()
    yield Settings
    Settings = old


@pytest.fixture(autouse=True)
def mock_openai_credentials() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-" + ("a" * 48)


class CachedOpenAIApiKeys:
    """
    Saves the users' OpenAI API key and OpenAI API type either in
    the environment variable or set to the library itself.
    This allows us to run tests by setting it without plowing over
    the local environment.
    """

    def __init__(
        self,
        set_env_key_to: Optional[str] = "",
        set_library_key_to: Optional[str] = None,
        set_fake_key: bool = False,
        set_env_type_to: Optional[str] = "",
        set_library_type_to: str = "open_ai",  # default value in openai package
    ):
        self.set_env_key_to = set_env_key_to
        self.set_library_key_to = set_library_key_to
        self.set_fake_key = set_fake_key
        self.set_env_type_to = set_env_type_to
        self.set_library_type_to = set_library_type_to

    def __enter__(self) -> None:
        self.api_env_variable_was = os.environ.get("OPENAI_API_KEY", "")
        self.api_env_type_was = os.environ.get("OPENAI_API_TYPE", "")
        self.openai_api_key_was = openai.api_key
        self.openai_api_type_was = openai.api_type

        os.environ["OPENAI_API_KEY"] = str(self.set_env_key_to)
        os.environ["OPENAI_API_TYPE"] = str(self.set_env_type_to)

        if self.set_fake_key:
            os.environ["OPENAI_API_KEY"] = "sk-" + "a" * 48

    # No matter what, set the environment variable back to what it was
    def __exit__(self, *exc: object) -> None:
        os.environ["OPENAI_API_KEY"] = str(self.api_env_variable_was)
        os.environ["OPENAI_API_TYPE"] = str(self.api_env_type_was)
        openai.api_key = self.openai_api_key_was
        openai.api_type = self.openai_api_type_was


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: mark test as integration")


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    if config.getoption("--integration"):
        # --integration given in cli: do not skip integration tests
        return
    skip_integration = pytest.mark.skip(reason="need --integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
