from typing import Optional, Type, Callable
from unittest import mock

import pytest

from llama_index import Prompt
from llama_index.callbacks import CallbackManager
from llama_index.llm_predictor.vellum import VellumPredictor, VellumPromptRegistry
from llama_index.prompts.prompt_type import PromptType


@pytest.fixture
def dummy_prompt_class() -> Type[Prompt]:
    class DummyPrompt(Prompt):
        prompt_type = PromptType.CUSTOM
        input_variables = ["thing"]

    return DummyPrompt


@pytest.fixture
def dummy_prompt(dummy_prompt_class: Type[Prompt]) -> Prompt:
    return dummy_prompt_class(template="What's your favorite {thing}?")


@pytest.fixture
def fake_vellum_api_key() -> str:
    return "abc-123"


@pytest.fixture
def mock_vellum_client_factory() -> Callable[..., mock.MagicMock]:
    import vellum

    def _create_vellum_client(
        compiled_prompt_text: str = "<example-compiled-prompt-text>",
        compiled_prompt_num_tokens: int = 0,
        completion_text: str = "<example_completion>",
    ) -> mock.MagicMock:
        mocked_vellum_client = mock.MagicMock()

        mocked_vellum_client.model_versions.model_version_compile_prompt.return_value.prompt = vellum.ModelVersionCompiledPrompt(  # noqa: E501
            text=compiled_prompt_text, num_tokens=compiled_prompt_num_tokens
        )
        mocked_vellum_client.generate.return_value = vellum.GenerateResponse(
            results=[
                vellum.GenerateResult(
                    data=vellum.GenerateResultData(
                        completions=[
                            vellum.EnrichedNormalizedCompletion(
                                id="<example-generation-id>",
                                external_id=None,
                                text=completion_text,
                                model_version_id="<example-model-version-id>",
                            )
                        ]
                    ),
                    error=None,
                )
            ]
        )

        return mocked_vellum_client

    return _create_vellum_client


@pytest.fixture
def mock_vellum_async_client_factory() -> Callable[..., mock.MagicMock]:
    def _create_async_vellum_client() -> mock.MagicMock:
        return mock.MagicMock()

    return _create_async_vellum_client


@pytest.fixture
def vellum_prompt_registry_factory(
    fake_vellum_api_key: str,
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
) -> Callable[..., VellumPromptRegistry]:
    def _create_vellum_prompt_registry(
        vellum_client: Optional[mock.MagicMock] = None,
    ) -> VellumPromptRegistry:
        prompt_registry = VellumPromptRegistry(vellum_api_key=fake_vellum_api_key)
        prompt_registry._vellum_client = vellum_client or mock_vellum_client_factory()

        return prompt_registry

    return _create_vellum_prompt_registry


@pytest.fixture
def vellum_predictor_factory(
    fake_vellum_api_key: str,
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
    mock_vellum_async_client_factory: Callable[..., mock.MagicMock],
    vellum_prompt_registry_factory: Callable[..., mock.MagicMock],
) -> Callable[..., VellumPredictor]:
    def _create_vellum_predictor(
        callback_manager: Optional[CallbackManager] = None,
        vellum_client: Optional[mock.MagicMock] = None,
        async_vellum_client: Optional[mock.MagicMock] = None,
        vellum_prompt_registry: Optional[mock.MagicMock] = None,
    ) -> VellumPredictor:
        predictor = VellumPredictor(
            vellum_api_key=fake_vellum_api_key, callback_manager=callback_manager
        )

        vellum_client = vellum_client or mock_vellum_client_factory()
        async_vellum_client = async_vellum_client or mock_vellum_async_client_factory()
        vellum_prompt_registry = (
            vellum_prompt_registry
            or vellum_prompt_registry_factory(vellum_client=vellum_client)
        )

        predictor._vellum_client = vellum_client
        predictor._async_vellum_client = async_vellum_client
        predictor._prompt_registry = vellum_prompt_registry

        return predictor

    return _create_vellum_predictor
