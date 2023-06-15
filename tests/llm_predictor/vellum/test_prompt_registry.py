from typing import Callable, Type
from unittest import mock

from llama_index import Prompt
from llama_index.llm_predictor.vellum import (
    VellumRegisteredPrompt,
    VellumCompiledPrompt,
    VellumPromptRegistry,
)


def test_from_prompt__new(
    dummy_prompt_class: Type[Prompt],
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
    vellum_prompt_registry_factory: Callable[..., VellumPromptRegistry],
) -> None:
    """We should register a new prompt if no deployment exists"""

    from vellum.core import ApiError

    dummy_prompt = dummy_prompt_class(template="What's your favorite {thing}?")

    vellum_client = mock_vellum_client_factory()

    vellum_client.deployments.retrieve.side_effect = ApiError(status_code=404)

    prompt_registry = vellum_prompt_registry_factory(vellum_client=vellum_client)
    prompt_registry.from_prompt(dummy_prompt)

    vellum_client.registered_prompts.register_prompt.assert_called_once()


def test_from_prompt__existing(
    dummy_prompt_class: Type[Prompt],
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
    vellum_prompt_registry_factory: Callable[..., VellumPromptRegistry],
) -> None:
    """We shouldn't register a new prompt if a deployment id or name is provided"""

    dummy_prompt = dummy_prompt_class(
        template="What's your favorite {thing}?",
        metadata={"vellum_deployment_id": "abc"},
    )

    mock_deployment = mock.MagicMock(active_model_version_ids=["abc"])

    vellum_client = mock_vellum_client_factory()
    vellum_client.deployments = mock.MagicMock()
    vellum_client.deployments.retrieve.return_value = mock_deployment

    prompt_registry = vellum_prompt_registry_factory(vellum_client=vellum_client)
    prompt_registry.from_prompt(dummy_prompt)

    vellum_client.registered_prompts.register_prompt.assert_not_called()


def test_get_compiled_prompt__basic(
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
    vellum_prompt_registry_factory: Callable[..., VellumPromptRegistry],
) -> None:
    """Verify that we can get a compiled prompt from the registry"""

    registered_prompt = VellumRegisteredPrompt(
        deployment_id="abc",
        deployment_name="my-deployment",
        model_version_id="123",
    )

    vellum_client = mock_vellum_client_factory()
    mock_model_version_compile_prompt = mock.MagicMock()
    mock_model_version_compile_prompt.prompt.text = "What's your favorite greeting?"
    mock_model_version_compile_prompt.prompt.num_tokens = 5

    vellum_client.model_versions.model_version_compile_prompt.return_value = (
        mock_model_version_compile_prompt
    )

    prompt_registry = vellum_prompt_registry_factory(vellum_client=vellum_client)

    compiled_prompt = prompt_registry.get_compiled_prompt(
        registered_prompt, input_values={"thing": "greeting"}
    )

    assert compiled_prompt == VellumCompiledPrompt(
        text="What's your favorite greeting?", num_tokens=5
    )
