from typing import Callable, Iterator
from unittest import mock

import pytest

from llama_index import Prompt
from llama_index.callbacks import CBEventType
from llama_index.llm_predictor.vellum import (
    VellumRegisteredPrompt,
    VellumPredictor,
    VellumPromptRegistry,
)


def test_predict__basic(
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
    vellum_predictor_factory: Callable[..., VellumPredictor],
    dummy_prompt: Prompt,
) -> None:
    """When the Vellum API returns expected values, so should our predictor"""

    vellum_client = mock_vellum_client_factory(
        compiled_prompt_text="What's you're favorite greeting?",
        completion_text="Hello, world!",
    )

    predictor = vellum_predictor_factory(vellum_client=vellum_client)

    completion_text, compiled_prompt_text = predictor.predict(
        dummy_prompt, thing="greeting"
    )

    assert completion_text == "Hello, world!"
    assert compiled_prompt_text == "What's you're favorite greeting?"


def test_predict__callback_manager(
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
    vellum_predictor_factory: Callable[..., VellumPredictor],
    vellum_prompt_registry_factory: Callable[..., VellumPromptRegistry],
    dummy_prompt: Prompt,
) -> None:
    """Ensure we invoke a callback manager, when provided"""

    callback_manager = mock.MagicMock()

    vellum_client = mock_vellum_client_factory(
        compiled_prompt_text="What's you're favorite greeting?",
        completion_text="Hello, world!",
    )

    registered_prompt = VellumRegisteredPrompt(
        deployment_id="abc",
        deployment_name="my-deployment",
        model_version_id="123",
    )
    prompt_registry = vellum_prompt_registry_factory(vellum_client=vellum_client)

    with mock.patch.object(prompt_registry, "from_prompt") as mock_from_prompt:
        mock_from_prompt.return_value = registered_prompt

        predictor = vellum_predictor_factory(
            callback_manager=callback_manager,
            vellum_client=vellum_client,
            vellum_prompt_registry=prompt_registry,
        )

        predictor.predict(dummy_prompt, thing="greeting")

    callback_manager.on_event_start.assert_called_once_with(
        CBEventType.LLM,
        payload={
            "thing": "greeting",
            "deployment_id": registered_prompt.deployment_id,
            "model_version_id": registered_prompt.model_version_id,
        },
    )
    callback_manager.on_event_end.assert_called_once_with(
        CBEventType.LLM,
        payload={
            "response": "Hello, world!",
            "formatted_prompt": "What's you're favorite greeting?",
        },
        event_id=mock.ANY,
    )


def test_stream__basic(
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
    vellum_predictor_factory: Callable[..., VellumPredictor],
    dummy_prompt: Prompt,
) -> None:
    """When the Vellum API streams expected values, so should our predictor"""

    import vellum

    vellum_client = mock_vellum_client_factory(
        compiled_prompt_text="What's you're favorite greeting?",
    )

    def fake_stream() -> Iterator[vellum.GenerateStreamResponse]:
        yield vellum.GenerateStreamResponse(
            delta=vellum.GenerateStreamResult(
                request_index=0,
                data=vellum.GenerateStreamResultData(
                    completion_index=0,
                    completion=vellum.EnrichedNormalizedCompletion(
                        id="123", text="Hello,", model_version_id="abc"
                    ),
                ),
                error=None,
            )
        )
        yield vellum.GenerateStreamResponse(
            delta=vellum.GenerateStreamResult(
                request_index=0,
                data=vellum.GenerateStreamResultData(
                    completion_index=0,
                    completion=vellum.EnrichedNormalizedCompletion(
                        id="456", text=" world!", model_version_id="abc"
                    ),
                ),
                error=None,
            )
        )

    vellum_client.generate_stream.return_value = fake_stream()

    predictor = vellum_predictor_factory(vellum_client=vellum_client)

    completion_generator, compiled_prompt_text = predictor.stream(
        dummy_prompt, thing="greeting"
    )

    assert next(completion_generator) == "Hello,"
    assert next(completion_generator) == " world!"
    with pytest.raises(StopIteration):
        next(completion_generator)

    assert compiled_prompt_text == "What's you're favorite greeting?"


def test_stream__callback_manager(
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
    vellum_predictor_factory: Callable[..., VellumPredictor],
    vellum_prompt_registry_factory: Callable[..., VellumPromptRegistry],
    dummy_prompt: Prompt,
) -> None:
    """Ensure we invoke a callback manager, when provided"""

    import vellum

    callback_manager = mock.MagicMock()

    vellum_client = mock_vellum_client_factory(
        compiled_prompt_text="What's you're favorite greeting?",
        completion_text="Hello, world!",
    )

    def fake_stream() -> Iterator[vellum.GenerateStreamResponse]:
        yield vellum.GenerateStreamResponse(
            delta=vellum.GenerateStreamResult(
                request_index=0,
                data=vellum.GenerateStreamResultData(
                    completion_index=0,
                    completion=vellum.EnrichedNormalizedCompletion(
                        id="123", text="Hello,", model_version_id="abc"
                    ),
                ),
                error=None,
            )
        )
        yield vellum.GenerateStreamResponse(
            delta=vellum.GenerateStreamResult(
                request_index=0,
                data=vellum.GenerateStreamResultData(
                    completion_index=0,
                    completion=vellum.EnrichedNormalizedCompletion(
                        id="456", text=" world!", model_version_id="abc"
                    ),
                ),
                error=None,
            )
        )

    vellum_client.generate_stream.return_value = fake_stream()

    registered_prompt = VellumRegisteredPrompt(
        deployment_id="abc",
        deployment_name="my-deployment",
        model_version_id="123",
    )
    prompt_registry = vellum_prompt_registry_factory(vellum_client=vellum_client)

    with mock.patch.object(prompt_registry, "from_prompt") as mock_from_prompt:
        mock_from_prompt.return_value = registered_prompt

        predictor = vellum_predictor_factory(
            callback_manager=callback_manager,
            vellum_client=vellum_client,
            vellum_prompt_registry=prompt_registry,
        )

        completion_generator, compiled_prompt_text = predictor.stream(
            dummy_prompt, thing="greeting"
        )

    assert next(completion_generator) == "Hello,"
    assert next(completion_generator) == " world!"
    with pytest.raises(StopIteration):
        next(completion_generator)

    callback_manager.on_event_start.assert_called_once_with(
        CBEventType.LLM,
        payload={
            "thing": "greeting",
            "deployment_id": registered_prompt.deployment_id,
            "model_version_id": registered_prompt.model_version_id,
        },
    )
    callback_manager.on_event_end.assert_called_once_with(
        CBEventType.LLM,
        payload={
            "response": "Hello, world!",
            "formatted_prompt": "What's you're favorite greeting?",
        },
        event_id=mock.ANY,
    )
