from typing import Callable, Iterator
from unittest import mock

import pytest

from llama_index.llm_predictor.vellum import VellumPredictor
from llama_index.prompts import BasePromptTemplate


def test_predict__basic(
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
    vellum_predictor_factory: Callable[..., VellumPredictor],
    dummy_prompt: BasePromptTemplate,
) -> None:
    """When the Vellum API returns expected values, so should our predictor"""

    vellum_client = mock_vellum_client_factory(
        compiled_prompt_text="What's you're favorite greeting?",
        completion_text="Hello, world!",
    )

    predictor = vellum_predictor_factory(vellum_client=vellum_client)

    completion_text = predictor.predict(dummy_prompt, thing="greeting")

    assert completion_text == "Hello, world!"


def test_stream__basic(
    mock_vellum_client_factory: Callable[..., mock.MagicMock],
    vellum_predictor_factory: Callable[..., VellumPredictor],
    dummy_prompt: BasePromptTemplate,
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

    completion_generator = predictor.stream(dummy_prompt, thing="greeting")

    assert next(completion_generator) == "Hello,"
    assert next(completion_generator) == " world!"
    with pytest.raises(StopIteration):
        next(completion_generator)
