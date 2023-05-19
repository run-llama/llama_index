from __future__ import annotations

from typing import Any, Tuple, Generator

import vellum
from vellum.client import AsyncVellum, Vellum

from llama_index import Prompt
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.llm_predictor.vellum.exceptions import VellumGenerateException
from llama_index.llm_predictor.vellum.prompt_registry import VellumPromptRegistry


class VellumPredictor(LLMPredictor):
    def __init__(self, vellum_api_key: str) -> None:
        super().__init__(llm=None, retry_on_throttling=False, ensure_llm=False)

        self._vellum_client = Vellum(api_key=vellum_api_key)
        self._async_vellum_client = AsyncVellum(api_key=vellum_api_key)
        self._prompt_registry = VellumPromptRegistry(vellum_api_key=vellum_api_key)

    # Override
    def _get_formatted_prompt(self, prompt: Prompt, **prompt_args: Any) -> str:
        # TODO: Expose endpoint for retrieving a compiled prompt
        #   (endpoint already exists).

        return ""

    # Override
    def _predict(self, prompt: Prompt, **prompt_args: Any) -> str:
        deployment = self._prompt_registry.from_prompt(prompt)

        result = self._vellum_client.generate(
            deployment_id=deployment.deployment_id,
            requests=[
                vellum.GenerateRequest(
                    input_values=prompt.get_full_format_args(prompt_args)
                )
            ],
        )

        return result.text

    # Override
    def stream(self, prompt: Prompt, **prompt_args: Any) -> Tuple[Generator, str]:
        deployment = self._prompt_registry.from_prompt(prompt)

        responses = self._vellum_client.generate_stream(
            deployment_id=deployment.deployment_id,
            deployment_name=deployment.deployment_name,
            requests=[
                vellum.GenerateRequest(
                    input_values=prompt.get_full_format_args(prompt_args)
                )
            ],
        )

        def text_generator() -> Generator:
            for stream_response in responses:
                result: vellum.GenerateStreamResult = stream_response.delta

                if result.error:
                    raise VellumGenerateException(result.error.message)
                elif not result.data:
                    raise VellumGenerateException(
                        "Unknown error occurred while generating"
                    )

                yield result.data.completion.text

        return text_generator(), ""

    # Override
    async def _apredict(self, prompt: Prompt, **prompt_args: Any) -> str:
        deployment = self._prompt_registry.from_prompt(prompt)

        result = await self._async_vellum_client.generate(
            deployment_id=deployment.deployment_id,
            deployment_name=deployment.deployment_name,
            requests=[
                vellum.GenerateRequest(
                    input_values=prompt.get_full_format_args(prompt_args)
                )
            ],
        )

        return result.text
