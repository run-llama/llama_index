from __future__ import annotations

from typing import Any, Tuple, Generator, Optional

import vellum
from vellum.client import AsyncVellum, Vellum

from llama_index import Prompt
from llama_index.callbacks import CallbackManager, CBEventType
from llama_index.constants import NUM_OUTPUTS, MAX_CHUNK_SIZE
from llama_index.llm_predictor.base import BaseLLMPredictor, LLMMetadata
from llama_index.llm_predictor.vellum.exceptions import VellumGenerateException
from llama_index.llm_predictor.vellum.prompt_registry import VellumPromptRegistry
from llama_index.llm_predictor.vellum.types import (
    VellumDeployment,
    VellumCompiledPrompt,
)
from llama_index.utils import globals_helper


class VellumPredictor(BaseLLMPredictor):
    def __init__(
        self,
        vellum_api_key: str,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        # Needed by BaseLLMPredictor
        self._total_tokens_used = 0
        self._last_token_usage: Optional[int] = None
        self.callback_manager = callback_manager or CallbackManager([])

        # Vellum-specific
        self._vellum_client = Vellum(api_key=vellum_api_key)
        self._async_vellum_client = AsyncVellum(api_key=vellum_api_key)
        self._prompt_registry = VellumPromptRegistry(vellum_api_key=vellum_api_key)

    def predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Predict the answer to a query."""

        deployment, compiled_prompt, event_id = self._prepare_generate_call(
            prompt, **prompt_args
        )

        result = self._vellum_client.generate(
            deployment_id=deployment.deployment_id,
            requests=[
                vellum.GenerateRequest(
                    input_values=prompt.get_full_format_args(prompt_args)
                )
            ],
        )

        completion_text = self._process_generate_response(
            result, compiled_prompt, event_id
        )

        return completion_text, compiled_prompt.text

    def stream(self, prompt: Prompt, **prompt_args: Any) -> Tuple[Generator, str]:
        """Stream the answer to a query."""

        deployment, compiled_prompt, event_id = self._prepare_generate_call(
            prompt, **prompt_args
        )

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
            self._increment_token_usage(text=compiled_prompt.text)
            complete_text = ""

            while True:
                try:
                    stream_response = next(responses)
                except StopIteration:
                    self.callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            "response": complete_text,
                            "formatted_prompt": compiled_prompt.text,
                        },
                        event_id=event_id,
                    )
                    break

                result: vellum.GenerateStreamResult = stream_response.delta

                if result.error:
                    raise VellumGenerateException(result.error.message)
                elif not result.data:
                    raise VellumGenerateException(
                        "Unknown error occurred while generating"
                    )

                completion_text_delta = result.data.completion.text
                complete_text += completion_text_delta

                self._increment_token_usage(text=completion_text_delta)

                yield completion_text_delta

        return text_generator(), compiled_prompt.text

    async def apredict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Asynchronously predict the answer to a query."""

        deployment, compiled_prompt, event_id = self._prepare_generate_call(
            prompt, **prompt_args
        )

        result = await self._async_vellum_client.generate(
            deployment_id=deployment.deployment_id,
            requests=[
                vellum.GenerateRequest(
                    input_values=prompt.get_full_format_args(prompt_args)
                )
            ],
        )

        completion_text = self._process_generate_response(
            result, compiled_prompt, event_id
        )

        return completion_text, compiled_prompt.text

    def get_llm_metadata(self) -> LLMMetadata:
        """Get LLM metadata."""

        # TODO: Consider whether Vellum should use non-standard values
        return LLMMetadata(max_input_size=MAX_CHUNK_SIZE, num_output=NUM_OUTPUTS)

    @property
    def total_tokens_used(self) -> int:
        """Get the total tokens used so far."""
        return self._total_tokens_used

    @property
    def last_token_usage(self) -> int:
        """Get the last token usage."""
        if self._last_token_usage is None:
            return 0
        return self._last_token_usage

    @last_token_usage.setter
    def last_token_usage(self, value: int) -> None:
        """Set the last token usage."""
        self._last_token_usage = value

    def _get_compiled_prompt(
        self, deployment: VellumDeployment, **prompt_args: Any
    ) -> VellumCompiledPrompt:
        """Retrieve the final, compiled prompt used by Vellum."""

        # TODO: Expose endpoint for retrieving a compiled prompt and
        #  the number of tokens it uses
        return VellumCompiledPrompt(text="")

    def _prepare_generate_call(
        self, prompt: Prompt, **prompt_args: Any
    ) -> Tuple[VellumDeployment, VellumCompiledPrompt, str]:
        """Prepare a generate call."""

        deployment = self._prompt_registry.from_prompt(prompt)
        compiled_prompt = self._get_compiled_prompt(deployment, **prompt_args)

        cb_payload = {
            **prompt_args,
            "deployment_id": deployment.deployment_id,
        }
        event_id = self.callback_manager.on_event_start(
            CBEventType.LLM,
            payload=cb_payload,
        )
        return deployment, compiled_prompt, event_id

    def _process_generate_response(
        self,
        result: vellum.GenerateResponse,
        compiled_prompt: VellumCompiledPrompt,
        event_id: str,
    ) -> str:
        """Process the response from a generate call."""

        completion_text = result.text

        self._increment_token_usage(text=compiled_prompt.text)
        self._increment_token_usage(text=completion_text)

        self.callback_manager.on_event_end(
            CBEventType.LLM,
            payload={
                "response": completion_text,
                "formatted_prompt": compiled_prompt.text,
            },
            event_id=event_id,
        )

        return completion_text

    def _increment_token_usage(
        self, text: Optional[str] = None, num_tokens: Optional[int] = None
    ) -> None:
        """Update internal state to track token usage."""

        if text is not None and num_tokens is not None:
            raise ValueError("Only one of text and num_tokens can be specified")

        if text is not None:
            num_tokens = self._count_tokens(text)

        self._total_tokens_used += num_tokens or 0

    @staticmethod
    def _count_tokens(text: str) -> int:
        # This is considered an approximation of the number of tokens used.
        # As a future improvement, Vellum will make it possible to get back the
        # exact number of tokens used via API.
        tokens = globals_helper.tokenizer(text)
        return len(tokens)
