from __future__ import annotations

from typing import Any, Tuple, Generator, Optional, cast

from llama_index import Prompt
from llama_index.callbacks import CallbackManager, CBEventType
from llama_index.llm_predictor.base import BaseLLMPredictor, LLMMetadata
from llama_index.llm_predictor.vellum.exceptions import VellumGenerateException
from llama_index.llm_predictor.vellum.prompt_registry import VellumPromptRegistry
from llama_index.llm_predictor.vellum.types import (
    VellumCompiledPrompt,
    VellumRegisteredPrompt,
)
from llama_index.utils import globals_helper


class VellumPredictor(BaseLLMPredictor):
    def __init__(
        self,
        vellum_api_key: str,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        import_err_msg = (
            "`vellum` package not found, please run `pip install vellum-ai`"
        )
        try:
            from vellum.client import Vellum, AsyncVellum  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

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

        from vellum import GenerateRequest

        registered_prompt, compiled_prompt, event_id = self._prepare_generate_call(
            prompt, **prompt_args
        )

        result = self._vellum_client.generate(
            deployment_id=registered_prompt.deployment_id,
            requests=[
                GenerateRequest(input_values=prompt.get_full_format_args(prompt_args))
            ],
        )

        completion_text = self._process_generate_response(
            result, compiled_prompt, event_id
        )

        return completion_text, compiled_prompt.text

    def stream(self, prompt: Prompt, **prompt_args: Any) -> Tuple[Generator, str]:
        """Stream the answer to a query."""

        from vellum import GenerateRequest, GenerateStreamResult

        registered_prompt, compiled_prompt, event_id = self._prepare_generate_call(
            prompt, **prompt_args
        )

        responses = self._vellum_client.generate_stream(
            deployment_id=registered_prompt.deployment_id,
            requests=[
                GenerateRequest(input_values=prompt.get_full_format_args(prompt_args))
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

                result: GenerateStreamResult = stream_response.delta

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

        from vellum import GenerateRequest

        registered_prompt, compiled_prompt, event_id = self._prepare_generate_call(
            prompt, **prompt_args
        )

        result = await self._async_vellum_client.generate(
            deployment_id=registered_prompt.deployment_id,
            requests=[
                GenerateRequest(input_values=prompt.get_full_format_args(prompt_args))
            ],
        )

        completion_text = self._process_generate_response(
            result, compiled_prompt, event_id
        )

        return completion_text, compiled_prompt.text

    def get_llm_metadata(self) -> LLMMetadata:
        """Get LLM metadata."""

        # Note: We use default values here, but ideally we would retrieve this metadata
        # via Vellum's API based on the LLM that backs the registered prompt's
        # deployment. This is not currently possible, so we use default values.
        return LLMMetadata()

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

    def _prepare_generate_call(
        self, prompt: Prompt, **prompt_args: Any
    ) -> Tuple[VellumRegisteredPrompt, VellumCompiledPrompt, str]:
        """Prepare a generate call."""

        registered_prompt = self._prompt_registry.from_prompt(prompt)
        compiled_prompt = self._prompt_registry.get_compiled_prompt(
            registered_prompt, prompt_args
        )

        cb_payload = {
            **prompt_args,
            "deployment_id": registered_prompt.deployment_id,
            "model_version_id": registered_prompt.model_version_id,
        }
        event_id = self.callback_manager.on_event_start(
            CBEventType.LLM,
            payload=cb_payload,
        )
        return registered_prompt, compiled_prompt, event_id

    def _process_generate_response(
        self,
        result: Any,
        compiled_prompt: VellumCompiledPrompt,
        event_id: str,
    ) -> str:
        """Process the response from a generate call."""

        from vellum import GenerateResponse

        result = cast(GenerateResponse, result)

        completion_text = result.text

        self._increment_token_usage(num_tokens=compiled_prompt.num_tokens)
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
