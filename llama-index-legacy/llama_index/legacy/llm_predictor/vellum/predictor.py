from __future__ import annotations

from typing import Any, Tuple, cast

from deprecated import deprecated

from llama_index.legacy.bridge.pydantic import PrivateAttr
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.callbacks.schema import CBEventType, EventPayload
from llama_index.legacy.llm_predictor.base import LLM, BaseLLMPredictor, LLMMetadata
from llama_index.legacy.llm_predictor.vellum.exceptions import VellumGenerateException
from llama_index.legacy.llm_predictor.vellum.prompt_registry import VellumPromptRegistry
from llama_index.legacy.llm_predictor.vellum.types import (
    VellumCompiledPrompt,
    VellumRegisteredPrompt,
)
from llama_index.legacy.prompts import BasePromptTemplate
from llama_index.legacy.types import TokenAsyncGen, TokenGen


@deprecated("VellumPredictor is deprecated and will be removed in a future release.")
class VellumPredictor(BaseLLMPredictor):
    _callback_manager: CallbackManager = PrivateAttr(default_factory=CallbackManager)

    _vellum_client: Any = PrivateAttr()
    _async_vellum_client = PrivateAttr()
    _prompt_registry: Any = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vellum_api_key: str,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        import_err_msg = (
            "`vellum` package not found, please run `pip install vellum-ai`"
        )
        try:
            from vellum.client import AsyncVellum, Vellum
        except ImportError:
            raise ImportError(import_err_msg)

        self._callback_manager = callback_manager or CallbackManager([])

        # Vellum-specific
        self._vellum_client = Vellum(api_key=vellum_api_key)
        self._async_vellum_client = AsyncVellum(api_key=vellum_api_key)
        self._prompt_registry = VellumPromptRegistry(vellum_api_key=vellum_api_key)

        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        return "VellumPredictor"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # Note: We use default values here, but ideally we would retrieve this metadata
        # via Vellum's API based on the LLM that backs the registered prompt's
        # deployment. This is not currently possible, so we use default values.
        return LLMMetadata()

    @property
    def callback_manager(self) -> CallbackManager:
        """Get callback manager."""
        return self._callback_manager

    @property
    def llm(self) -> LLM:
        """Get the LLM."""
        raise NotImplementedError("Vellum does not expose the LLM.")

    def predict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        """Predict the answer to a query."""
        from vellum import GenerateRequest

        registered_prompt, compiled_prompt, event_id = self._prepare_generate_call(
            prompt, **prompt_args
        )

        input_values = {
            **prompt.kwargs,
            **prompt_args,
        }
        result = self._vellum_client.generate(
            deployment_id=registered_prompt.deployment_id,
            requests=[GenerateRequest(input_values=input_values)],
        )

        return self._process_generate_response(result, compiled_prompt, event_id)

    def stream(self, prompt: BasePromptTemplate, **prompt_args: Any) -> TokenGen:
        """Stream the answer to a query."""
        from vellum import GenerateRequest, GenerateStreamResult

        registered_prompt, compiled_prompt, event_id = self._prepare_generate_call(
            prompt, **prompt_args
        )

        input_values = {
            **prompt.kwargs,
            **prompt_args,
        }
        responses = self._vellum_client.generate_stream(
            deployment_id=registered_prompt.deployment_id,
            requests=[GenerateRequest(input_values=input_values)],
        )

        def text_generator() -> TokenGen:
            complete_text = ""

            while True:
                try:
                    stream_response = next(responses)
                except StopIteration:
                    self.callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            EventPayload.RESPONSE: complete_text,
                            EventPayload.PROMPT: compiled_prompt.text,
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

                yield completion_text_delta

        return text_generator()

    async def apredict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        """Asynchronously predict the answer to a query."""
        from vellum import GenerateRequest

        registered_prompt, compiled_prompt, event_id = self._prepare_generate_call(
            prompt, **prompt_args
        )

        input_values = {
            **prompt.kwargs,
            **prompt_args,
        }
        result = await self._async_vellum_client.generate(
            deployment_id=registered_prompt.deployment_id,
            requests=[GenerateRequest(input_values=input_values)],
        )

        return self._process_generate_response(result, compiled_prompt, event_id)

    async def astream(
        self, prompt: BasePromptTemplate, **prompt_args: Any
    ) -> TokenAsyncGen:
        async def gen() -> TokenAsyncGen:
            for token in self.stream(prompt, **prompt_args):
                yield token

        # NOTE: convert generator to async generator
        return gen()

    def _prepare_generate_call(
        self, prompt: BasePromptTemplate, **prompt_args: Any
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

        self.callback_manager.on_event_end(
            CBEventType.LLM,
            payload={
                EventPayload.RESPONSE: completion_text,
                EventPayload.PROMPT: compiled_prompt.text,
            },
            event_id=event_id,
        )

        return completion_text
