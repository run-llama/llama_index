from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import anthropic
from anthropic.types import ContentBlockDeltaEvent
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.anthropic.utils import (
    ANTHROPIC_MULTI_MODAL_MODELS,
    generate_anthropic_multi_modal_chat_message,
)


DEFAULT_ANTHROPIC_MULTIMODAL_MODEL = "claude-3-opus-20240229"
DEFAULT_ANTHROPIC_MAX_TOKENS = 512


class AnthropicMultiModal(MultiModalLLM):
    model: str = Field(
        default=DEFAULT_ANTHROPIC_MULTIMODAL_MODEL,
        description="The Anthropic multi-modal model to use.",
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: Optional[int] = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description=" The maximum numbers of tokens to generate.",
        gt=0,
    )

    base_url: str = Field(default=None, description="Anthropic API base URL.")
    timeout: float = Field(
        default=60.0,
        description="Timeout, in seconds, for API requests.",
        ge=0,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries.",
        ge=0,
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Anthropic API."
    )
    # context_window: Optional[int] = Field(
    #     description="The maximum number of context tokens for the model.",
    #     gt=0,
    # )
    # api_key: str = Field(
    #     default=None, description="The Anthropic API key.", exclude=True
    # )
    # system_prompt: str = Field(default="", description="System Prompt.")
    # base_url: str = Field(default=None, description="The base URL for Anthropic API.")
    # api_version: str = Field(description="The API version for Anthropic API.")
    # additional_kwargs: Dict[str, Any] = Field(
    #     default_factory=dict, description="Additional kwargs for the Anthropic API."
    # )
    # default_headers: Optional[Dict[str, str]] = Field(
    #     default=None, description="The default headers for API requests."
    # )

    _client: Union[
        anthropic.Anthropic, anthropic.AnthropicVertex, anthropic.AnthropicBedrock
    ] = PrivateAttr()
    _aclient: Union[
        anthropic.AsyncAnthropic,
        anthropic.AsyncAnthropicVertex,
        anthropic.AsyncAnthropicBedrock,
    ] = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MULTIMODAL_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = DEFAULT_NUM_OUTPUTS,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        region: Optional[str] = None,
        project_id: Optional[str] = None,
        aws_region: Optional[str] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])
        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or (lambda x: x)

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            model=model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            model=model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        if region and project_id and not aws_region:
            self._client = anthropic.AnthropicVertex(
                region=region,
                project_id=project_id,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )

            self._aclient = anthropic.AsyncAnthropicVertex(
                region=region,
                project_id=project_id,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )
        elif aws_region:
            # Note: this assumes you have AWS credentials configured.
            self._client = anthropic.AnthropicBedrock(
                aws_region=aws_region,
            )
            self._aclient = anthropic.AsyncAnthropicBedrock(
                aws_region=aws_region,
            )
        else:
            self._client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )
            self._aclient = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )

    @classmethod
    def class_name(cls) -> str:
        return "anthropic_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            context_window=a or DEFAULT_CONTEXT_WINDOW,
            num_output=self.max_tokens,
            model_name=self.model,
        )

    def _get_multi_modal_chat_messages(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[ImageNode],
        **kwargs: Any,
    ) -> List[Dict]:
        return generate_anthropic_multi_modal_chat_message(
            prompt=prompt,
            role=role,
            image_documents=image_documents,
        )

    # Model Params for Anthropic Multi Modal model.
    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        if self.model not in ANTHROPIC_MULTI_MODAL_MODELS:
            raise ValueError(
                f"Invalid model {self.model}. "
                f"Available models are: {list(ANTHROPIC_MULTI_MODAL_MODELS.keys())}"
            )
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        if self.max_tokens is not None:
            base_kwargs["max_tokens"] = self.max_tokens
        return {**base_kwargs, **self.additional_kwargs}

    def _get_response_token_counts(self, raw_response: Any) -> dict:
        """Get the token usage reported by the response."""
        if not isinstance(raw_response, dict):
            return {}

        usage = raw_response.get("usage", {})
        # NOTE: other model providers that use the Anthropic client may not report usage
        if usage is None:
            return {}

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def _complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        response = self._client.messages.create(
            messages=message_dict,
            system=self.system_prompt,
            stream=False,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.content[0].text,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        def gen() -> CompletionResponseGen:
            text = ""

            for response in self._client.messages.create(
                messages=message_dict,
                stream=True,
                system=self.system_prompt,
                **all_kwargs,
            ):
                if isinstance(response, ContentBlockDeltaEvent):
                    # update using deltas
                    content_delta = response.delta.text or ""
                    text += content_delta

                    yield CompletionResponse(
                        delta=content_delta,
                        text=text,
                        raw=response,
                        additional_kwargs=self._get_response_token_counts(response),
                    )

        return gen()

    def complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt, image_documents, **kwargs)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        return self._stream_complete(prompt, image_documents, **kwargs)

    def chat(
        self,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("This function is not yet implemented.")

    def stream_chat(
        self,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("This function is not yet implemented.")

    # ===== Async Endpoints =====

    async def _acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )
        response = await self._aclient.messages.create(
            messages=message_dict,
            stream=False,
            system=self.system_prompt,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.content[0].text,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        return await self._acomplete(prompt, image_documents, **kwargs)

    async def _astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        async def gen() -> CompletionResponseAsyncGen:
            text = ""

            async for response in await self._aclient.messages.create(
                messages=message_dict,
                stream=True,
                system=self.system_prompt,
                **all_kwargs,
            ):
                if isinstance(response, ContentBlockDeltaEvent):
                    # update using deltas
                    content_delta = response.delta.text or ""
                    text += content_delta

                    yield CompletionResponse(
                        delta=content_delta,
                        text=text,
                        raw=response,
                        additional_kwargs=self._get_response_token_counts(response),
                    )

        return gen()

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await self._astream_complete(prompt, image_documents, **kwargs)

    async def achat(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This function is not yet implemented.")

    async def astream_chat(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This function is not yet implemented.")
