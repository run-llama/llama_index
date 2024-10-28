from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import httpx
from anthropic.types import ContentBlockDeltaEvent
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_TEMPERATURE,
)
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.anthropic.utils import (
    generate_anthropic_multi_modal_chat_message,
    resolve_anthropic_credentials,
)


DEFAULT_ANTHROPIC_MODEL = "claude-3-opus-20240229"
DEFAULT_ANTHROPIC_MAX_TOKENS = 512


class AnthropicMultiModal(MultiModalLLM):
    model: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL,
        description="The Anthropic multi-modal model to use.",
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        description="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.",
        gt=0,
    )
    context_window: Optional[int] = Field(
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    base_url: Optional[str] = Field(
        default=None, description="The Anthropic API base URL."
    )
    timeout: Optional[float] = Field(
        default=None,
        description="API timeout to use, in seconds.",
        ge=0,
    )
    max_retries: int = Field(
        default=10,
        description="Maximum number of API retries.",
        ge=0,
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Anthropic API."
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()
    _client: Union[
        anthropic.Anthropic, anthropic.AnthropicVertex, anthropic.AnthropicBedrock
    ] = PrivateAttr()
    _aclient: Union[
        anthropic.AsyncAnthropic,
        anthropic.AsyncAnthropicVertex,
        anthropic.AsyncAnthropicBedrock,
    ] = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = DEFAULT_ANTHROPIC_MAX_TOKENS,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
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

        api_key, base_url, api_version = resolve_anthropic_credentials(
            api_key=api_key,
            api_base=base_url,
            api_version=api_version,
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
        return "Anthropic_Multi_Modal_LLM"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            context_window=anthropic_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            is_function_calling_model=is_function_calling_model(self.model),
        )

    @property
    def tokenizer(self) -> Tokenizer:
        return self._client.get_tokenizer()

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def _get_content_and_tool_calls(
        self, response: Any
    ) -> Tuple[str, List[ToolUseBlock]]:
        tool_calls = []
        content = ""
        for content_block in response.content:
            if isinstance(content_block, TextBlock):
                content += content_block.text
            elif isinstance(content_block, ToolUseBlock):
                tool_calls.append(content_block.dict())

        return content, tool_calls

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

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.messages.create(
            messages=anthropic_messages,
            stream=False,
            system=system_prompt,
            **all_kwargs,
        )

        content, tool_calls = self._get_content_and_tool_calls(response)

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    # # Model Params for Anthropic Multi Modal model.
    # def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
    #     if self.model not in ANTHROPIC_MULTI_MODAL_MODELS:
    #         raise ValueError(
    #             f"Invalid model {self.model}. "
    #             f"Available models are: {list(ANTHROPIC_MULTI_MODAL_MODELS.keys())}"
    #         )
    #     base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
    #     if self.max_tokens is not None:
    #         base_kwargs["max_tokens"] = self.max_tokens
    #     return {**base_kwargs, **self.additional_kwargs}

    # def _get_response_token_counts(self, raw_response: Any) -> dict:
    #     """Get the token usage reported by the response."""
    #     if not isinstance(raw_response, dict):
    #         return {}
    #
    #     usage = raw_response.get("usage", {})
    #     # NOTE: other model providers that use the Anthropic client may not report usage
    #     if usage is None:
    #         return {}
    #
    #     return {
    #         "prompt_tokens": usage.get("prompt_tokens", 0),
    #         "completion_tokens": usage.get("completion_tokens", 0),
    #         "total_tokens": usage.get("total_tokens", 0),
    #     }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.messages.create(
            messages=anthropic_messages,
            stream=False,
            system=system_prompt,
            **all_kwargs,
        )

        content, tool_calls = self._get_content_and_tool_calls(response)

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=dict(response),
        )

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
