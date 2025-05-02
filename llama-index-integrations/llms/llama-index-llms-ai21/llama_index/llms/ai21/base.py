from typing import Any, Callable, Dict, Optional, Sequence, List, Union

from ai21 import AI21Client, AsyncAI21Client
from ai21.models.chat import ChatCompletionChunk, ToolCall
from ai21_tokenizer import Tokenizer, BaseTokenizer  # pants: no-infer-dep
from llama_index.core.base.llms.generic_utils import (
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.program.openai.utils import parse_partial_json

from llama_index.llms.ai21.utils import (
    ai21_model_to_context_size,
    message_to_ai21_message,
    message_to_ai21_j2_message,
    is_function_calling_model,
    from_ai21_message_to_chat_message,
)

_DEFAULT_AI21_MODEL = "jamba-1.5-mini"
_DEFAULT_TEMPERATURE = 0.4
_DEFAULT_MAX_TOKENS = 512

_TOKENIZER_MAP = {
    "j2-ultra": "j2-tokenizer",
    "j2-mid": "j2-tokenizer",
    "jamba-instruct": "jamba-instruct-tokenizer",
    "jamba-1.5-mini": "jamba-1.5-mini-tokenizer",
    "jamba-1.5-large": "jamba-1.5-large-tokenizer",
}


class AI21(FunctionCallingLLM):
    """AI21 Labs LLM.

    Examples:
        `pip install llama-index-llms-ai21`

        ```python
        from llama_index.llms.ai21 import AI21

        llm = AI21(model="jamba-instruct", api_key=api_key)
        resp = llm.complete("Paul Graham is ")
        print(resp)
        ```
    """

    model: str = Field(
        description="The AI21 model to use.", default=_DEFAULT_AI21_MODEL
    )
    max_tokens: int = Field(
        description="The maximum number of tokens to generate.",
        default=_DEFAULT_MAX_TOKENS,
        gt=0,
    )
    temperature: float = Field(
        description="The temperature to use for sampling.",
        default=_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=1.0,
    )
    base_url: Optional[str] = Field(default=None, description="The base URL to use.")
    timeout: Optional[float] = Field(
        default=None, description="The timeout to use in seconds.", ge=0
    )

    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", ge=0
    )

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the anthropic API."
    )

    _client: Any = PrivateAttr()
    _async_client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = _DEFAULT_AI21_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = _DEFAULT_MAX_TOKENS,
        max_retries: int = 10,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = _DEFAULT_TEMPERATURE,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """Initialize params."""
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._client = AI21Client(
            api_key=api_key,
            api_host=base_url,
            timeout_sec=timeout,
            num_retries=max_retries,
            headers=default_headers,
            via="llama-index",
        )

        self._async_client = AsyncAI21Client(
            api_key=api_key,
            api_host=base_url,
            timeout_sec=timeout,
            num_retries=max_retries,
            headers=default_headers,
            via="llama-index",
        )

    @classmethod
    def class_name(cls) -> str:
        """Get Class Name."""
        return "AI21_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=ai21_model_to_context_size(self.model),
            num_output=self.max_tokens,
            model_name=self.model,
            is_function_calling_model=is_function_calling_model(
                model=self.model,
            ),
            is_chat_model=True,
        )

    @property
    def tokenizer(self) -> BaseTokenizer:
        return Tokenizer.get_tokenizer(_TOKENIZER_MAP.get(self.model))

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        if self._is_j2_model():
            return self._j2_completion(prompt, formatted, **all_kwargs)

        completion_fn = chat_to_completion_decorator(self.chat)

        return completion_fn(prompt, **all_kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        if self._is_j2_model():
            raise ValueError("Stream completion is not supported for J2 models.")

        all_kwargs = self._get_all_kwargs(**kwargs)
        completion_fn = stream_chat_to_completion_decorator(self.stream_chat)

        return completion_fn(prompt, **all_kwargs)

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tool_specs = [tool.metadata.to_openai_tool() for tool in tools]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        if self._is_j2_model():
            return self._j2_chat(messages, **all_kwargs)

        messages = [message_to_ai21_message(message) for message in messages]
        response = self._client.chat.completions.create(
            messages=messages,
            stream=False,
            **all_kwargs,
        )

        message = from_ai21_message_to_chat_message(response.choices[0].message)

        return ChatResponse(
            message=message,
            raw=response.to_dict(),
        )

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        if self._is_j2_model():
            return await self._j2_async_chat(messages, **all_kwargs)

        messages = [message_to_ai21_message(message) for message in messages]
        response = await self._async_client.chat.completions.create(
            messages=messages,
            stream=False,
            **all_kwargs,
        )

        message = from_ai21_message_to_chat_message(response.choices[0].message)

        return ChatResponse(
            message=message,
            raw=response.to_dict(),
        )

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        if self._is_j2_model():
            raise ValueError("Async Stream chat is not supported for J2 models.")

        all_kwargs = self._get_all_kwargs(**kwargs)
        messages = [message_to_ai21_message(message) for message in messages]
        response = await self._async_client.chat.completions.create(
            messages=messages,
            stream=True,
            **all_kwargs,
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            role = MessageRole.ASSISTANT

            async for r in response:
                if isinstance(r, ChatCompletionChunk):
                    content_delta = r.choices[0].delta.content

                    if content_delta is None:
                        content += ""
                    else:
                        content += r.choices[0].delta.content

                    yield ChatResponse(
                        message=ChatMessage(role=role, content=content),
                        delta=content_delta,
                        raw=r.to_dict(),
                    )

        return gen()

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        if self._is_j2_model():
            return await self._j2_async_complete(prompt, formatted, **all_kwargs)

        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)

    def _j2_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        system, messages = message_to_ai21_j2_message(messages)
        response = self._client.chat.create(
            system=system,
            messages=messages,
            stream=False,
            **kwargs,
        )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.outputs[0].text,
            ),
            raw=response.to_dict(),
        )

    async def _j2_async_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        system, messages = message_to_ai21_j2_message(messages)
        response = await self._async_client.chat.create(
            system=system,
            messages=messages,
            stream=False,
            **kwargs,
        )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.outputs[0].text,
            ),
            raw=response.to_dict(),
        )

    async def _j2_async_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = await self._async_client.completion.create(
            prompt=prompt,
            stream=False,
            **kwargs,
        )

        return CompletionResponse(
            text=response.completions[0].data.text,
            raw=response.to_dict(),
        )

    def _j2_completion(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = self._client.completion.create(
            prompt=prompt,
            stream=False,
            **kwargs,
        )

        return CompletionResponse(
            text=response.completions[0].data.text,
            raw=response.to_dict(),
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if self._is_j2_model():
            raise ValueError("Stream chat is not supported for J2 models.")

        all_kwargs = self._get_all_kwargs(**kwargs)
        messages = [message_to_ai21_message(message) for message in messages]
        response = self._client.chat.completions.create(
            messages=messages,
            stream=True,
            **all_kwargs,
        )

        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT

            for r in response:
                if isinstance(r, ChatCompletionChunk):
                    content_delta = r.choices[0].delta.content

                    if content_delta is None:
                        content += ""
                    else:
                        content += r.choices[0].delta.content

                    yield ChatResponse(
                        message=ChatMessage(role=role, content=content),
                        delta=content_delta,
                        raw=r.to_dict(),
                    )

        return gen()

    def _is_j2_model(self) -> bool:
        return "j2" in self.model

    def _parse_tool(self, tool_call: ToolCall) -> ToolSelection:
        if not isinstance(tool_call, ToolCall):
            raise ValueError("Invalid tool_call object")

        if tool_call.type != "function":
            raise ValueError(f"Unsupported tool call type: {tool_call.type}")

        try:
            argument_dict = parse_partial_json(tool_call.function.arguments)
        except ValueError:
            argument_dict = {}

        return ToolSelection(
            tool_id=tool_call.id,
            tool_name=tool_call.function.name,
            tool_kwargs=argument_dict,
        )

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        return [self._parse_tool(tool_call) for tool_call in tool_calls]
