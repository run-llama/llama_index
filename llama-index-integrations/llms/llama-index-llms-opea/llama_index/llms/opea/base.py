import httpx
from typing import Any, AsyncGenerator, Dict, Generator, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.base.llms.generic_utils import (
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
)
from llama_index.core.llms.llm import LLM

from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class OPEA(LLM):
    """Adapter for a OPEA LLM.

    Examples:
        `pip install llama-index-llms-opea`

        ```python
        from llama_index.llms.opea import OPEA

        llm = OPEA(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            endpoint="http://localhost:8080",
        )
        ```
    """

    model: str = Field(description="The model name to use.")
    endpoint: str = Field(description="The endpoint to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE, description="The temperature to use."
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW, description="The context window to use."
    )
    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS, description="The max tokens to use."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs to use."
    )
    timeout: float = Field(default=30.0, description="The timeout to use.")

    @classmethod
    def class_name(cls) -> str:
        return "OPEA"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=2048,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
        )

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model": self.model,
            **self.additional_kwargs,
            **kwargs,
        }

    def _call(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatCompletionResponse:
        message_dicts = [message.model_dump() for message in messages]
        model_kwargs = self._get_model_kwargs(messages=message_dicts, **kwargs)
        request = ChatCompletionRequest(**model_kwargs)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.endpoint, json=request.model_dump())
        return ChatCompletionResponse.model_validate_json(response.json())

    async def _acall(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatCompletionResponse | ChatCompletionStreamResponse:
        message_dicts = [message.model_dump() for message in messages]
        model_kwargs = self._get_model_kwargs(messages=message_dicts, **kwargs)
        request = ChatCompletionRequest(**model_kwargs)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.endpoint, json=request.model_dump())
        return ChatCompletionResponse.model_validate_json(response.json())

    def _stream_call(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> Generator[ChatCompletionStreamResponse, None, None]:
        message_dicts = [message.model_dump() for message in messages]
        model_kwargs = self._get_model_kwargs(
            messages=message_dicts, stream=True, **kwargs
        )
        request = ChatCompletionRequest(**model_kwargs)

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST", self.endpoint, json=request.model_dump()
            ) as response:
                for line in response.iter_lines():
                    if line:
                        yield ChatCompletionStreamResponse.model_validate_json(line)

    async def _astream_call(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        message_dicts = [message.model_dump() for message in messages]
        model_kwargs = self._get_model_kwargs(
            messages=message_dicts, stream=True, **kwargs
        )
        request = ChatCompletionRequest(**model_kwargs)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", self.endpoint, json=request.model_dump()
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield ChatCompletionStreamResponse.model_validate_json(line)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        response = self._call(messages, **kwargs)

        return ChatResponse(
            message=ChatMessage(
                content=response.choices[0].message.content,
                role=response.choices[0].message.role,
                additional_kwargs=response.choices[0].metadata or {},
            ),
            raw=response.model_dump(),
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        chat_fn = chat_to_completion_decorator(self.chat)
        return chat_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        response_gen = self._stream_call(messages, **kwargs)

        def gen() -> ChatResponseGen:
            response_str = ""
            for response in response_gen:
                response_str += response.choices[0].delta.content or ""

                yield ChatResponse(
                    message=ChatMessage(
                        content=response_str,
                        role=response.choices[0].delta.role or "assistant",
                    ),
                    raw=response.model_dump(),
                    delta=response.choices[0].delta.content or "",
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        response = await self._acall(messages, **kwargs)

        return ChatResponse(
            message=ChatMessage(
                content=response.choices[0].message.content,
                role=response.choices[0].message.role,
                additional_kwargs=response.choices[0].metadata or {},
            ),
            raw=response.model_dump(),
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        chat_fn = achat_to_completion_decorator(self.achat)
        return await chat_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_gen = await self._astream_call(messages, **kwargs)

        async def gen() -> ChatResponseAsyncGen:
            response_str = ""
            async for response in response_gen:
                response_str += response.choices[0].delta.content or ""

                yield ChatResponse(
                    message=ChatMessage(
                        content=response_str,
                        role=response.choices[0].delta.role or "assistant",
                    ),
                    raw=response.model_dump(),
                    delta=response.choices[0].delta.content or "",
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        stream_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await stream_fn(prompt, **kwargs)
