from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Union,
)

import httpx

from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
    TextBlock,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

DEFAULT_XORTRON_BASE_URL = "http://localhost:8000"
DEFAULT_REQUEST_TIMEOUT = 60.0


class Xortron(CustomLLM):
    """
    Xortron LLM inference integration.

    Connects to a Xortron inference server for text generation.

    Examples:
        `pip install llama-index-llms-xortron`

        ```python
        from llama_index.llms.xortron import Xortron

        llm = Xortron(model="xortron-7b", base_url="http://localhost:8000")

        response = llm.complete("What is the capital of France?")
        print(response)
        ```

    """

    base_url: str = Field(
        default=DEFAULT_XORTRON_BASE_URL,
        description="Base URL for the Xortron inference server.",
    )
    model: str = Field(
        default="xortron-default",
        description="The Xortron model to use for inference.",
    )
    temperature: float = Field(
        default=0.7,
        description="The temperature to use for sampling.",
    )
    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authenticating with the Xortron server.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the Xortron API.",
    )
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for HTTP requests to the Xortron server.",
    )

    _client: Optional[httpx.Client] = PrivateAttr(default=None)
    _async_client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)

    def __init__(
        self,
        model: str = "xortron-default",
        base_url: str = DEFAULT_XORTRON_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        api_key: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            api_key=api_key,
            additional_kwargs=additional_kwargs or {},
            request_timeout=request_timeout,
            **kwargs,
        )
        self._client = None
        self._async_client = None

    @classmethod
    def class_name(cls) -> str:
        return "Xortron_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.request_timeout,
                headers=self._get_headers(),
            )
        return self._client

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.request_timeout,
                headers=self._get_headers(),
            )
        return self._async_client

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_kwargs,
            **kwargs,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if messages is not None:
            payload["messages"] = messages
        return payload

    def _convert_messages(
        self, messages: Sequence[ChatMessage]
    ) -> List[Dict[str, str]]:
        converted = []
        for message in messages:
            content = ""
            for block in message.blocks:
                if isinstance(block, TextBlock):
                    content += block.text
            converted.append({"role": message.role.value, "content": content})
        return converted

    def _parse_completion_response(self, data: Dict[str, Any]) -> CompletionResponse:
        text = data.get("text", data.get("output", data.get("completion", "")))
        return CompletionResponse(
            text=text,
            raw=data,
        )

    def _parse_chat_response(self, data: Dict[str, Any]) -> ChatResponse:
        message_data = data.get("message", data)
        text = message_data.get(
            "content", message_data.get("text", data.get("output", ""))
        )
        role = message_data.get("role", MessageRole.ASSISTANT)

        return ChatResponse(
            message=ChatMessage(
                role=role,
                blocks=[TextBlock(text=text)],
            ),
            raw=data,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        payload = self._build_payload(prompt=prompt, **kwargs)
        response = self.client.post("/v1/completions", json=payload)
        response.raise_for_status()
        return self._parse_completion_response(response.json())

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        payload = self._build_payload(prompt=prompt, **kwargs)
        response = await self.async_client.post("/v1/completions", json=payload)
        response.raise_for_status()
        return self._parse_completion_response(response.json())

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        payload = self._build_payload(prompt=prompt, stream=True, **kwargs)

        def gen() -> Generator[CompletionResponse, None, None]:
            with self.client.stream("POST", "/v1/completions", json=payload) as resp:
                resp.raise_for_status()
                full_text = ""
                for line in resp.iter_lines():
                    if not line or line.startswith("data: [DONE]"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        import json

                        chunk = json.loads(line)
                        delta = chunk.get(
                            "text", chunk.get("delta", chunk.get("token", ""))
                        )
                        full_text += delta
                        yield CompletionResponse(
                            text=full_text,
                            delta=delta,
                            raw=chunk,
                        )
                    except (ValueError, KeyError):
                        continue

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        payload = self._build_payload(prompt=prompt, stream=True, **kwargs)

        async def gen() -> AsyncGenerator[CompletionResponse, None]:
            async with self.async_client.stream(
                "POST", "/v1/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                full_text = ""
                async for line in resp.aiter_lines():
                    if not line or line.startswith("data: [DONE]"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        import json

                        chunk = json.loads(line)
                        delta = chunk.get(
                            "text", chunk.get("delta", chunk.get("token", ""))
                        )
                        full_text += delta
                        yield CompletionResponse(
                            text=full_text,
                            delta=delta,
                            raw=chunk,
                        )
                    except (ValueError, KeyError):
                        continue

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        converted = self._convert_messages(messages)
        payload = self._build_payload(messages=converted, **kwargs)
        response = self.client.post("/v1/chat", json=payload)
        response.raise_for_status()
        return self._parse_chat_response(response.json())

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        converted = self._convert_messages(messages)
        payload = self._build_payload(messages=converted, **kwargs)
        response = await self.async_client.post("/v1/chat", json=payload)
        response.raise_for_status()
        return self._parse_chat_response(response.json())

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        converted = self._convert_messages(messages)
        payload = self._build_payload(messages=converted, stream=True, **kwargs)

        def gen() -> Generator[ChatResponse, None, None]:
            with self.client.stream("POST", "/v1/chat", json=payload) as resp:
                resp.raise_for_status()
                full_text = ""
                for line in resp.iter_lines():
                    if not line or line.startswith("data: [DONE]"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        import json

                        chunk = json.loads(line)
                        delta = chunk.get(
                            "text", chunk.get("delta", chunk.get("token", ""))
                        )
                        full_text += delta
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                blocks=[TextBlock(text=full_text)],
                            ),
                            delta=delta,
                            raw=chunk,
                        )
                    except (ValueError, KeyError):
                        continue

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        converted = self._convert_messages(messages)
        payload = self._build_payload(messages=converted, stream=True, **kwargs)

        async def gen() -> AsyncGenerator[ChatResponse, None]:
            async with self.async_client.stream(
                "POST", "/v1/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                full_text = ""
                async for line in resp.aiter_lines():
                    if not line or line.startswith("data: [DONE]"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        import json

                        chunk = json.loads(line)
                        delta = chunk.get(
                            "text", chunk.get("delta", chunk.get("token", ""))
                        )
                        full_text += delta
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                blocks=[TextBlock(text=full_text)],
                            ),
                            delta=delta,
                            raw=chunk,
                        )
                    except (ValueError, KeyError):
                        continue

        return gen()
