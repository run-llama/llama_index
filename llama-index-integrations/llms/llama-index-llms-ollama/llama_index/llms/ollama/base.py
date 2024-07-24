from typing import Any, Dict, List, Optional, Sequence, Tuple

from ollama import Client, AsyncClient
from ollama import ChatResponse as OllamaChatResponse
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.generic_utils import (
    chat_to_completion_decorator,
    achat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    astream_chat_to_completion_decorator,
)
from llama_index.core.tools import ToolSelection

DEFAULT_REQUEST_TIMEOUT = 30.0


def get_additional_kwargs(
    response: Dict[str, Any], exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    return {k: v for k, v in response.items() if k not in exclude}


class Ollama(CustomLLM):
    """Ollama LLM.

    Visit https://ollama.com/ to download and install Ollama.

    Run `ollama serve` to start a server.

    Run `ollama pull <name>` to download a model to run.

    Examples:
        `pip install llama-index-llms-ollama`

        ```python
        from llama_index.llms.ollama import Ollama

        llm = Ollama(model="llama2", request_timeout=60.0)

        response = llm.complete("What is the capital of France?")
        print(response)
        ```
    """

    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted under.",
    )
    model: str = Field(description="The Ollama model to use.")
    temperature: float = Field(
        default=0.75,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to Ollama API server",
    )
    prompt_key: str = Field(
        default="prompt", description="The key to use for the prompt in API calls."
    )
    json_mode: bool = Field(
        default=False,
        description="Whether to use JSON mode for the Ollama API.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model parameters for the Ollama API.",
    )

    _client: Optional[Client] = PrivateAttr()
    _async_client: Optional[AsyncClient] = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.75,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        prompt_key: str = "prompt",
        json_mode: bool = False,
        additional_kwargs: Dict[str, Any] = {},
        client: Optional[Client] = None,
        async_client: Optional[AsyncClient] = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            temperature=temperature,
            context_window=context_window,
            request_timeout=request_timeout,
            prompt_key=prompt_key,
            json_mode=json_mode,
            additional_kwargs=additional_kwargs,
        )

        self._client = client
        self._async_client = async_client

    @classmethod
    def class_name(cls) -> str:
        return "Ollama_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=True,  # Ollama supports chat API for all models
        )

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(host=self.base_url, timeout=self.request_timeout)
        return self._client

    @property
    def async_client(self) -> AsyncClient:
        if self._async_client is None:
            self._async_client = AsyncClient(
                host=self.base_url, timeout=self.request_timeout
            )
        return self._async_client

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "num_ctx": self.context_window,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _convert_to_ollama_messages(self, messages: Sequence[ChatMessage]) -> Dict:
        return [
            {
                "role": message.role.value,
                "content": message.content or "",
            }
            for message in messages
        ]

    def _parse_tool_calls_from_response(
        response: OllamaChatResponse,
    ) -> List[ToolSelection]:
        return None

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)

        response = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            stream=False,
            format="json" if self.json_mode else "",
            tools=tools,
            options=self._model_kwargs,
        )

        return ChatResponse(
            message=ChatMessage(
                content=response["message"]["content"],
                role=response["message"]["role"],
            ),
            raw=response,
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)

        def gen() -> ChatResponseGen:
            response = self.client.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True,
                format="json" if self.json_mode else "",
                tools=tools,
                options=self._model_kwargs,
            )

            response_txt = ""

            for r in response:
                if r["message"]["content"] is None:
                    continue

                response_txt += r["message"]["content"]
                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=r["message"]["role"],
                    ),
                    delta=r["message"]["content"],
                    raw=r,
                )

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)

        async def gen() -> ChatResponseAsyncGen:
            response = await self.async_client.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True,
                format="json" if self.json_mode else "",
                tools=tools,
                options=self._model_kwargs,
            )

            response_txt = ""

            async for r in response:
                if r["message"]["content"] is None:
                    continue

                response_txt += r["message"]["content"]
                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=r["message"]["role"],
                    ),
                    delta=r["message"]["content"],
                    raw=r,
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)

        response = await self.async_client.chat(
            model=self.model,
            messages=ollama_messages,
            stream=False,
            format="json" if self.json_mode else "",
            tools=tools,
            options=self._model_kwargs,
        )

        return ChatResponse(
            message=ChatMessage(
                content=response["message"]["content"],
                role=response["message"]["role"],
            ),
            raw=response,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return chat_to_completion_decorator(self.chat)(prompt, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return await achat_to_completion_decorator(self.achat)(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return stream_chat_to_completion_decorator(self.stream_chat)(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await astream_chat_to_completion_decorator(self.astream_chat)(
            prompt, **kwargs
        )
