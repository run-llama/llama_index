from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from ollama import Client, AsyncClient
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.base.llms.generic_utils import (
    chat_to_completion_decorator,
    achat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    astream_chat_to_completion_decorator,
)
from llama_index.core.tools import ToolSelection

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

DEFAULT_REQUEST_TIMEOUT = 30.0


def get_additional_kwargs(
    response: Dict[str, Any], exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    return {k: v for k, v in response.items() if k not in exclude}


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


class Ollama(FunctionCallingLLM):
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
    is_function_calling_model: bool = Field(
        default=True,
        description="Whether the model is a function calling model.",
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
        is_function_calling_model: bool = True,
        **kwargs: Any,
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
            is_function_calling_model=is_function_calling_model,
            **kwargs,
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
            # TODO: Detect if selected model is a function calling model?
            is_function_calling_model=self.is_function_calling_model,
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

    def _get_response_token_counts(self, raw_response: dict) -> dict:
        """Get the token usage reported by the response."""
        try:
            prompt_tokens = raw_response["prompt_eval_count"]
            completion_tokens = raw_response["eval_count"]
            total_tokens = prompt_tokens + completion_tokens
        except KeyError:
            return {}
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: List["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Validate the response from chat_with_tools."""
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            argument_dict = tool_call["function"]["arguments"]

            tool_selections.append(
                ToolSelection(
                    # tool ids not provided by Ollama
                    tool_id=tool_call["function"]["name"],
                    tool_name=tool_call["function"]["name"],
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

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

        tool_calls = response["message"].get("tool_calls", [])
        token_counts = self._get_response_token_counts(response)
        if token_counts:
            response["usage"] = token_counts

        return ChatResponse(
            message=ChatMessage(
                content=response["message"]["content"],
                role=response["message"]["role"],
                additional_kwargs={"tool_calls": tool_calls},
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

                tool_calls = r["message"].get("tool_calls", [])
                token_counts = self._get_response_token_counts(r)
                if token_counts:
                    r["usage"] = token_counts

                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=r["message"]["role"],
                        additional_kwargs={"tool_calls": tool_calls},
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

                tool_calls = r["message"].get("tool_calls", [])
                token_counts = self._get_response_token_counts(r)
                if token_counts:
                    r["usage"] = token_counts

                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=r["message"]["role"],
                        additional_kwargs={"tool_calls": tool_calls},
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

        tool_calls = response["message"].get("tool_calls", [])
        token_counts = self._get_response_token_counts(response)
        if token_counts:
            response["usage"] = token_counts

        return ChatResponse(
            message=ChatMessage(
                content=response["message"]["content"],
                role=response["message"]["role"],
                additional_kwargs={"tool_calls": tool_calls},
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
