import json
import aiohttp
import functools
import requests
import tenacity
import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)
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
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_CONTEXT_WINDOW
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

DEFAULT_SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_REQUEST_TIMEOUT = 60
AVAILABLE_OPTIONS = [
    "deepseek-ai/DeepSeek-V2.5",
    "deepseek-ai/DeepSeek-V2-Chat",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "Qwen/Qwen2.5-72B-Instruct-128K",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Math-72B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-57B-A14B-Instruct",
    "TeleAI/TeleChat2",
    "TeleAI/TeleMM",
    "01-ai/Yi-1.5-34B-Chat-16K",
    "01-ai/Yi-1.5-9B-Chat-16K",
    "01-ai/Yi-1.5-6B-Chat",
    "THUDM/chatglm3-6b",
    "THUDM/glm-4-9b-chat",
    "Vendor-A/Qwen/Qwen2-72B-Instruct",
    "Vendor-A/Qwen/Qwen2.5-72B-Instruct",
    "internlm/internlm2_5-7b-chat",
    "internlm/internlm2_5-20b-chat",
    "OpenGVLab/InternVL2-Llama3-76B",
    "OpenGVLab/InternVL2-26B",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "Pro/Qwen/Qwen2.5-7B-Instruct",
    "Pro/Qwen/Qwen2-7B-Instruct",
    "Pro/Qwen/Qwen2-1.5B-Instruct",
    "Pro/Qwen/Qwen2-VL-7B-Instruct",
    "Pro/01-ai/Yi-1.5-9B-Chat-16K",
    "Pro/01-ai/Yi-1.5-6B-Chat",
    "Pro/THUDM/chatglm3-6b",
    "Pro/THUDM/glm-4-9b-chat",
    "Pro/internlm/internlm2_5-7b-chat",
    "Pro/OpenGVLab/InternVL2-8B",
    "Pro/meta-llama/Meta-Llama-3-8B-Instruct",
    "Pro/meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Pro/google/gemma-2-9b-it",
]
FUNCTION_CALLING_OPTIONS = [
    "deepseek-ai/DeepSeek-V2.5",
    "internlm/internlm2_5-20b-chat",
    "internlm/internlm2_5-7b-chat",
    "Pro/internlm/internlm2_5-7b-chat",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Pro/Qwen/Qwen2.5-7B-Instruct",
    "THUDM/glm-4-9b-chat",
    "Pro/THUDM/glm-4-9b-chat",
]


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


def is_function_calling_llm(model: str):
    return model in FUNCTION_CALLING_OPTIONS


def create_retry_decorator(
    max_retries: int,
    min_seconds: float = 1,
    max_seconds: float = 20,
    random_exponential: bool = True,
    stop_after_delay_seconds: Optional[float] = None,
) -> Callable[[Any], Any]:
    """Create a retry decorator with custom parameters."""
    return tenacity.Retrying(
        stop=(
            tenacity.stop_after_attempt(max_retries)
            if stop_after_delay_seconds is None
            else tenacity.stop_any(
                tenacity.stop_after_delay(stop_after_delay_seconds),
                tenacity.stop_after_attempt(max_retries),
            )
        ),
        wait=(
            tenacity.wait_random_exponential(min=min_seconds, max=max_seconds)
            if random_exponential
            else tenacity.wait_random(min=min_seconds, max=max_seconds)
        ),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True,
    )


def llm_retry_decorator(f: Callable[..., Any]) -> Callable[..., Any]:
    """Retry decorator for LLM calls."""

    @functools.wraps(f)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        max_retries = getattr(self, "max_retries", 0)
        if max_retries <= 0:
            return f(self, *args, **kwargs)

        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(max_retries),
            wait=tenacity.wait_random_exponential(min=1, max=20),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
        )
        return retryer(lambda: f(self, *args, **kwargs))

    @functools.wraps(f)
    async def async_wrapper(self, *args: Any, **kwargs: Any) -> Any:
        max_retries = getattr(self, "max_retries", 0)
        if max_retries <= 0:
            return await f(self, *args, **kwargs)

        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(max_retries),
            wait=tenacity.wait_random_exponential(min=1, max=20),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
        )
        return await retryer(lambda: f(self, *args, **kwargs))

    return async_wrapper if asyncio.iscoroutinefunction(f) else wrapper


class SiliconFlow(FunctionCallingLLM):
    """SiliconFlow LLM.

    Visit https://siliconflow.cn/ to get more information about SiliconFlow.

    Examples:
        `pip install llama-index-llms-siliconflow`

        ```python
        from llama_index.llms.siliconflow import SiliconFlow

        llm = SiliconFlow(api_key="YOUR API KEY")

        response = llm.complete("who are you?")
        print(response)
        ```
    """

    model: str = Field(
        default="deepseek-ai/DeepSeek-V2.5",
        description="The name of the model to query.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The API key to use for the SiliconFlow API.",
    )
    base_url: str = Field(
        default=DEFAULT_SILICONFLOW_API_URL,
        description="The base URL for the SiliconFlow API.",
    )
    temperature: float = Field(
        default=0.7,
        description="Determines the degree of randomness in the response.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=512,
        description="The maximum number of tokens to generate.",
    )
    frequency_penalty: float = Field(default=0.5)
    timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to ZhipuAI API server",
    )
    stop: Optional[str] = Field(
        default=None,
        description="Up to 4 sequences where the API will stop generating further tokens.",
    )
    max_retries: int = Field(
        default=3,
        description="The maximum number of API retries.",
        ge=0,
    )

    _headers: Any = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-ai/DeepSeek-V2.5",
        base_url: str = DEFAULT_SILICONFLOW_API_URL,
        temperature: float = 0.7,
        max_tokens: int = 512,
        frequency_penalty: float = 0.5,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        stop: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            timeout=timeout,
            stop=stop,
            max_retries=max_retries,
            **kwargs,
        )

        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @classmethod
    def class_name(cls) -> str:
        return "SiliconFlow"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=DEFAULT_CONTEXT_WINDOW,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=is_function_calling_llm(self.model),
        )

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop,
        }

    def _convert_to_llm_messages(self, messages: Sequence[ChatMessage]) -> List:
        return [
            {
                "role": message.role.value,
                "content": message.content or "",
            }
            for message in messages
        ]

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
        response: Union[ChatResponse, CompletionResponse],
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        if isinstance(response, ChatResponse):
            tool_calls = response.message.additional_kwargs.get("tool_calls", [])
        else:
            tool_calls = response.additional_kwargs.get("tool_calls", [])
        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} "
                    "tool calls."
                )
            return []

        tool_selections = []
        for tool_call in tool_calls:
            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call["id"],
                    tool_name=tool_call["function"]["name"],
                    tool_kwargs=json.loads(tool_call["function"]["arguments"]),
                )
            )

        return tool_selections

    @llm_chat_callback()
    @llm_retry_decorator
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messages_dict = self._convert_to_llm_messages(messages)
        response_format = kwargs.get("response_format", {"type": "text"})
        with requests.Session() as session:
            input_json = {
                "model": self.model,
                "messages": messages_dict,
                "stream": False,
                "n": 1,
                "tools": kwargs.get("tools", None),
                "response_format": response_format,
                **self.model_kwargs,
            }
            response = session.post(
                self.base_url,
                json=input_json,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_json = response.json()
            message: dict = response_json["choices"][0]["message"]
            return ChatResponse(
                message=ChatMessage(
                    content=message["content"],
                    role=message["role"],
                    additional_kwargs={"tool_calls": message.get("tool_calls")},
                ),
                raw=response_json,
            )

    @llm_chat_callback()
    @llm_retry_decorator
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        messages_dict = self._convert_to_llm_messages(messages)
        response_format = kwargs.get("response_format", {"type": "text"})
        async with aiohttp.ClientSession() as session:
            input_json = {
                "model": self.model,
                "messages": messages_dict,
                "stream": False,
                "n": 1,
                "tools": kwargs.get("tools", None),
                "response_format": response_format,
                **self.model_kwargs,
            }

            async with session.post(
                self.base_url,
                json=input_json,
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response_json = await response.json()
                message: dict = response_json["choices"][0]["message"]
                response.raise_for_status()
                return ChatResponse(
                    message=ChatMessage(
                        content=message["content"],
                        role=message["role"],
                        additional_kwargs={"tool_calls": message.get("tool_calls")},
                    ),
                    raw=response_json,
                )

    @llm_chat_callback()
    @llm_retry_decorator
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        messages_dict = self._convert_to_llm_messages(messages)
        response_format = kwargs.get("response_format", {"type": "text"})

        def gen() -> ChatResponseGen:
            with requests.Session() as session:
                input_json = {
                    "model": self.model,
                    "messages": messages_dict,
                    "stream": True,
                    "n": 1,
                    "tools": kwargs.get("tools", None),
                    "response_format": response_format,
                    **self.model_kwargs,
                }
                response = session.post(
                    self.base_url,
                    json=input_json,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response_txt = ""
                response_role = "assistant"
                for line in response.iter_lines():
                    line = cast(bytes, line).decode("utf-8")
                    if line.startswith("data:"):
                        if line.strip() == "data: [DONE]":
                            break
                        chunk_json = json.loads(line[5:])
                        delta: dict = chunk_json["choices"][0]["delta"]
                        response_role = delta.get("role") or response_role
                        response_txt += delta["content"]
                        tool_calls = delta.get("tool_calls")
                        yield ChatResponse(
                            message=ChatMessage(
                                content=response_txt,
                                role=response_role,
                                additional_kwargs={"tool_calls": tool_calls},
                            ),
                            delta=delta["content"],
                            raw=chunk_json,
                        )

        return gen()

    @llm_chat_callback()
    @llm_retry_decorator
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        messages_dict = self._convert_to_llm_messages(messages)
        response_format = kwargs.get("response_format", {"type": "text"})

        async def gen() -> ChatResponseAsyncGen:
            async with aiohttp.ClientSession(trust_env=True) as session:
                input_json = {
                    "model": self.model,
                    "messages": messages_dict,
                    "stream": True,
                    "n": 1,
                    "tools": kwargs.get("tools", None),
                    "response_format": response_format,
                    **self.model_kwargs,
                }
                async with session.post(
                    self.base_url,
                    json=input_json,
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    response_txt = ""
                    response_role = "assistant"
                    async for line in response.content.iter_any():
                        line = cast(bytes, line).decode("utf-8")
                        chunks = list(filter(None, line.split("data: ")))
                        for chunk in chunks:
                            if chunk.strip() == "[DONE]":
                                break
                            chunk_json = json.loads(chunk)
                            delta: dict = chunk_json["choices"][0]["delta"]
                            response_role = delta.get("role") or response_role
                            response_txt += delta["content"]
                            tool_calls = delta.get("tool_calls")
                            yield ChatResponse(
                                message=ChatMessage(
                                    content=response_txt,
                                    role=response_role,
                                    additional_kwargs={"tool_calls": tool_calls},
                                ),
                                delta=delta["content"],
                                raw=line,
                            )

        return gen()

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
