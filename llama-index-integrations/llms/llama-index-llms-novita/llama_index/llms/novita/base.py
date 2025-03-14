import json

from llama_index.core.llms.llm import ToolSelection
from openai import OpenAI, AsyncOpenAI
from openai.types import Model
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
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
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.base.llms.generic_utils import (
    chat_to_completion_decorator,
    achat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    astream_chat_to_completion_decorator,
)


if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

DEFAULT_REQUEST_TIMEOUT = 30.0


def is_function_calling_model(model: str) -> bool:
    function_calling_models = {"deepseek_v3", "deepseek-r1-turbo", "deepseek-v3-turbo", "qwq-32b"}
    return any(model_name in model for model_name in function_calling_models)


class NovitaAI(FunctionCallingLLM):
    """NovitaAI LLM.
    Visit https://novita.ai to get more information about Novita.
    """

    model: str = Field(description="The NovitaAI model to use.")
    api_key: Optional[str] = Field(
        default=None,
        description="The API key to use for the NovitaAI API.",
    )
    temperature: float = Field(
        default=0.95,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=1024,
        description="The maximum number of tokens for model output.",
        gt=0
    )
    timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to NovitaAI API server",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default=None,
        description="Additional kwargs for the NovitaAI API."
    )
    _client: Optional[OpenAI] = PrivateAttr()
    _async_client: Optional[AsyncOpenAI] = PrivateAttr()
    _context_size: Optional[int] = PrivateAttr()

    def __init__(
            self,
            model: str,
            api_key: str,
            temperature: float = 0.95,
            max_tokens: int = 1024,
            timeout: float = DEFAULT_REQUEST_TIMEOUT,
            additional_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            additional_kwargs=additional_kwargs,
            **kwargs,
        )
        base_url = "https://api.novita.ai/v3/openai"
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.additional_kwargs = additional_kwargs or {}
        self._context_size = self._client.models.retrieve(model=model, timeout=timeout).context_size

    @classmethod
    def class_name(cls) -> str:
        return "NovitaAI"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._context_size,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=is_function_calling_model(self.model),
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @property
    def available_models(self) -> List[Model]:
        return self._client.models.list()

    @property
    def retrieve_model(self) -> Model:
        return self._client.models.retrieve(model=self.model, timeout=self.timeout)

    @staticmethod
    def _convert_to_llm_messages(messages: Sequence[ChatMessage]) -> List:
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

    def get_tool_calls_from_response(
            self,
            response: "CompletionResponse",
            error_on_no_tool_call: bool = True,
            **kwargs: Any,
    ) -> List[ToolSelection]:
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            return []

        tool_selections = []
        for tool_call in tool_calls:
            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_kwargs=json.loads(tool_call.function.arguments),
                )
            )

        return tool_selections

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messages_dict = self._convert_to_llm_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        raw_response = self._client.chat.completions.create(
            model=self.model,
            messages=messages_dict,
            stream=False,
            timeout=self.timeout,
            **all_kwargs,
        )
        tool_calls = raw_response.choices[0].message.tool_calls or []
        return ChatResponse(
            message=ChatMessage(
                content=raw_response.choices[0].message.content,
                role=raw_response.choices[0].message.role,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=raw_response
        )

    @llm_chat_callback()
    async def achat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        messages_dict = self._convert_to_llm_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        raw_response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages_dict,
            stream=False,
            timeout=self.timeout,
            **all_kwargs,
        )
        tool_calls = raw_response.choices[0].message.tool_calls or []
        return ChatResponse(
            message=ChatMessage(
                content=raw_response.choices[0].message.content,
                role=raw_response.choices[0].message.role,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=raw_response
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        messages_dict = self._convert_to_llm_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        def gen() -> ChatResponseGen:
            raw_response = self._client.chat.completions.create(
                model=self.model,
                messages=messages_dict,
                stream=True,
                timeout=self.timeout,
                **all_kwargs,
            )
            response_txt = ""
            for chunk in raw_response:
                if chunk.choices[0].delta.content is None:
                    continue
                response_txt += chunk.choices[0].delta.content
                tool_calls = chunk.choices[0].delta.tool_calls
                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=MessageRole.ASSISTANT,
                        additional_kwargs={"tool_calls": tool_calls},
                    ),
                    delta=chunk.choices[0].delta.content,
                    raw=chunk,
                )
        return gen()

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        messages_dict = self._convert_to_llm_messages(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        async def gen() -> ChatResponseAsyncGen:
            raw_response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages_dict,
                stream=True,
                timeout=self.timeout,
                **all_kwargs,
            )
            response_txt = ""
            async for chunk in raw_response:
                if chunk.choices[0].delta.content is None:
                    continue
                response_txt += chunk.choices[0].delta.content
                tool_calls = chunk.choices[0].delta.tool_calls
                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=MessageRole.ASSISTANT,
                        additional_kwargs={"tool_calls": tool_calls},
                    ),
                    delta=chunk.choices[0].delta.content,
                    raw=chunk,
                )

        return gen()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return chat_to_completion_decorator(self.chat)(prompt, **kwargs)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return await achat_to_completion_decorator(self.achat)(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        return stream_chat_to_completion_decorator(self.stream_chat)(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        return await astream_chat_to_completion_decorator(self.astream_chat)(prompt, **kwargs)
