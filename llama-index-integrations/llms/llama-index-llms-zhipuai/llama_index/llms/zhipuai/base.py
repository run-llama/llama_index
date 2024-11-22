import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from zhipuai import ZhipuAI as ZhipuAIClient
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
from llama_index.core.tools import ToolSelection

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

DEFAULT_REQUEST_TIMEOUT = 30.0
SUCCESS = "SUCCESS"
FAILED = "FAILED"
GLM_CHAT_MODELS = {
    "glm-4-plus": 128_000,
    "glm-4-0520": 128_000,
    "glm-4-long": 1_000_000,
    "glm-4-airx": 8_000,
    "glm-4-air": 128_000,
    "glm-4-flashx": 128_000,
    "glm-4-flash": 128_000,
    "glm-4v": 2_000,
    "glm-4-alltools": 128_000,
    "glm-4": 128_000,
}


def glm_model_to_context_size(model: str) -> Union[int, None]:
    token_limit = GLM_CHAT_MODELS.get(model, None)

    if token_limit is None:
        raise ValueError(f"Model name {model} not found in {GLM_CHAT_MODELS.keys()}")

    return token_limit


def is_function_calling_model(model: str) -> bool:
    return "4v" not in model


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


def async_llm_generate(item):
    try:
        return next(item)
    except StopIteration:
        return None


class ZhipuAI(FunctionCallingLLM):
    """ZhipuAI LLM.

    Visit https://open.bigmodel.cn to get more information about ZhipuAI.

    Examples:
        `pip install llama-index-llms-zhipuai`

        ```python
        from llama_index.llms.zhipuai import ZhipuAI

        llm = ZhipuAI(model="glm-4", api_key="YOUR API KEY")

        response = llm.complete("who are you?")
        print(response)
        ```
    """

    model: str = Field(description="The ZhipuAI model to use.")
    api_key: Optional[str] = Field(
        default=None,
        description="The API key to use for the ZhipuAI API.",
    )
    temperature: float = Field(
        default=0.95,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=1024,
        description="The maximum number of tokens for model output.",
        gt=0,
        le=4096,
    )
    timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to ZhipuAI API server",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the ZhipuAI API."
    )
    _client: Optional[ZhipuAIClient] = PrivateAttr()

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

        self._client = ZhipuAIClient(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "ZhipuAI"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=glm_model_to_context_size(self.model),
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=is_function_calling_model(self.model),
        )

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
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
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])
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
        raw_response = self._client.chat.completions.create(
            model=self.model,
            messages=messages_dict,
            stream=False,
            tools=kwargs.get("tools", None),
            tool_choice=kwargs.get("tool_choice", None),
            stop=kwargs.get("stop", None),
            timeout=self.timeout,
            extra_body=self.model_kwargs,
        )
        tool_calls = raw_response.choices[0].message.tool_calls or []
        return ChatResponse(
            message=ChatMessage(
                content=raw_response.choices[0].message.content,
                role=raw_response.choices[0].message.role,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=raw_response,
        )

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        messages_dict = self._convert_to_llm_messages(messages)
        raw_response = self._client.chat.asyncCompletions.create(
            model=self.model,
            messages=messages_dict,
            tools=kwargs.get("tools", None),
            tool_choice=kwargs.get("tool_choice", None),
            stop=kwargs.get("stop", None),
            timeout=self.timeout,
            extra_body=self.model_kwargs,
        )
        task_id = raw_response.id
        task_status = raw_response.task_status
        get_count = 0
        while task_status not in [SUCCESS, FAILED] and get_count < self.timeout:
            task_result = self._client.chat.asyncCompletions.retrieve_completion_result(
                task_id
            )
            raw_response = task_result
            task_status = raw_response.task_status
            get_count += 1
            await asyncio.sleep(1)
        tool_calls = raw_response.choices[0].message.tool_calls or []
        return ChatResponse(
            message=ChatMessage(
                content=raw_response.choices[0].message.content,
                role=raw_response.choices[0].message.role,
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=raw_response,
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        messages_dict = self._convert_to_llm_messages(messages)

        def gen() -> ChatResponseGen:
            raw_response = self._client.chat.completions.create(
                model=self.model,
                messages=messages_dict,
                stream=True,
                tools=kwargs.get("tools", None),
                tool_choice=kwargs.get("tool_choice", None),
                stop=kwargs.get("stop", None),
                timeout=self.timeout,
                extra_body=self.model_kwargs,
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
                        role=chunk.choices[0].delta.role,
                        additional_kwargs={"tool_calls": tool_calls},
                    ),
                    delta=chunk.choices[0].delta.content,
                    raw=chunk,
                )

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        messages_dict = self._convert_to_llm_messages(messages)

        async def gen() -> ChatResponseAsyncGen:
            # TODO async interfaces don't support streaming
            # needs to find a more suitable implementation method
            raw_response = self._client.chat.completions.create(
                model=self.model,
                messages=messages_dict,
                stream=True,
                tools=kwargs.get("tools", None),
                tool_choice=kwargs.get("tool_choice", None),
                stop=kwargs.get("stop", None),
                timeout=self.timeout,
                extra_body=self.model_kwargs,
            )
            response_txt = ""
            while True:
                chunk = await asyncio.to_thread(async_llm_generate, raw_response)
                if not chunk:
                    break
                if chunk.choices[0].delta.content is None:
                    continue
                response_txt += chunk.choices[0].delta.content
                tool_calls = chunk.choices[0].delta.tool_calls
                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=chunk.choices[0].delta.role,
                        additional_kwargs={"tool_calls": tool_calls},
                    ),
                    delta=chunk.choices[0].delta.content,
                    raw=chunk,
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
