import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

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
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    get_from_param_or_env,
    stream_chat_to_completion_decorator,
)
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.llms.mistralai.utils import (
    is_mistralai_function_calling_model,
    is_mistralai_code_model,
    mistralai_modelname_to_contextsize,
)

from mistralai import Mistral
from mistralai.models import ToolCall
from mistralai.models import (
    Messages,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

DEFAULT_MISTRALAI_MODEL = "mistral-large-latest"
DEFAULT_MISTRALAI_ENDPOINT = "https://api.mistral.ai"
DEFAULT_MISTRALAI_MAX_TOKENS = 512


def to_mistral_chatmessage(
    messages: Sequence[ChatMessage],
) -> List[Messages]:
    new_messages = []
    for m in messages:
        tool_calls = m.additional_kwargs.get("tool_calls")
        if m.role == MessageRole.USER:
            new_messages.append(UserMessage(content=m.content))
        elif m.role == MessageRole.ASSISTANT:
            new_messages.append(
                AssistantMessage(content=m.content, tool_calls=tool_calls)
            )
        elif m.role == MessageRole.SYSTEM:
            new_messages.append(SystemMessage(content=m.content))
        elif m.role == MessageRole.TOOL or m.role == MessageRole.FUNCTION:
            new_messages.append(
                ToolMessage(
                    content=m.content,
                    tool_call_id=m.additional_kwargs.get("tool_call_id"),
                    name=m.additional_kwargs.get("name"),
                )
            )
        else:
            raise ValueError(f"Unsupported message role {m.role}")

    return new_messages


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


class MistralAI(FunctionCallingLLM):
    """MistralAI LLM.

    Examples:
        `pip install llama-index-llms-mistralai`

        ```python
        from llama_index.llms.mistralai import MistralAI

        # To customize your API key, do this
        # otherwise it will lookup MISTRAL_API_KEY from your env variable
        # llm = MistralAI(api_key="<api_key>")

        # You can specify a custom endpoint by passing the `endpoint` variable or setting
        # MISTRAL_ENDPOINT in your environment
        # llm = MistralAI(endpoint="<endpoint>")

        llm = MistralAI()

        resp = llm.complete("Paul Graham is ")

        print(resp)
        ```
    """

    model: str = Field(
        default=DEFAULT_MISTRALAI_MODEL, description="The mistralai model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_MISTRALAI_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    timeout: float = Field(
        default=120, description="The timeout to use in seconds.", ge=0
    )
    max_retries: int = Field(
        default=5, description="The maximum number of API retries.", ge=0
    )
    random_seed: Optional[int] = Field(
        default=None, description="The random seed to use for sampling."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the MistralAI API."
    )

    _client: Mistral = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_MISTRALAI_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MISTRALAI_MAX_TOKENS,
        timeout: int = 120,
        max_retries: int = 5,
        safe_mode: bool = False,
        random_seed: Optional[int] = None,
        api_key: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_key = get_from_param_or_env("api_key", api_key, "MISTRAL_API_KEY", "")

        if not api_key:
            raise ValueError(
                "You must provide an API key to use mistralai. "
                "You can either pass it in as an argument or set it `MISTRAL_API_KEY`."
            )

        # Use the custom endpoint if provided, otherwise default to DEFAULT_MISTRALAI_ENDPOINT
        endpoint = get_from_param_or_env(
            "endpoint", endpoint, "MISTRAL_ENDPOINT", DEFAULT_MISTRALAI_ENDPOINT
        )

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            safe_mode=safe_mode,
            random_seed=random_seed,
            model=model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._client = Mistral(
            api_key=api_key,
            server_url=endpoint,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MistralAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=mistralai_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            random_seed=self.random_seed,
            is_function_calling_model=is_mistralai_function_calling_model(self.model),
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "random_seed": self.random_seed,
            "retries": self.max_retries,
            "timeout_ms": self.timeout * 1000,
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

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = self._client.chat.complete(messages=messages, **all_kwargs)

        tool_calls = response.choices[0].message.tool_calls

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.choices[0].message.content,
                additional_kwargs=(
                    {"tool_calls": tool_calls} if tool_calls is not None else {}
                ),
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.chat.stream(messages=messages, **all_kwargs)

        def gen() -> ChatResponseGen:
            content = ""
            for chunk in response:
                delta = chunk.data.choices[0].delta
                role = delta.role or MessageRole.ASSISTANT
                # NOTE: Unlike openAI, we are directly injecting the tool calls
                additional_kwargs = {}
                if delta.tool_calls:
                    additional_kwargs["tool_calls"] = delta.tool_calls

                content_delta = delta.content
                if content_delta is None:
                    pass
                    # continue
                else:
                    content += content_delta
                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content_delta,
                    raw=chunk,
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = await self._client.chat.complete_async(
            messages=messages, **all_kwargs
        )
        tool_calls = response.choices[0].message.tool_calls
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.choices[0].message.content,
                additional_kwargs=(
                    {"tool_calls": tool_calls} if tool_calls is not None else {}
                ),
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = await self._client.chat.stream_async(messages=messages, **all_kwargs)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            async for chunk in response:
                delta = chunk.data.choices[0].delta
                role = delta.role or MessageRole.ASSISTANT
                # NOTE: Unlike openAI, we are directly injecting the tool calls
                additional_kwargs = {}
                if delta.tool_calls:
                    additional_kwargs["tool_calls"] = delta.tool_calls

                content_delta = delta.content
                if content_delta is None:
                    pass
                    # continue
                else:
                    content += content_delta
                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content_delta,
                    raw=chunk,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the chat with tools."""
        # misralai uses the same openai tool format
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
            **kwargs,
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
            if not isinstance(tool_call, ToolCall):
                raise ValueError("Invalid tool_call object")

            argument_dict = json.loads(tool_call.function.arguments)

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    def fill_in_middle(
        self, prompt: str, suffix: str, stop: Optional[List[str]] = None
    ) -> CompletionResponse:
        if not is_mistralai_code_model(self.model):
            raise ValueError(
                "Please provide code model from MistralAI. Currently supported code model is 'codestral-latest'."
            )

        if stop:
            response = self._client.fim.complete(
                model=self.model, prompt=prompt, suffix=suffix, stop=stop
            )
        else:
            response = self._client.fim.complete(
                model=self.model, prompt=prompt, suffix=suffix
            )

        return CompletionResponse(
            text=response.choices[0].message.content, raw=dict(response)
        )
