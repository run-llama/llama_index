"""Azure AI model inference chat completions client."""

import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    TYPE_CHECKING,
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
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr, BaseModel
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

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.aio import ChatCompletionsClient as ChatCompletionsClientAsync

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool
    from llama_index.core.chat_engine.types import AgentChatResponse
    from azure.core.credentials import TokenCredential

from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import (
    ChatCompletionsToolCall,
    ChatRequestMessage,
    ChatResponseMessage,
)


def to_inference_message(
    messages: Sequence[ChatMessage],
) -> List[ChatRequestMessage]:
    """Converts a sequence of `ChatMessage` to a list of `ChatRequestMessage`
    which can be used for Azure AI model inference.

    Args:
        messages (Sequence[ChatMessage]): The messages to convert.

    Returns:
        List[ChatRequestMessage]: The converted messages.
    """
    new_messages = []
    for m in messages:
        message_dict = {
            "role": m.role.value,
            "content": m.content,
        }

        message_dict.update(
            {k: v for k, v in m.additional_kwargs.items() if v is not None}
        )
        new_messages.append(ChatRequestMessage(message_dict))
    return new_messages


def to_inference_tool(metadata: Type[BaseModel]) -> Dict[str, Any]:
    """Converts a tool metadata to a tool dict for Azure AI model inference.

    Args:
        tool_metadata (Type[ToolMedata]): The metadata of the tool to convert.

    Returns:
        Dict[str, Any]: The converted tool dict.
    """
    return {
        "type": "function",
        "function": {
            "name": metadata.name,
            "description": metadata.description,
            "parameters": metadata.get_parameters_dict(),
        },
    }


def from_inference_message(message: ChatResponseMessage) -> ChatMessage:
    """Convert an inference message dict to generic message."""
    role = message.role
    content = message.as_dict().get("content", "")

    # function_call = None  # deprecated in OpenAI v 1.1.0

    additional_kwargs: Dict[str, Any] = {}
    if message.tool_calls is not None:
        tool_calls: List[ChatCompletionsToolCall] = message.tool_calls
        additional_kwargs.update(tool_calls=tool_calls)

    return ChatMessage(role=role, content=content, additional_kwargs=additional_kwargs)


def force_single_tool_call(response: ChatResponse) -> None:
    """Forces the response to have only one tool call.

    Args:
        response (ChatResponse): The response to modify.
    """
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


class AzureAICompletionsModel(FunctionCallingLLM):
    """Azure AI model inference for LLM.

    Examples:
        ```python
        from llama_index.core import Settings
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.azure_inference import AzureAICompletionsModel

        llm = AzureAICompletionsModel(
            endpoint="https://[your-endpoint].inference.ai.azure.com",
            credential="your-api-key",
            temperature=0
        )

        # If using Microsoft Entra ID authentication, you can create the
        # client as follows:
        #
        # from azure.identity import DefaultAzureCredential
        #
        # llm = AzureAICompletionsModel(
        #     endpoint="https://[your-endpoint].inference.ai.azure.com",
        #     credential=DefaultAzureCredential()
        # )
        #
        # # If you plan to use asynchronous calling, make sure to use the async
        # # credentials as follows:
        #
        # from azure.identity.aio import DefaultAzureCredential as DefaultAzureCredentialAsync
        #
        # llm = AzureAICompletionsModel(
        #     endpoint="https://[your-endpoint].inference.ai.azure.com",
        #     credential=DefaultAzureCredentialAsync()
        # )

        resp = llm.chat(
            messages=ChatMessage(role="user", content="Who is Paul Graham?")
        )

        print(resp)

        # Once the client is instantiated, you can set the context to use the model
        Settings.llm = llm
        ```
    """

    model_name: Optional[str] = Field(
        default=None,
        description="The model id to use. Optional for endpoints running a single model.",
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    seed: str = Field(default=None, description="The random seed to use for sampling.")
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs model parameters."
    )

    _client: ChatCompletionsClient = PrivateAttr()
    _async_client: ChatCompletionsClientAsync = PrivateAttr()
    _model_name: str = PrivateAttr(None)
    _model_type: str = PrivateAttr(None)
    _model_provider: str = PrivateAttr(None)

    def __init__(
        self,
        endpoint: str = None,
        credential: Union[str, AzureKeyCredential, "TokenCredential"] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        client_kwargs: Dict[str, Any] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        client_kwargs = client_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        endpoint = get_from_param_or_env(
            "endpoint", endpoint, "AZURE_INFERENCE_ENDPOINT", None
        )
        credential = get_from_param_or_env(
            "credential", credential, "AZURE_INFERENCE_CREDENTIAL", None
        )
        credential = (
            AzureKeyCredential(credential)
            if isinstance(credential, str)
            else credential
        )

        if not endpoint:
            raise ValueError(
                "You must provide an endpoint to use the Azure AI model inference LLM."
                "Pass the endpoint as a parameter or set the AZURE_INFERENCE_ENDPOINT"
                "environment variable."
            )

        if not credential:
            raise ValueError(
                "You must provide an credential to use the Azure AI model inference LLM."
                "Pass the credential as a parameter or set the AZURE_INFERENCE_CREDENTIAL"
            )

        self._client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential,
            user_agent="llamaindex",
            **client_kwargs,
        )

        self._async_client = ChatCompletionsClientAsync(
            endpoint=endpoint,
            credential=credential,
            user_agent="llamaindex",
            **client_kwargs,
        )

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "AzureAICompletionsModel"

    @property
    def metadata(self) -> LLMMetadata:
        if not self._model_name:
            model_info = self._client.get_model_info()
            if model_info:
                self._model_name = model_info.get("model_name", None)
                self._model_type = model_info.get("model_type", None)
                self._model_provider = model_info.get("model_provider_name", None)

        return LLMMetadata(
            is_chat_model=self._model_type == "chat-completions",
            model_name=self._model_name,
            model_type=self._model_type,
            model_provider=self._model_provider,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model_name:
            base_kwargs["model"] = self.model_name

        return {
            **base_kwargs,
            **self.model_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messages = to_inference_message(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = self._client.complete(messages=messages, **all_kwargs)

        response_message = from_inference_message(response.choices[0].message)

        return ChatResponse(
            message=response_message,
            raw=response.as_dict(),
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
        messages = to_inference_message(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.complete(messages=messages, stream=True, **all_kwargs)

        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for chunk in response:
                content_delta = (
                    chunk.choices[0].delta.content if len(chunk.choices) > 0 else None
                )
                if content_delta is None:
                    continue
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
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
        messages = to_inference_message(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = await self._async_client.complete(messages=messages, **all_kwargs)

        response_message = from_inference_message(response.choices[0].message)

        return ChatResponse(
            message=response_message,
            raw=response.as_dict(),
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
        messages = to_inference_message(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._async_client.complete(
            messages=messages, stream=True, **all_kwargs
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            role = MessageRole.ASSISTANT
            async for chunk in response:
                content_delta = (
                    chunk.choices[0].delta.content if chunk.choices else None
                )
                if content_delta is None:
                    continue
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=chunk,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, stream=True, **kwargs)

    def chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Predict and call the tool."""
        # Azure AI model inference uses the same openai tool format
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        response = self.chat(
            messages,
            tools=tool_specs,
            **kwargs,
        )
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    async def achat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Predict and call the tool."""
        # Azure AI model inference uses the same openai tool format
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        response = await self.achat(
            messages,
            tools=tool_specs,
            **kwargs,
        )
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: "AgentChatResponse",
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
            if not isinstance(tool_call, ChatCompletionsToolCall):
                raise ValueError("Invalid tool_call object")
            if tool_call.type != "function":
                raise ValueError(
                    "Invalid tool type. Only `function` is supported but `{tool_call.type}` was received."
                )
            argument_dict = json.loads(tool_call.function.arguments)

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the arguments needed to let the LLM chat with tools."""
        chat_history = chat_history or []

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
            chat_history.append(user_msg)

        tool_dicts = [to_inference_tool(tool.metadata) for tool in tools]

        return {
            "messages": chat_history,
            "tools": tool_dicts or None,
            **kwargs,
        }
