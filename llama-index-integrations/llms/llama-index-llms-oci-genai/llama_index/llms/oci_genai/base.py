import json
from typing import Any, Callable, Dict, Optional, Sequence, List, Union, TYPE_CHECKING

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
from llama_index.core.base.llms.generic_utils import (
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager

from llama_index.core.constants import (
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM, ToolSelection
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

from llama_index.llms.oci_genai.utils import (
    CHAT_MODELS,
    create_client,
    get_provider,
    get_serving_mode,
    get_completion_generator,
    get_chat_generator,
    get_context_size,
    _format_oci_tool_calls,
    force_single_tool_call,
    validate_tool_call,
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


class OCIGenAI(FunctionCallingLLM):
    """OCI large language models with function calling support."""

    model: str = Field(description="Id of the OCI Generative AI model to use.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_size: int = Field("The maximum number of tokens available for input.")

    service_endpoint: Optional[str] = Field(
        default=None,
        description="service endpoint url.",
    )

    compartment_id: Optional[str] = Field(
        default=None,
        description="OCID of compartment.",
    )

    auth_type: Optional[str] = Field(
        description="Authentication type, can be: API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL. If not specified, API_KEY will be used",
        default="API_KEY",
    )

    auth_profile: Optional[str] = Field(
        description="The name of the profile in ~/.oci/config. If not specified , DEFAULT will be used",
        default="DEFAULT",
    )

    auth_file_location: Optional[str] = Field(
        description="Path to the config file. If not specified, ~/.oci/config will be used",
        default="~/.oci/config",
    )

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the OCI Generative AI request.",
    )

    _client: Any = PrivateAttr()
    _provider: str = PrivateAttr()
    _serving_mode: str = PrivateAttr()
    _completion_generator: str = PrivateAttr()
    _chat_generator: str = PrivateAttr()

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = 512,
        context_size: Optional[int] = None,
        service_endpoint: Optional[str] = None,
        compartment_id: Optional[str] = None,
        auth_type: Optional[str] = "API_KEY",
        auth_profile: Optional[str] = "DEFAULT",
        auth_file_location: Optional[str] = "~/.oci/config",
        client: Optional[Any] = None,
        provider: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """
        Initializes the OCIGenAI class.

        Args:
            model (str): The Id of the model to be used for generating embeddings, e.g., "meta.llama-2-70b-chat".

            temperature (Optional[float]): The temperature to use for sampling. Default specified in lama_index.core.constants.DEFAULT_TEMPERATURE.

            max_tokens (Optional[int]): The maximum number of tokens to generate. Default is 512.

            context_size (Optional[int]): The maximum number of tokens available for input. If not specified, the default context size for the model will be used.

            service_endpoint (str): service endpoint url, e.g., "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

            compartment_id (str): OCID of the compartment.

            auth_type (Optional[str]): Authentication type, can be: API_KEY (default), SECURITY_TOKEN, INSTANCEAL, RESOURCE_PRINCIPAL. If not specified, API_KEY will be used

            auth_profile (Optional[str]): The name of the profile in ~/.oci/config. If not specified , DEFAULT will be used

            auth_file_location (Optional[str]): Path to the config file, If not specified, ~/.oci/config will be used.

            client (Optional[Any]): An optional OCI client object. If not provided, the client will be created using the
                                    provided service endpoint and authentifcation method.

            provider (Optional[str]): Provider name of the model. If not specified, the provider will be derived from the model name.

            additional_kwargs (Optional[Dict[str, Any]]): Additional kwargs for the the LLM.
        """
        context_size = get_context_size(model, context_size)

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_size=context_size,
            service_endpoint=service_endpoint,
            compartment_id=compartment_id,
            auth_type=auth_type,
            auth_profile=auth_profile,
            auth_file_location=auth_file_location,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._client = client or create_client(
            auth_type, auth_profile, auth_file_location, service_endpoint
        )

        self._provider = get_provider(model, provider)

        self._serving_mode = get_serving_mode(model)

        self._completion_generator = get_completion_generator()

        self._chat_generator = get_chat_generator()

    @classmethod
    def class_name(cls) -> str:
        return "OCIGenAI_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_size,
            num_output=self.max_tokens,
            is_chat_model=self.model in CHAT_MODELS,
            model_name=self.model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        oci_params = self._provider.messages_to_oci_params(messages)
        oci_params["is_stream"] = False
        tools = kwargs.pop("tools", None)
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_params = {**all_kwargs, **oci_params}
        if tools:
            chat_params["tools"] = [
                self._provider.convert_to_oci_tool(tool) for tool in tools
            ]

        request = self._chat_generator(
            compartment_id=self.compartment_id,
            serving_mode=self._serving_mode,
            chat_request=self._provider.oci_chat_request(**chat_params),
        )

        response = self._client.chat(request)

        generation_info = self._provider.chat_generation_info(response)

        llm_output = {
            "model_id": response.data.model_id,
            "model_version": response.data.model_version,
            "request_id": response.request_id,
            "content-length": response.headers["content-length"],
        }

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=self._provider.chat_response_to_text(response),
                additional_kwargs=generation_info,
            ),
            raw=response.__dict__,
        )

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        oci_params = self._provider.messages_to_oci_params(messages)
        oci_params["is_stream"] = True
        tools = kwargs.pop("tools", None)
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_params = {**all_kwargs, **oci_params}
        if tools:
            chat_params["tools"] = [
                self._provider.convert_to_oci_tool(tool) for tool in tools
            ]

        request = self._chat_generator(
            compartment_id=self.compartment_id,
            serving_mode=self._serving_mode,
            chat_request=self._provider.oci_chat_request(**chat_params),
        )

        response = self._client.chat(request)

        def gen() -> ChatResponseGen:
            content = ""
            tool_calls_accumulated = []

            for event in response.data.events():
                content_delta = self._provider.chat_stream_to_text(
                    json.loads(event.data)
                )
                content += content_delta

                try:
                    event_data = json.loads(event.data)

                    tool_calls_data = None
                    for key in ["toolCalls", "tool_calls", "functionCalls"]:
                        if key in event_data:
                            tool_calls_data = event_data[key]
                            break

                    if tool_calls_data:
                        new_tool_calls = _format_oci_tool_calls(tool_calls_data)
                        for tool_call in new_tool_calls:
                            existing = next(
                                (
                                    t
                                    for t in tool_calls_accumulated
                                    if t["name"] == tool_call["name"]
                                ),
                                None,
                            )
                            if existing:
                                existing.update(tool_call)
                            else:
                                tool_calls_accumulated.append(tool_call)

                    generation_info = self._provider.chat_stream_generation_info(
                        event_data
                    )
                    if tool_calls_accumulated:
                        generation_info["tool_calls"] = tool_calls_accumulated

                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=content,
                            additional_kwargs=generation_info,
                        ),
                        delta=content_delta,
                        raw=event.__dict__,
                    )

                except json.JSONDecodeError:
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT, content=content
                        ),
                        delta=content_delta,
                        raw=event.__dict__,
                    )

                except Exception as e:
                    print(f"Error processing stream chunk: {e}")
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT, content=content
                        ),
                        delta=content_delta,
                        raw=event.__dict__,
                    )

        return gen()

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError("Async chat is not implemented yet.")

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError("Async complete is not implemented yet.")

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError("Async stream chat is not implemented yet.")

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError("Async stream complete is not implemented yet.")

    # Function tooling integration methods
    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tool_specs = [self._provider.convert_to_oci_tool(tool) for tool in tools]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        oci_params = self._provider.messages_to_oci_params(messages)
        chat_params = self._get_all_kwargs(**kwargs)

        return {
            "messages": messages,
            "tools": tool_specs,
            **oci_params,
            **chat_params,
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
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            validate_tool_call(tool_call)
            argument_dict = (
                json.loads(tool_call["input"])
                if isinstance(tool_call["input"], str)
                else tool_call["input"]
            )

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call["toolUseId"],
                    tool_name=tool_call["name"],
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections
