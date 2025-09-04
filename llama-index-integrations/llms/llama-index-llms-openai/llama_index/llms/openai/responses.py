import functools
import httpx
import tiktoken
import base64
from openai import AsyncOpenAI, AzureOpenAI
from openai import OpenAI as SyncOpenAI
from openai.types.responses import (
    Response,
    ResponseStreamEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFileSearchCallCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseTextDeltaEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseComputerToolCall,
    ResponseReasoningItem,
    ResponseCodeInterpreterToolCall,
    ResponseImageGenCallPartialImageEvent,
)
from openai.types.responses.response_output_item import ImageGenerationCall, McpCall
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import llama_index.core.instrumentation as instrument
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
    ContentBlock,
    TextBlock,
    ImageBlock,
)
from llama_index.core.bridge.pydantic import (
    Field,
    PrivateAttr,
)
from llama_index.core.constants import (
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection, Model
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.prompts import PromptTemplate
from llama_index.core.program.utils import FlexibleModel
from llama_index.llms.openai.utils import (
    O1_MODELS,
    create_retry_decorator,
    is_function_calling_model,
    openai_modelname_to_contextsize,
    resolve_openai_credentials,
    resolve_tool_choice,
    to_openai_message_dicts,
)


dispatcher = instrument.get_dispatcher(__name__)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def llm_retry_decorator(f: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(f)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        max_retries = getattr(self, "max_retries", 0)
        if max_retries <= 0:
            return f(self, *args, **kwargs)

        retry = create_retry_decorator(
            max_retries=max_retries,
            random_exponential=True,
            stop_after_delay_seconds=60,
            min_seconds=1,
            max_seconds=20,
        )
        return retry(f)(self, *args, **kwargs)

    return wrapper


@runtime_checkable
class Tokenizer(Protocol):
    """Tokenizers support an encode function that returns a list of ints."""

    def encode(self, text: str) -> List[int]:  # fmt: skip
        ...


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


class OpenAIResponses(FunctionCallingLLM):
    """
    OpenAI Responses LLM.

    Args:
        model: name of the OpenAI model to use.
        temperature: a float from 0 to 1 controlling randomness in generation; higher will lead to more creative, less deterministic responses.
        max_output_tokens: the maximum number of tokens to generate.
        reasoning_options: Optional dictionary to configure reasoning for O1 models.
                    Corresponds to the 'reasoning' parameter in the OpenAI API.
                    Example: {"effort": "low", "summary": "concise"}
        include: Additional output data to include in the model response.
        instructions: Instructions for the model to follow.
        track_previous_responses: Whether to track previous responses. If true, the LLM class will statefully track previous responses.
        store: Whether to store previous responses in OpenAI's storage.
        built_in_tools: The built-in tools to use for the model to augment responses.
        truncation: Whether to auto-truncate the input if it exceeds the model's context window.
        user: An optional identifier to help track the user's requests for abuse.
        strict: Whether to enforce strict validation of the structured output.
        additional_kwargs: Add additional parameters to OpenAI request body.
        max_retries: How many times to retry the API call if it fails.
        timeout: How long to wait, in seconds, for an API call before failing.
        api_key: Your OpenAI api key
        api_base: The base URL of the API to call
        api_version: the version of the API to call
        default_headers: override the default headers for API requests.
        http_client: pass in your own httpx.Client instance.
        async_http_client: pass in your own httpx.AsyncClient instance.

    Examples:
        `pip install llama-index-llms-openai`

        ```python
        from llama_index.llms.openai import OpenAIResponses

        llm = OpenAIResponses(model="gpt-4o-mini", api_key="sk-...")

        response = llm.complete("Hi, write a short story")
        print(response.text)
        ```
    """

    model: str = Field(
        default=DEFAULT_OPENAI_MODEL, description="The OpenAI model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default=1.0,
        description="The top-p value to use during generation.",
        ge=0.0,
        le=1.0,
    )
    max_output_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    reasoning_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary to configure reasoning for O1 models. Example: {'effort': 'low', 'summary': 'concise'}",
    )
    include: Optional[List[str]] = Field(
        default=None,
        description="Additional output data to include in the model response.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Instructions for the model to follow.",
    )
    track_previous_responses: bool = Field(
        default=False,
        description="Whether to track previous responses. If true, the LLM class will statefully track previous responses.",
    )
    store: bool = Field(
        default=False,
        description="Whether to store previous responses in OpenAI's storage.",
    )
    built_in_tools: Optional[List[dict]] = Field(
        default=None,
        description="The built-in tools to use for the model to augment responses.",
    )
    truncation: str = Field(
        default="disabled",
        description="Whether to auto-truncate the input if it exceeds the model's context window.",
    )
    user: Optional[str] = Field(
        default=None,
        description="An optional identifier to help track the user's requests for abuse.",
    )
    call_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata to include in the API call.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the OpenAI API at inference time.",
    )
    max_retries: int = Field(
        default=3,
        description="The maximum number of API retries.",
        ge=0,
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout, in seconds, for API requests.",
        ge=0,
    )
    strict: bool = Field(
        default=False,
        description="Whether to enforce strict validation of the structured output.",
    )
    default_headers: Optional[Dict[str, str]] = Field(
        default=None, description="The default headers for API requests."
    )
    api_key: str = Field(default=None, description="The OpenAI API key.")
    api_base: str = Field(description="The base URL for OpenAI API.")
    api_version: str = Field(description="The API version for OpenAI API.")
    context_window: Optional[int] = Field(
        default=None,
        description="The context window override for the model.",
    )

    _client: SyncOpenAI = PrivateAttr()
    _aclient: AsyncOpenAI = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()
    _async_http_client: Optional[httpx.AsyncClient] = PrivateAttr()
    _previous_response_id: Optional[str] = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: Optional[int] = None,
        reasoning_options: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        track_previous_responses: bool = False,
        store: bool = False,
        built_in_tools: Optional[List[dict]] = None,
        truncation: str = "disabled",
        user: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        call_metadata: Optional[Dict[str, Any]] = None,
        strict: bool = False,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        openai_client: Optional[SyncOpenAI] = None,
        async_openai_client: Optional[AsyncOpenAI] = None,
        context_window: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        api_key, api_base, api_version = resolve_openai_credentials(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )

        # TODO: Temp forced to 1.0 for o1
        if model in O1_MODELS:
            temperature = 1.0

        super().__init__(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_options=reasoning_options,
            include=include,
            instructions=instructions,
            track_previous_responses=track_previous_responses,
            store=store,
            built_in_tools=built_in_tools,
            truncation=truncation,
            user=user,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            api_key=api_key,
            api_version=api_version,
            api_base=api_base,
            timeout=timeout,
            default_headers=default_headers,
            call_metadata=call_metadata,
            strict=strict,
            context_window=context_window,
            **kwargs,
        )

        self._previous_response_id = previous_response_id

        # store is set to true if track_previous_responses is true
        if self.track_previous_responses:
            self.store = True

        self._http_client = http_client
        self._async_http_client = async_http_client
        self._client = openai_client or SyncOpenAI(**self._get_credential_kwargs())
        self._aclient = async_openai_client or AsyncOpenAI(
            **self._get_credential_kwargs(is_async=True)
        )

    @classmethod
    def class_name(cls) -> str:
        return "openai_responses_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window
            or openai_modelname_to_contextsize(self._get_model_name()),
            num_output=self.max_output_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=is_function_calling_model(
                model=self._get_model_name()
            ),
            model_name=self.model,
        )

    @property
    def _tokenizer(self) -> Optional[Tokenizer]:
        """
        Get a tokenizer for this model, or None if a tokenizing method is unknown.

        OpenAI can do this using the tiktoken package, subclasses may not have
        this convenience.
        """
        return tiktoken.encoding_for_model(self._get_model_name())

    def _get_model_name(self) -> str:
        model_name = self.model
        if "ft-" in model_name:  # legacy fine-tuning
            model_name = model_name.split(":")[0]
        elif model_name.startswith("ft:"):
            model_name = model_name.split(":")[1]
        return model_name

    def _is_azure_client(self) -> bool:
        return isinstance(self._get_client(), AzureOpenAI)

    def _get_credential_kwargs(self, is_async: bool = False) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "default_headers": self.default_headers,
            "http_client": self._async_http_client if is_async else self._http_client,
        }

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        initial_tools = self.built_in_tools or []
        model_kwargs = {
            "model": self.model,
            "include": self.include,
            "instructions": self.instructions,
            "max_output_tokens": self.max_output_tokens,
            "metadata": self.call_metadata,
            "previous_response_id": self._previous_response_id,
            "store": self.store,
            "temperature": self.temperature,
            "tools": [*initial_tools, *kwargs.pop("tools", [])],
            "top_p": self.top_p,
            "truncation": self.truncation,
            "user": self.user,
        }

        if self.model in O1_MODELS and self.reasoning_options is not None:
            model_kwargs["reasoning"] = self.reasoning_options

        # priority is class args > additional_kwargs > runtime args
        model_kwargs.update(self.additional_kwargs)

        kwargs = kwargs or {}
        model_kwargs.update(kwargs)

        return model_kwargs

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._chat(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        return self._stream_chat(messages, **kwargs)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self._chat)

        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self._stream_chat)

        return stream_complete_fn(prompt, **kwargs)

    def _parse_response_output(self, output: List[ResponseOutputItem]) -> ChatResponse:
        message = ChatMessage(role=MessageRole.ASSISTANT, blocks=[])
        additional_kwargs = {"built_in_tool_calls": []}
        tool_calls = []
        blocks: List[ContentBlock] = []
        for item in output:
            if isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if hasattr(part, "text"):
                        blocks.append(TextBlock(text=part.text))
                    if hasattr(part, "annotations"):
                        additional_kwargs["annotations"] = part.annotations
                    if hasattr(part, "refusal"):
                        additional_kwargs["refusal"] = part.refusal

                message.blocks.extend(blocks)
            elif isinstance(item, ImageGenerationCall):
                # return an ImageBlock if there is image generation
                if item.status != "failed":
                    additional_kwargs["built_in_tool_calls"].append(item)
                    if item.result is not None:
                        image_bytes = base64.b64decode(item.result)
                        blocks.append(ImageBlock(image=image_bytes))
            elif isinstance(item, ResponseCodeInterpreterToolCall):
                additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, McpCall):
                additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, ResponseFileSearchToolCall):
                additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, ResponseFunctionToolCall):
                tool_calls.append(item)
            elif isinstance(item, ResponseFunctionWebSearch):
                additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, ResponseComputerToolCall):
                additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, ResponseReasoningItem):
                additional_kwargs["reasoning"] = item

        if tool_calls and message:
            message.additional_kwargs["tool_calls"] = tool_calls

        return ChatResponse(message=message, additional_kwargs=additional_kwargs)

    @llm_retry_decorator
    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        kwargs_dict = self._get_model_kwargs(**kwargs)
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        response: Response = self._client.responses.create(
            input=message_dicts,
            stream=False,
            **kwargs_dict,
        )

        if self.track_previous_responses:
            self._previous_response_id = response.id

        chat_response = self._parse_response_output(response.output)
        chat_response.raw = response
        chat_response.additional_kwargs["usage"] = response.usage

        return chat_response

    @staticmethod
    def process_response_event(
        event: ResponseStreamEvent,
        tool_calls: List[ResponseFunctionToolCall],
        built_in_tool_calls: List[Any],
        additional_kwargs: Dict[str, Any],
        current_tool_call: Optional[ResponseFunctionToolCall],
        track_previous_responses: bool,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[
        List[ContentBlock],
        List[ResponseFunctionToolCall],
        List[Any],
        Dict[str, Any],
        Optional[ResponseFunctionToolCall],
        Optional[str],
        str,
    ]:
        """
        Process a ResponseStreamEvent and update the state accordingly.

        Args:
            event: The response stream event to process
            content: Current accumulated content string
            tool_calls: List of completed tool calls
            built_in_tool_calls: List of built-in tool calls
            additional_kwargs: Additional keyword arguments to include in ChatResponse
            current_tool_call: The currently in-progress tool call, if any
            track_previous_responses: Whether to track previous response IDs
            previous_response_id: Previous response ID if tracking

        Returns:
            A tuple containing the updated state:
            (content, tool_calls, built_in_tool_calls, additional_kwargs, current_tool_call, updated_previous_response_id, delta)
        """
        delta = ""
        updated_previous_response_id = previous_response_id
        # we use blocks instead of content, since now we also support images! :)
        blocks: List[ContentBlock] = []
        if isinstance(event, ResponseCreatedEvent) or isinstance(
            event, ResponseInProgressEvent
        ):
            # Initial events, track the response id
            if track_previous_responses:
                updated_previous_response_id = event.response.id
        elif isinstance(event, ResponseOutputItemAddedEvent):
            # New output item (message, tool call, etc.)
            if isinstance(event.item, ResponseFunctionToolCall):
                current_tool_call = event.item
        elif isinstance(event, ResponseTextDeltaEvent):
            # Text content is being added
            delta = event.delta
            blocks.append(TextBlock(text=delta))
        elif isinstance(event, ResponseImageGenCallPartialImageEvent):
            # Partial image
            if event.partial_image_b64:
                blocks.append(
                    ImageBlock(
                        image=base64.b64decode(event.partial_image_b64),
                        detail=f"id_{event.partial_image_index}",
                    )
                )
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            # Function call arguments are being streamed
            if current_tool_call is not None:
                current_tool_call.arguments += event.delta
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            # Function call arguments are complete
            if current_tool_call is not None:
                current_tool_call.arguments = event.arguments
                current_tool_call.status = "completed"

                # append a copy of the tool call to the list
                tool_calls.append(
                    ResponseFunctionToolCall(**current_tool_call.model_dump())
                )

                # clear the current tool call
                current_tool_call = None
        elif isinstance(event, ResponseOutputTextAnnotationAddedEvent):
            # Annotations for the text
            annotations = additional_kwargs.get("annotations", [])
            annotations.append(event.annotation)
            additional_kwargs["annotations"] = annotations
        elif isinstance(event, ResponseFileSearchCallCompletedEvent):
            # File search tool call completed
            built_in_tool_calls.append(event)
        elif isinstance(event, ResponseWebSearchCallCompletedEvent):
            # Web search tool call completed
            built_in_tool_calls.append(event)
        elif isinstance(event, ResponseReasoningItem):
            # Reasoning information
            additional_kwargs["reasoning"] = event
        elif isinstance(event, ResponseCompletedEvent):
            # Response is complete
            if hasattr(event, "response") and hasattr(event.response, "usage"):
                additional_kwargs["usage"] = event.response.usage

        return (
            blocks,
            tool_calls,
            built_in_tool_calls,
            additional_kwargs,
            current_tool_call,
            updated_previous_response_id,
            delta,
        )

    @llm_retry_decorator
    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        def gen() -> ChatResponseGen:
            tool_calls = []
            built_in_tool_calls = []
            additional_kwargs = {"built_in_tool_calls": []}
            current_tool_call: Optional[ResponseFunctionToolCall] = None
            local_previous_response_id = self._previous_response_id

            for event in self._client.responses.create(
                input=message_dicts,
                stream=True,
                **self._get_model_kwargs(**kwargs),
            ):
                # Process the event and update state
                (
                    blocks,
                    tool_calls,
                    built_in_tool_calls,
                    additional_kwargs,
                    current_tool_call,
                    local_previous_response_id,
                    delta,
                ) = OpenAIResponses.process_response_event(
                    event=event,
                    tool_calls=tool_calls,
                    built_in_tool_calls=built_in_tool_calls,
                    additional_kwargs=additional_kwargs,
                    current_tool_call=current_tool_call,
                    track_previous_responses=self.track_previous_responses,
                    previous_response_id=local_previous_response_id,
                )

                if (
                    self.track_previous_responses
                    and local_previous_response_id != self._previous_response_id
                ):
                    self._previous_response_id = local_previous_response_id

                if built_in_tool_calls:
                    additional_kwargs["built_in_tool_calls"] = built_in_tool_calls

                # For any event, yield a ChatResponse with the current state
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        blocks=blocks,
                        additional_kwargs={"tool_calls": tool_calls}
                        if tool_calls
                        else {},
                    ),
                    delta=delta,
                    raw=event,
                    additional_kwargs=additional_kwargs,
                )

        return gen()

    # ===== Async Endpoints =====
    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return await self._achat(messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        return await self._astream_chat(messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self._achat)

        return await acomplete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self._astream_chat)

        return await astream_complete_fn(prompt, **kwargs)

    @llm_retry_decorator
    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        response: Response = await self._aclient.responses.create(
            input=message_dicts,
            stream=False,
            **self._get_model_kwargs(**kwargs),
        )

        if self.track_previous_responses:
            self._previous_response_id = response.id

        chat_response = self._parse_response_output(response.output)
        chat_response.raw = response
        chat_response.additional_kwargs["usage"] = response.usage

        return chat_response

    @llm_retry_decorator
    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        async def gen() -> ChatResponseAsyncGen:
            tool_calls = []
            built_in_tool_calls = []
            additional_kwargs = {"built_in_tool_calls": []}
            current_tool_call: Optional[ResponseFunctionToolCall] = None
            local_previous_response_id = self._previous_response_id

            response_stream = await self._aclient.responses.create(
                input=message_dicts,
                stream=True,
                **self._get_model_kwargs(**kwargs),
            )

            async for event in response_stream:
                # Process the event and update state
                (
                    blocks,
                    tool_calls,
                    built_in_tool_calls,
                    additional_kwargs,
                    current_tool_call,
                    local_previous_response_id,
                    delta,
                ) = OpenAIResponses.process_response_event(
                    event=event,
                    tool_calls=tool_calls,
                    built_in_tool_calls=built_in_tool_calls,
                    additional_kwargs=additional_kwargs,
                    current_tool_call=current_tool_call,
                    track_previous_responses=self.track_previous_responses,
                    previous_response_id=local_previous_response_id,
                )

                if (
                    self.track_previous_responses
                    and local_previous_response_id != self._previous_response_id
                ):
                    self._previous_response_id = local_previous_response_id

                if built_in_tool_calls:
                    additional_kwargs["built_in_tool_calls"] = built_in_tool_calls

                # For any event, yield a ChatResponse with the current state
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        blocks=blocks,
                        additional_kwargs={"tool_calls": tool_calls}
                        if tool_calls
                        else {},
                    ),
                    delta=delta,
                    raw=event,
                    additional_kwargs=additional_kwargs,
                )

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        allow_parallel_tool_calls: bool = True,
        tool_required: bool = False,
        tool_choice: Optional[Union[str, dict]] = None,
        verbose: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Predict and call the tool."""

        # openai responses api has a slightly different tool spec format
        tool_specs = [
            {
                "type": "function",
                **tool.metadata.to_openai_tool(skip_length_check=True)["function"],
            }
            for tool in tools
        ]

        if strict is not None:
            strict = strict
        else:
            strict = self.strict

        if strict:
            for tool_spec in tool_specs:
                tool_spec["strict"] = True
                tool_spec["parameters"]["additionalProperties"] = False

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": resolve_tool_choice(tool_choice, tool_required)
            if tool_specs
            else None,
            "parallel_tool_calls": allow_parallel_tool_calls,
            **kwargs,
        }

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls: List[ResponseFunctionToolCall] = (
            response.message.additional_kwargs.get("tool_calls", [])
        )

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            # this should handle both complete and partial jsons
            try:
                argument_dict = parse_partial_json(tool_call.arguments)
            except ValueError:
                argument_dict = {}

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.call_id,
                    tool_name=tool_call.name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    @dispatcher.span
    def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Structured predict."""
        llm_kwargs = llm_kwargs or {}

        llm_kwargs["tool_choice"] = (
            "required" if "tool_choice" not in llm_kwargs else llm_kwargs["tool_choice"]
        )
        # by default structured prediction uses function calling to extract structured outputs
        # here we force tool_choice to be required
        return super().structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    @dispatcher.span
    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Structured predict."""
        llm_kwargs = llm_kwargs or {}

        llm_kwargs["tool_choice"] = (
            "required" if "tool_choice" not in llm_kwargs else llm_kwargs["tool_choice"]
        )
        # by default structured prediction uses function calling to extract structured outputs
        # here we force tool_choice to be required
        return await super().astructured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    @dispatcher.span
    def stream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Generator[Union[Model, FlexibleModel], None, None]:
        """Stream structured predict."""
        llm_kwargs = llm_kwargs or {}

        llm_kwargs["tool_choice"] = (
            "required" if "tool_choice" not in llm_kwargs else llm_kwargs["tool_choice"]
        )
        # by default structured prediction uses function calling to extract structured outputs
        # here we force tool_choice to be required
        return super().stream_structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    @dispatcher.span
    async def astream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[Union[Model, FlexibleModel], None]:
        """Stream structured predict."""
        llm_kwargs = llm_kwargs or {}

        llm_kwargs["tool_choice"] = (
            "required" if "tool_choice" not in llm_kwargs else llm_kwargs["tool_choice"]
        )
        # by default structured prediction uses function calling to extract structured outputs
        # here we force tool_choice to be required
        return await super().astream_structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )
