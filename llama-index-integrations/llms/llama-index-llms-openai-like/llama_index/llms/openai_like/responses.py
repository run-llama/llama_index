from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    Callable,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
    ContentBlock,
    TextBlock,
    ImageBlock,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.llm import Model
from llama_index.core.prompts import PromptTemplate
from llama_index.core.program.utils import FlexibleModel  
from llama_index.core.types import PydanticProgramMode
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.llms.openai.responses import (
    OpenAIResponses,
    ResponseFunctionToolCall,
    DEFAULT_OPENAI_MODEL,
)

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

# Import OpenAI responses types
try:
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
        ResponseFunctionToolCall as OpenAIResponseFunctionToolCall,
        ResponseFunctionWebSearch,
        ResponseComputerToolCall,
        ResponseReasoningItem,
        ResponseCodeInterpreterToolCall,
        ResponseImageGenCallPartialImageEvent,
    )
    from openai.types.responses.response_output_item import ImageGenerationCall, McpCall
except ImportError:
    # Fallback for older OpenAI client versions
    Response = Any
    ResponseStreamEvent = Any
    ResponseCompletedEvent = Any
    ResponseCreatedEvent = Any
    ResponseFileSearchCallCompletedEvent = Any
    ResponseFunctionCallArgumentsDeltaEvent = Any
    ResponseFunctionCallArgumentsDoneEvent = Any
    ResponseInProgressEvent = Any
    ResponseOutputItemAddedEvent = Any
    ResponseOutputTextAnnotationAddedEvent = Any
    ResponseTextDeltaEvent = Any
    ResponseWebSearchCallCompletedEvent = Any
    ResponseOutputItem = Any
    ResponseOutputMessage = Any
    ResponseFileSearchToolCall = Any
    OpenAIResponseFunctionToolCall = Any
    ResponseFunctionWebSearch = Any
    ResponseComputerToolCall = Any
    ResponseReasoningItem = Any
    ResponseCodeInterpreterToolCall = Any
    ResponseImageGenCallPartialImageEvent = Any
    ImageGenerationCall = Any
    McpCall = Any

from llama_index.llms.openai.utils import (
    to_openai_message_dicts,
    is_json_schema_supported,
    resolve_openai_credentials,
    resolve_tool_choice,
    openai_modelname_to_contextsize,
    is_function_calling_model,
    create_retry_decorator,
)
from llama_index.llms.openai.base import Tokenizer
import httpx
import tiktoken
from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI
import functools


class OpenAILikeResponses(FunctionCallingLLM):
    """
    OpenAI-like Responses LLM with structured output support.

    This class extends OpenAILike to support the OpenAI /responses API for
    OpenAI-compatible servers. It combines the flexibility of OpenAILike for
    different API endpoints with the responses-specific functionality, including
    full support for structured output via Pydantic models.

    Features:
    - Support for OpenAI /responses API 
    - Structured output with Pydantic models
    - Function calling support
    - Streaming capabilities
    - Full async support

    Args:
        model: name of the model to use.
        api_base: The base URL for the API.
        api_key: API key for authentication.
        temperature: a float from 0 to 1 controlling randomness in generation.
        max_output_tokens: the maximum number of tokens to generate.
        reasoning_options: Optional dictionary to configure reasoning for O1 models.
        include: Additional output data to include in the model response.
        instructions: Instructions for the model to follow.
        track_previous_responses: Whether to track previous responses.
        store: Whether to store previous responses in OpenAI's storage.
        built_in_tools: The built-in tools to use for the model to augment responses.
        truncation: Whether to auto-truncate the input if it exceeds the model's context window.
        user: An optional identifier to help track the user's requests for abuse.
        strict: Whether to enforce strict validation of the structured output.
        context_window: The context window to use for the api.
        is_chat_model: Whether the model uses the chat or completion endpoint.
        is_function_calling_model: Whether the model supports OpenAI function calling/tools.
        pydantic_program_mode: Mode for structured output (DEFAULT, OPENAI_JSON, LLM).
        additional_kwargs: Add additional parameters to OpenAI request body.
        max_retries: How many times to retry the API call if it fails.
        timeout: How long to wait, in seconds, for an API call before failing.
        default_headers: override the default headers for API requests.
        http_client: pass in your own httpx.Client instance.
        async_http_client: pass in your own httpx.AsyncClient instance.

    Examples:
        `pip install llama-index-llms-openai-like`

        Basic usage:
        ```python
        from llama_index.llms.openai_like import OpenAILikeResponses

        llm = OpenAILikeResponses(
            model="my-model",
            api_base="https://my-openai-compatible-api.com/v1",
            api_key="my-api-key",
            context_window=128000,
            is_chat_model=True,
            is_function_calling_model=True,
        )

        response = llm.complete("Hi, write a short story")
        print(response.text)
        ```

        Structured output with Pydantic models:
        ```python
        from pydantic import BaseModel, Field

        class PersonInfo(BaseModel):
            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age")

        structured_llm = llm.as_structured_llm(PersonInfo)
        response = structured_llm.complete("Tell me about Alice, age 25")
        person_data = response.raw  # PersonInfo object
        print(f"Name: {person_data.name}, Age: {person_data.age}")
        ```

    """

    # Core LLM fields
    model: str = Field(
        default=DEFAULT_OPENAI_MODEL, description="The model name to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=2.0,
    )

    # Response-specific fields
    max_output_tokens: Optional[int] = Field(
        default=None,
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
    # OpenAI-like specific fields
    api_key: Optional[str] = Field(default=None, description="The API key for the service.")
    api_base: Optional[str] = Field(default=None, description="The base URL for the API.")
    api_version: Optional[str] = Field(default=None, description="The API version.")
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The context window to use for the api.",
    )
    is_chat_model: bool = Field(
        default=True,
        description="Whether the model uses the chat or completion endpoint.",
    )
    is_function_calling_model: bool = Field(
        default=True,
        description="Whether the model supports OpenAI function calling/tools over the API.",
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
    default_headers: Optional[Dict[str, str]] = Field(
        default=None, description="The default headers for API requests."
    )
    tokenizer: Union[Tokenizer, str, None] = Field(
        default=None,
        description=(
            "An instance of a tokenizer object that has an encode method, or the name"
            " of a tokenizer model from Hugging Face. If left as None, then this"
            " disables inference of max_tokens."
        ),
    )
    strict: bool = Field(
        default=False,
        description="Whether to enforce strict validation of the structured output.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )
    pydantic_program_mode: PydanticProgramMode = Field(
        default=PydanticProgramMode.DEFAULT,
        description="Pydantic program mode for structured output.",
    )

    _previous_response_id: Optional[str] = PrivateAttr()
    _client: Optional[SyncOpenAI] = PrivateAttr()
    _aclient: Optional[AsyncOpenAI] = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()
    _async_http_client: Optional[httpx.AsyncClient] = PrivateAttr()

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
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        # OpenAI-like specific parameters
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        is_chat_model: bool = True,
        is_function_calling_model: bool = True,
        max_retries: int = 3,
        timeout: float = 60.0,
        default_headers: Optional[Dict[str, str]] = None,
        tokenizer: Union[Tokenizer, str, None] = None,
        strict: bool = False,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        openai_client: Optional[SyncOpenAI] = None,
        async_openai_client: Optional[AsyncOpenAI] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        # Resolve credentials if not provided
        if api_key is None or api_base is None:
            resolved_key, resolved_base, resolved_version = resolve_openai_credentials(
                api_key=api_key, api_base=api_base, api_version=api_version
            )
            api_key = api_key or resolved_key
            api_base = api_base or resolved_base
            api_version = api_version or resolved_version

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
            call_metadata=call_metadata,
            pydantic_program_mode=pydantic_program_mode,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            context_window=context_window,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model,
            max_retries=max_retries,
            timeout=timeout,
            default_headers=default_headers,
            tokenizer=tokenizer,
            strict=strict,
            additional_kwargs=additional_kwargs,
            **kwargs,
        )

        self._previous_response_id = previous_response_id
        self._http_client = http_client
        self._async_http_client = async_http_client

        # store is set to true if track_previous_responses is true
        if self.track_previous_responses:
            self.store = True

        # Initialize OpenAI clients
        self._client = openai_client
        self._aclient = async_openai_client

    @classmethod
    def class_name(cls) -> str:
        return "openai_like_responses_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_output_tokens or -1,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    @property
    def _tokenizer(self) -> Optional[Tokenizer]:
        """Get tokenizer for this model."""
        if isinstance(self.tokenizer, str):
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(self.tokenizer)
            except ImportError:
                return None
        return self.tokenizer

    def _get_model_name(self) -> str:
        """Get the model name."""
        model_name = self.model
        if "ft-" in model_name:  # legacy fine-tuning
            model_name = model_name.split(":")[0]
        elif model_name.startswith("ft:"):
            model_name = model_name.split(":")[1]
        return model_name

    def _get_credential_kwargs(self, is_async: bool = False) -> Dict[str, Any]:
        """Get credential kwargs for OpenAI client."""
        return {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "default_headers": self.default_headers,
            "http_client": self._async_http_client if is_async else self._http_client,
        }

    def _get_client(self) -> SyncOpenAI:
        """Get sync OpenAI client."""
        if self._client is None:
            self._client = SyncOpenAI(**self._get_credential_kwargs())
        return self._client

    def _get_aclient(self) -> AsyncOpenAI:
        """Get async OpenAI client."""
        if self._aclient is None:
            self._aclient = AsyncOpenAI(**self._get_credential_kwargs(is_async=True))
        return self._aclient

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Get model kwargs for responses API calls."""
        initial_tools = self.built_in_tools or []
        
        # Responses API specific parameters
        responses_supported_params = {
            "model", "include", "instructions", "max_output_tokens", "metadata",
            "previous_response_id", "store", "temperature", "tools", "top_p",
            "truncation", "user", "tool_choice", "parallel_tool_calls"
        }
        
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
            "top_p": getattr(self, "top_p", 1.0),
            "truncation": self.truncation,
            "user": self.user,
        }

        if hasattr(self, "reasoning_options") and self.reasoning_options is not None:
            model_kwargs["reasoning"] = self.reasoning_options

        # priority is class args > additional_kwargs > runtime args
        model_kwargs.update(self.additional_kwargs)

        kwargs = kwargs or {}
        # Only include supported parameters from kwargs
        filtered_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in responses_supported_params
        }
        model_kwargs.update(filtered_kwargs)

        return model_kwargs

    def _parse_response_output(self, output: List[ResponseOutputItem]) -> ChatResponse:
        """Parse response output items into a ChatResponse."""
        import base64

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
            elif hasattr(item, "type") and item.type == "image_generation":
                # Handle image generation calls
                if getattr(item, "status", None) != "failed":
                    additional_kwargs["built_in_tool_calls"].append(item)
                    if hasattr(item, "result") and item.result is not None:
                        image_bytes = base64.b64decode(item.result)
                        blocks.append(ImageBlock(image=image_bytes))
            elif hasattr(item, "type") and item.type in [
                "code_interpreter",
                "mcp_call",
                "file_search",
                "web_search",
                "computer_tool",
            ]:
                # Handle various built-in tool calls
                additional_kwargs["built_in_tool_calls"].append(item)
            elif hasattr(item, "type") and item.type == "function_call":
                # Handle function tool calls
                tool_calls.append(item)
            elif hasattr(item, "type") and item.type == "reasoning":
                # Handle reasoning information
                additional_kwargs["reasoning"] = item

        if tool_calls and message:
            message.additional_kwargs["tool_calls"] = tool_calls

        return ChatResponse(message=message, additional_kwargs=additional_kwargs)

    @llm_retry_decorator
    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        client = self._get_client()
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        response: Response = client.responses.create(
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

            for event in self._get_client().responses.create(
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
        async def acomplete_fn(p: str, **kw: Any) -> CompletionResponse:
            return await self._convert_chat_to_completion_async(self._achat, p, **kw)

        return await acomplete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def astream_complete_fn(p: str, **kw: Any) -> CompletionResponseAsyncGen:
            return await self._convert_stream_chat_to_completion_async(
                self._astream_chat, p, **kw
            )

        return await astream_complete_fn(prompt, **kwargs)

    async def _convert_chat_to_completion_async(self, chat_fn, prompt: str, **kwargs):
        """Convert chat function to completion function asynchronously."""
        from llama_index.core.base.llms.generic_utils import (
            achat_to_completion_decorator,
        )

        completion_fn = achat_to_completion_decorator(chat_fn)
        return await completion_fn(prompt, **kwargs)

    async def _convert_stream_chat_to_completion_async(
        self, stream_chat_fn, prompt: str, **kwargs
    ):
        """Convert stream chat function to stream completion function asynchronously."""
        from llama_index.core.base.llms.generic_utils import (
            astream_chat_to_completion_decorator,
        )

        stream_completion_fn = astream_chat_to_completion_decorator(stream_chat_fn)
        return await stream_completion_fn(prompt, **kwargs)

    @llm_retry_decorator
    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        aclient = self._get_aclient()
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        response: Response = await aclient.responses.create(
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

            response_stream = await self._get_aclient().responses.create(
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

    # Override chat/complete methods to use responses API
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
        from llama_index.core.base.llms.generic_utils import (
            chat_to_completion_decorator,
        )

        complete_fn = chat_to_completion_decorator(self._chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        from llama_index.core.base.llms.generic_utils import (
            stream_chat_to_completion_decorator,
        )

        stream_complete_fn = stream_chat_to_completion_decorator(self._stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    # ===== Structured Output Methods =====
    def _should_use_structure_outputs(self) -> bool:
        """Check if structured output should be used."""
        return (
            getattr(self, "pydantic_program_mode", PydanticProgramMode.DEFAULT) == PydanticProgramMode.DEFAULT
            and is_json_schema_supported(self.model)
        )

    def _prepare_schema(
        self, llm_kwargs: Optional[Dict[str, Any]], output_cls: Type[Model]
    ) -> Dict[str, Any]:
        """Prepare schema for structured output."""
        try:
            from openai.resources.beta.chat.completions import _type_to_response_format
            response_format = _type_to_response_format(output_cls)
        except ImportError:
            # Fallback for older OpenAI client versions or unsupported formats
            response_format = {"type": "json_object"}

        llm_kwargs = llm_kwargs or {}
        llm_kwargs["response_format"] = response_format
        if "tool_choice" in llm_kwargs:
            del llm_kwargs["tool_choice"]
        return llm_kwargs

    def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Structured predict using responses API."""
        llm_kwargs = llm_kwargs or {}

        if self._should_use_structure_outputs():
            messages = self._extend_messages(prompt.format_messages(**prompt_args))
            llm_kwargs = self._prepare_schema(llm_kwargs, output_cls)
            response = self.chat(messages, **llm_kwargs)
            return output_cls.model_validate_json(str(response.message.content))

        # Fallback to function calling for structured outputs
        llm_kwargs["tool_choice"] = (
            "required" if "tool_choice" not in llm_kwargs else llm_kwargs["tool_choice"]
        )
        return super().structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Async structured predict using responses API."""
        llm_kwargs = llm_kwargs or {}

        if self._should_use_structure_outputs():
            messages = self._extend_messages(prompt.format_messages(**prompt_args))
            llm_kwargs = self._prepare_schema(llm_kwargs, output_cls)
            response = await self.achat(messages, **llm_kwargs)
            return output_cls.model_validate_json(str(response.message.content))

        # Fallback to function calling for structured outputs
        llm_kwargs["tool_choice"] = (
            "required" if "tool_choice" not in llm_kwargs else llm_kwargs["tool_choice"]
        )
        return await super().astructured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    def stream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Generator[Union[Model, FlexibleModel], None, None]:
        """Stream structured predict using responses API."""
        llm_kwargs = llm_kwargs or {}

        return super().stream_structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    async def astream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[Union[Model, FlexibleModel], None]:
        """Async stream structured predict using responses API.""" 
        llm_kwargs = llm_kwargs or {}
        return await super().astream_structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    def _extend_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Extend messages with any additional context if needed."""
        return messages

    # ===== Tool Calling Methods =====
    def _prepare_chat_with_tools(
        self, tools, user_msg=None, chat_history=None, **kwargs
    ):
        """Prepare chat with tools - adapted from OpenAIResponses implementation."""
        from llama_index.core.base.llms.types import ChatMessage, MessageRole
        from llama_index.llms.openai.utils import resolve_tool_choice

        # openai responses api has a slightly different tool spec format
        tool_specs = [
            {
                "type": "function",
                **tool.metadata.to_openai_tool(skip_length_check=True)["function"],
            }
            for tool in tools
        ]

        strict = getattr(self, "strict", False)

        if strict:
            for tool_spec in tool_specs:
                tool_spec["strict"] = True
                tool_spec["parameters"]["additionalProperties"] = False

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        tool_required = kwargs.get("tool_required", False)
        tool_choice = kwargs.get("tool_choice")
        allow_parallel_tool_calls = kwargs.get("allow_parallel_tool_calls", True)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": resolve_tool_choice(tool_choice, tool_required)
            if tool_specs
            else None,
            "parallel_tool_calls": allow_parallel_tool_calls,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in ["tool_required", "tool_choice", "allow_parallel_tool_calls"]
            },
        }

    def get_tool_calls_from_response(
        self, response, error_on_no_tool_call=True, **kwargs
    ):
        """Extract tool calls from response - adapted from OpenAIResponses implementation."""
        from llama_index.core.llms.llm import ToolSelection
        from llama_index.core.llms.utils import parse_partial_json

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
