import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

import llama_index.core.instrumentation as instrument
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
from llama_index.core.bridge.pydantic import (
    Field,
    PrivateAttr,
    model_validator,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.oci_data_science.client import AsyncClient, Client
from llama_index.llms.oci_data_science.utils import (
    DEFAULT_TOOL_CHOICE,
    _from_completion_logprobs_dict,
    _from_message_dict,
    _from_token_logprob_dicts,
    _get_response_token_counts,
    _resolve_tool_choice,
    _to_message_dicts,
    _update_tool_calls,
)


if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

dispatcher = instrument.get_dispatcher(__name__)


DEFAULT_MODEL = "odsc-llm"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 5

logger = logging.getLogger(__name__)


class OCIDataScience(FunctionCallingLLM):
    """
    LLM deployed on OCI Data Science Model Deployment.

    **Setup:**
        Install ``oracle-ads`` and ``llama-index-llms-oci-data-science``.

        ```bash
        pip install -U oracle-ads llama-index-llms-oci-data-science
        ```

        Use `ads.set_auth()` to configure authentication.
        For example, to use OCI resource_principal for authentication:

        ```python
        import ads
        ads.set_auth("resource_principal")
        ```

        For more details on authentication, see:
        https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

        Make sure to have the required policies to access the OCI Data
        Science Model Deployment endpoint. See:
        https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm

        To learn more about deploying LLM models in OCI Data Science, see:
        https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions-model-deploy.htm


    **Examples:**

        **Basic Usage:**

        ```python
        from llama_index.llms.oci_data_science import OCIDataScience
        import ads
        ads.set_auth(auth="security_token", profile="OC1")

        llm = OCIDataScience(
            endpoint="https://<MD_OCID>/predict",
            model="odsc-llm",
        )
        prompt = "What is the capital of France?"
        response = llm.complete(prompt)
        print(response)
        ```

        **Custom Parameters:**

        ```python
        llm = OCIDataScience(
            endpoint="https://<MD_OCID>/predict",
            model="odsc-llm",
            temperature=0.7,
            max_tokens=150,
            additional_kwargs={"top_p": 0.9},
        )
        ```

        **Using Chat Interface:**

        ```python
        messages = [
            ChatMessage(role="user", content="Tell me a joke."),
            ChatMessage(role="assistant", content="Why did the chicken cross the road?"),
            ChatMessage(role="user", content="I don't know, why?"),
        ]

        chat_response = llm.chat(messages)
        print(chat_response)
        ```

        **Streaming Completion:**

        ```python
        for chunk in llm.stream_complete("Once upon a time"):
            print(chunk.delta, end="")
        ```

        **Asynchronous Chat:**

        ```python
        import asyncio

        async def async_chat():
            messages = [
                ChatMessage(role="user", content="What's the weather like today?")
            ]
            response = await llm.achat(messages)
            print(response)

        asyncio.run(async_chat())
        ```

        **Using Tools (Function Calling):**

        ```python
        from llama_index.llms.oci_data_science import OCIDataScience
        from llama_index.core.tools import FunctionTool
        import ads
        ads.set_auth(auth="security_token", profile="OC1")

        def multiply(a: float, b: float) -> float:
            return a * b

        def add(a: float, b: float) -> float:
            return a + b

        def subtract(a: float, b: float) -> float:
            return a - b

        def divide(a: float, b: float) -> float:
            return a / b


        multiply_tool = FunctionTool.from_defaults(fn=multiply)
        add_tool = FunctionTool.from_defaults(fn=add)
        sub_tool = FunctionTool.from_defaults(fn=subtract)
        divide_tool = FunctionTool.from_defaults(fn=divide)

        llm = OCIDataScience(
            endpoint="https://<MD_OCID>/predict",
            model="odsc-llm",
            temperature=0.7,
            max_tokens=150,
            additional_kwargs={"top_p": 0.9},
        )

        response = llm.chat_with_tools(
            user_msg="Calculate the result of 2 + 2.",
            tools=[multiply_tool, add_tool, sub_tool, divide_tool],
        )
        print(response)
        ```
    """

    endpoint: str = Field(
        default=None, description="The URI of the endpoint from the deployed model."
    )

    auth: Dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description=(
            "The authentication dictionary used for OCI API requests. Default is an empty dictionary. "
            "If not provided, it will be autogenerated based on the environment variables. "
            "https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html."
        ),
    )
    model: Optional[str] = Field(
        default=DEFAULT_MODEL,
        description="The OCI Data Science default model. Defaults to `odsc-llm`.",
    )
    temperature: Optional[float] = Field(
        default=DEFAULT_TEMPERATURE,
        description="A non-negative float that tunes the degree of randomness in generation.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: Optional[int] = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Denotes the number of tokens to predict per generation.",
        gt=0,
    )
    timeout: float = Field(
        default=DEFAULT_TIMEOUT, description="The timeout to use in seconds.", ge=0
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        description="The maximum number of API retries.",
        ge=0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    is_chat_model: bool = Field(
        default=True,
        description="If the model exposes a chat interface.",
    )
    is_function_calling_model: bool = Field(
        default=True,
        description="If the model supports function calling messages.",
    )
    additional_kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional kwargs for the OCI Data Science AI request.",
    )
    strict: bool = Field(
        default=False,
        description="Whether to use strict mode for invoking tools/using schemas.",
    )
    default_headers: Optional[Dict[str, str]] = Field(
        default=None, description="The default headers for API requests."
    )

    _client: Client = PrivateAttr()
    _async_client: AsyncClient = PrivateAttr()

    def __init__(
        self,
        endpoint: str,
        auth: Optional[Dict[str, Any]] = None,
        model: Optional[str] = DEFAULT_MODEL,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
        context_window: Optional[int] = DEFAULT_CONTEXT_WINDOW,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        max_retries: Optional[int] = DEFAULT_MAX_RETRIES,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        is_chat_model: Optional[bool] = True,
        is_function_calling_model: Optional[bool] = True,
        default_headers: Optional[Dict[str, str]] = None,
        # base class
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        strict: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the OCIDataScience LLM class.

        Args:
            endpoint (str): The URI of the endpoint from the deployed model.
            auth (Optional[Dict[str, Any]]): Authentication dictionary for OCI API requests.
            model (Optional[str]): The model name to use. Defaults to `odsc-llm`.
            temperature (Optional[float]): Controls the randomness in generation.
            max_tokens (Optional[int]): Number of tokens to predict per generation.
            context_window (Optional[int]): Maximum number of context tokens for the model.
            timeout (Optional[float]): Timeout for API requests in seconds.
            max_retries (Optional[int]): Maximum number of API retries.
            additional_kwargs (Optional[Dict[str, Any]]): Additional parameters for the API request.
            callback_manager (Optional[CallbackManager]): Callback manager for LLM.
            is_chat_model (Optional[bool]): If the model exposes a chat interface. Defaults to `True`.
            is_function_calling_model (Optional[bool]): If the model supports function calling messages. Defaults to `True`.
            default_headers (Optional[Dict[str, str]]): The default headers for API requests.
            system_prompt (Optional[str]): System prompt to use.
            messages_to_prompt (Optional[Callable]): Function to convert messages to prompt.
            completion_to_prompt (Optional[Callable]): Function to convert completion to prompt.
            pydantic_program_mode (PydanticProgramMode): Pydantic program mode.
            output_parser (Optional[BaseOutputParser]): Output parser for the LLM.
            strict (bool): Whether to use strict mode for invoking tools/using schemas.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            endpoint=endpoint,
            model=model,
            auth=auth,
            temperature=temperature,
            context_window=context_window,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager or CallbackManager([]),
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model,
            default_headers=default_headers,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            strict=strict,
            **kwargs,
        )

        self._client: Client = None
        self._async_client: AsyncClient = None

        logger.debug(
            f"Initialized OCIDataScience LLM with endpoint: {self.endpoint} and model: {self.model}"
        )

    @model_validator(mode="before")
    # @_validate_dependency
    def validate_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the environment and dependencies."""
        return values

    @property
    def client(self) -> Client:
        """
        Synchronous client for interacting with the OCI Data Science Model Deployment endpoint.

        Returns:
            Client: The synchronous client instance.
        """
        if self._client is None:
            self._client = Client(
                endpoint=self.endpoint,
                auth=self.auth,
                retries=self.max_retries,
                timeout=self.timeout,
            )
        return self._client

    @property
    def async_client(self) -> AsyncClient:
        """
        Asynchronous client for interacting with the OCI Data Science Model Deployment endpoint.

        Returns:
            AsyncClient: The asynchronous client instance.
        """
        if self._async_client is None:
            self._async_client = AsyncClient(
                endpoint=self.endpoint,
                auth=self.auth,
                retries=self.max_retries,
                timeout=self.timeout,
            )
        return self._async_client

    @classmethod
    def class_name(cls) -> str:
        """
        Return the class name.

        Returns:
            str: The name of the class.
        """
        return "OCIDataScience_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        """
        Return the metadata of the LLM.

        Returns:
            LLMMetadata: The metadata of the LLM.
        """
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    def _model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Get model-specific parameters for the API request.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: The combined model parameters.
        """
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {**base_kwargs, **self.additional_kwargs, **kwargs}

    def _prepare_headers(
        self,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Construct and return the headers for a request.

        Args:
            headers (Optional[Dict[str, str]]): HTTP headers to include in the request.

        Returns:
            Dict[str, str]: The prepared headers.
        """
        return {**(self.default_headers or {}), **(headers or {})}

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str): The prompt to generate a completion for.
            formatted (bool): Whether the prompt is formatted.
            **kwargs: Additional keyword arguments.

        Returns:
            CompletionResponse: The response from the LLM.
        """
        logger.debug(f"Calling complete with prompt: {prompt}")
        response = self.client.generate(
            prompt=prompt,
            payload=self._model_kwargs(**kwargs),
            headers=self._prepare_headers(kwargs.pop("headers", {})),
            stream=False,
        )

        logger.debug(f"Received response: {response}")
        try:
            choice = response["choices"][0]
            text = choice.get("text", "")
            logprobs = _from_completion_logprobs_dict(choice.get("logprobs") or {})

            return CompletionResponse(
                text=text,
                raw=response,
                logprobs=logprobs,
                additional_kwargs=_get_response_token_counts(response),
            )
        except (IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to parse response: {e!s}") from e

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """
        Stream the completion for the given prompt.

        Args:
            prompt (str): The prompt to generate a completion for.
            formatted (bool): Whether the prompt is formatted.
            **kwargs: Additional keyword arguments.

        Yields:
            CompletionResponse: The streamed response from the LLM.
        """
        logger.debug(f"Starting stream_complete with prompt: {prompt}")
        text = ""
        for response in self.client.generate(
            prompt=prompt,
            payload=self._model_kwargs(**kwargs),
            headers=self._prepare_headers(kwargs.pop("headers", {})),
            stream=True,
        ):
            logger.debug(f"Received chunk: {response}")
            if len(response.get("choices", [])) > 0:
                delta = response["choices"][0].get("text")
                if delta is None:
                    delta = ""
            else:
                delta = ""
            text += delta

            yield CompletionResponse(
                delta=delta,
                text=text,
                raw=response,
                additional_kwargs=_get_response_token_counts(response),
            )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Generate a chat completion based on the input messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResponse: The chat response from the LLM.
        """
        logger.debug(f"Calling chat with messages: {messages}")
        response = self.client.chat(
            messages=_to_message_dicts(
                messages=messages, drop_none=kwargs.pop("drop_none", False)
            ),
            payload=self._model_kwargs(**kwargs),
            headers=self._prepare_headers(kwargs.pop("headers", {})),
            stream=False,
        )

        logger.debug(f"Received chat response: {response}")
        try:
            choice = response["choices"][0]
            message = _from_message_dict(choice.get("message", ""))
            logprobs = _from_token_logprob_dicts(
                (choice.get("logprobs") or {}).get("content", [])
            )
            return ChatResponse(
                message=message,
                raw=response,
                logprobs=logprobs,
                additional_kwargs=_get_response_token_counts(response),
            )
        except (IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to parse response: {e!s}") from e

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """
        Stream the chat completion based on the input messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatResponse: The streamed chat response from the LLM.
        """
        logger.debug(f"Starting stream_chat with messages: {messages}")
        content = ""
        is_function = False
        tool_calls = []
        for response in self.client.chat(
            messages=_to_message_dicts(
                messages=messages, drop_none=kwargs.pop("drop_none", False)
            ),
            payload=self._model_kwargs(**kwargs),
            headers=self._prepare_headers(kwargs.pop("headers", {})),
            stream=True,
        ):
            logger.debug(f"Received chat chunk: {response}")
            if len(response.get("choices", [])) > 0:
                delta = response["choices"][0].get("delta") or {}
            else:
                delta = {}

            # Check if this chunk is the start of a function call
            if delta.get("tool_calls"):
                is_function = True

            # Update using deltas
            role = delta.get("role") or MessageRole.ASSISTANT
            content_delta = delta.get("content") or ""
            content += content_delta

            additional_kwargs = {}
            if is_function:
                tool_calls = _update_tool_calls(tool_calls, delta.get("tool_calls"))
                if tool_calls:
                    additional_kwargs["tool_calls"] = tool_calls

            yield ChatResponse(
                message=ChatMessage(
                    role=role,
                    content=content,
                    additional_kwargs=additional_kwargs,
                ),
                delta=content_delta,
                raw=response,
                additional_kwargs=_get_response_token_counts(response),
            )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """
        Asynchronously generate a completion for the given prompt.

        Args:
            prompt (str): The prompt to generate a completion for.
            formatted (bool): Whether the prompt is formatted.
            **kwargs: Additional keyword arguments.

        Returns:
            CompletionResponse: The response from the LLM.
        """
        logger.debug(f"Calling acomplete with prompt: {prompt}")
        response = await self.async_client.generate(
            prompt=prompt,
            payload=self._model_kwargs(**kwargs),
            headers=self._prepare_headers(kwargs.pop("headers", {})),
            stream=False,
        )

        logger.debug(f"Received async response: {response}")
        try:
            choice = response["choices"][0]
            text = choice.get("text", "")
            logprobs = _from_completion_logprobs_dict(choice.get("logprobs", {}) or {})

            return CompletionResponse(
                text=text,
                raw=response,
                logprobs=logprobs,
                additional_kwargs=_get_response_token_counts(response),
            )
        except (IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to parse response: {e!s}") from e

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """
        Asynchronously stream the completion for the given prompt.

        Args:
            prompt (str): The prompt to generate a completion for.
            formatted (bool): Whether the prompt is formatted.
            **kwargs: Additional keyword arguments.

        Yields:
            CompletionResponse: The streamed response from the LLM.
        """

        async def gen() -> CompletionResponseAsyncGen:
            logger.debug(f"Starting astream_complete with prompt: {prompt}")
            text = ""

            async for response in await self.async_client.generate(
                prompt=prompt,
                payload=self._model_kwargs(**kwargs),
                headers=self._prepare_headers(kwargs.pop("headers", {})),
                stream=True,
            ):
                logger.debug(f"Received async chunk: {response}")
                if len(response.get("choices", [])) > 0:
                    delta = response["choices"][0].get("text")
                    if delta is None:
                        delta = ""
                else:
                    delta = ""
                text += delta

                yield CompletionResponse(
                    delta=delta,
                    text=text,
                    raw=response,
                    additional_kwargs=_get_response_token_counts(response),
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """
        Asynchronously generate a chat completion based on the input messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResponse: The chat response from the LLM.
        """
        logger.debug(f"Calling achat with messages: {messages}")
        response = await self.async_client.chat(
            messages=_to_message_dicts(
                messages=messages, drop_none=kwargs.pop("drop_none", False)
            ),
            payload=self._model_kwargs(**kwargs),
            headers=self._prepare_headers(kwargs.pop("headers", {})),
            stream=False,
        )

        logger.debug(f"Received async chat response: {response}")
        try:
            choice = response["choices"][0]
            message = _from_message_dict(choice.get("message", ""))
            logprobs = _from_token_logprob_dicts(
                (choice.get("logprobs") or {}).get("content", {})
            )
            return ChatResponse(
                message=message,
                raw=response,
                logprobs=logprobs,
                additional_kwargs=_get_response_token_counts(response),
            )
        except (IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to parse response: {e!s}") from e

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """
        Asynchronously stream the chat completion based on the input messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatResponse: The streamed chat response from the LLM.
        """

        async def gen() -> ChatResponseAsyncGen:
            logger.debug(f"Starting astream_chat with messages: {messages}")
            content = ""
            is_function = False
            tool_calls = []
            async for response in await self.async_client.chat(
                messages=_to_message_dicts(
                    messages=messages, drop_none=kwargs.pop("drop_none", False)
                ),
                payload=self._model_kwargs(**kwargs),
                headers=self._prepare_headers(kwargs.pop("headers", {})),
                stream=True,
            ):
                logger.debug(f"Received async chat chunk: {response}")
                if len(response.get("choices", [])) > 0:
                    delta = response["choices"][0].get("delta") or {}
                else:
                    delta = {}

                # Check if this chunk is the start of a function call
                if delta.get("tool_calls"):
                    is_function = True

                # Update using deltas
                role = delta.get("role") or MessageRole.ASSISTANT
                content_delta = delta.get("content") or ""
                content += content_delta

                additional_kwargs = {}
                if is_function:
                    tool_calls = _update_tool_calls(tool_calls, delta.get("tool_calls"))
                    if tool_calls:
                        additional_kwargs["tool_calls"] = tool_calls

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content_delta,
                    raw=response,
                    additional_kwargs=_get_response_token_counts(response),
                )

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Union[str, dict] = DEFAULT_TOOL_CHOICE,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Prepare the chat input with tools for function calling.

        Args:
            tools (List[BaseTool]): A list of tools to use.
            user_msg (Optional[Union[str, ChatMessage]]): The user's message.
            chat_history (Optional[List[ChatMessage]]): The chat history.
            verbose (bool): Whether to output verbose logs.
            allow_parallel_tool_calls (bool): Whether to allow parallel tool calls.
            tool_choice (Union[str, dict]): Tool choice strategy.
            strict (Optional[bool]): Whether to enforce strict mode.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: The prepared parameters for the chat request.
        """
        tool_specs = [tool.metadata.to_openai_tool() for tool in tools]

        logger.debug(
            f"Preparing chat with tools. Tools: {tool_specs}, User message: {user_msg}, "
            f"Chat history: {chat_history}"
        )

        # Determine strict mode
        strict = strict or self.strict

        if self.metadata.is_function_calling_model:
            for tool_spec in tool_specs:
                if tool_spec["type"] == "function":
                    if strict:
                        tool_spec["function"]["strict"] = strict
                    tool_spec["function"]["parameters"]["additionalProperties"] = False

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": (_resolve_tool_choice(tool_choice) if tool_specs else None),
            **kwargs,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: List["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Validate the response from chat_with_tools.

        Args:
            response (ChatResponse): The chat response to validate.
            tools (List[BaseTool]): A list of tools used.
            allow_parallel_tool_calls (bool): Whether parallel tool calls are allowed.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResponse: The validated chat response.
        """
        if not allow_parallel_tool_calls:
            # Ensures that the 'tool_calls' in the response contain only a single tool call.
            tool_calls = response.message.additional_kwargs.get("tool_calls", [])
            if len(tool_calls) > 1:
                logger.warning(
                    "Multiple tool calls detected but parallel tool calls are not allowed. "
                    "Limiting to the first tool call."
                )
                response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]
        return response

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """
        Extract tool calls from the chat response.

        Args:
            response (ChatResponse): The chat response containing tool calls.
            error_on_no_tool_call (bool): Whether to raise an error if no tool calls are found.
            **kwargs: Additional keyword arguments.

        Returns:
            List[ToolSelection]: A list of tool selections extracted from the response.

        Raises:
            ValueError: If no tool calls are found and error_on_no_tool_call is True.
        """
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])
        logger.debug(f"Getting tool calls from response: {tool_calls}")

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            if tool_call.get("type") != "function":
                raise ValueError(f"Invalid tool type detected: {tool_call.get('type')}")

            # Handle both complete and partial JSON
            try:
                argument_dict = parse_partial_json(
                    tool_call.get("function", {}).get("arguments", {})
                )
            except ValueError as e:
                logger.debug(f"Failed to parse tool call arguments: {e!s}")
                argument_dict = {}

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.get("id"),
                    tool_name=tool_call.get("function", {}).get("name"),
                    tool_kwargs=argument_dict,
                )
            )

        logger.debug(
            f"Extracted tool calls: { [tool_selection.model_dump() for tool_selection in tool_selections] }"
        )
        return tool_selections
