import json
from typing import Any, Callable, Dict, Optional, Sequence, List, Union, TYPE_CHECKING
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_TEMPERATURE

from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
    MessageRole,
    ChatMessage,
)

from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)

from llama_index.llms.deepinfra.utils import (
    chat_messages_to_list,
    maybe_extract_from_json,
    force_single_tool_call,
)

from llama_index.llms.deepinfra.constants import (
    API_BASE,
    INFERENCE_ENDPOINT,
    CHAT_API_ENDPOINT,
    ENV_VARIABLE,
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_TOKENS,
    TOOL_CHOICE,
)

from llama_index.llms.deepinfra.client import DeepInfraClient
from llama_index.llms.deepinfra.types import ToolCallMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM, ToolSelection

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


class DeepInfraLLM(FunctionCallingLLM):
    """DeepInfra LLM.

    Examples:
        `pip install llama-index-llms-deepinfra`

        ```python
        from llama_index.llms.deepinfra import DeepInfraLLM

        llm = DeepInfraLLM(
            model="mistralai/Mixtral-8x22B-Instruct-v0.1", # Default model name
            api_key = "your-deepinfra-api-key",
            temperature=0.5,
            max_tokens=50,
            additional_kwargs={"top_p": 0.9},
        )

        response = llm.complete("Hello World!")
        print(response)
        ```
    """

    model: str = Field(
        default=DEFAULT_MODEL_NAME, description="The DeepInfra model to use."
    )

    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: Optional[int] = Field(
        default=DEFAULT_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    timeout: Optional[float] = Field(
        default=None, description="The timeout to use in seconds.", ge=0
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", ge=0
    )

    _api_key: Optional[str] = PrivateAttr()

    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments for generation."
    )

    _client: DeepInfraClient = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_MODEL_NAME,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
        max_retries: int = 10,
        api_base: str = API_BASE,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )
        self._api_key = get_from_param_or_env("api_key", api_key, ENV_VARIABLE)
        self._client = DeepInfraClient(
            api_key=self._api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def class_name(cls) -> str:
        return "DeepInfra_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            num_output=self.max_tokens,
            is_chat_model=self._is_chat_model,
            model=self.model,
            model_name=self.model,
            is_function_calling_model=self._client.is_function_calling_model(
                self.model
            ),
        )

    @property
    def _is_chat_model(self) -> bool:
        return True

    # Synchronous Methods
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Generate completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            str: The generated text completion.
        """
        payload = self._build_payload(prompt=prompt, **kwargs)
        result = self._client.request(INFERENCE_ENDPOINT, payload)
        return CompletionResponse(text=maybe_extract_from_json(result), raw=result)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """
        Generate a synchronous streaming completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Yields:
            CompletionResponseGen: The streaming text completion.
        """
        payload = self._build_payload(prompt=prompt, **kwargs)

        content = ""
        for response_dict in self._client.request_stream(INFERENCE_ENDPOINT, payload):
            content_delta = maybe_extract_from_json(response_dict)
            content += content_delta
            yield CompletionResponse(
                text=content, delta=content_delta, raw=response_dict
            )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        """
        Generate a chat response for the given messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            ChatResponse: The chat response containing a sequence of messages.
        """
        messages = chat_messages_to_list(messages)
        payload = self._build_payload(messages=messages, **kwargs)
        result = self._client.request(CHAT_API_ENDPOINT, payload)
        mo = result["choices"][-1]["message"]
        additional_kwargs = {
            "tool_calls": mo.get("tool_calls", []) or [],
        }
        return ChatResponse(
            message=ChatMessage(
                role=mo["role"],
                content=mo["content"],
                additional_kwargs=additional_kwargs,
            ),
            raw=result,
        )

    @llm_chat_callback()
    def stream_chat(
        self, chat_messages: Sequence[ChatMessage], **kwargs
    ) -> ChatResponseGen:
        """
        Generate a synchronous streaming chat response for the given messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API request.

        Yields:
            ChatResponseGen: The chat response containing a sequence of messages.
        """
        messages = chat_messages_to_list(chat_messages)
        payload = self._build_payload(messages=messages, **kwargs)

        content = ""
        role = MessageRole.ASSISTANT
        for response_dict in self._client.request_stream(CHAT_API_ENDPOINT, payload):
            delta = response_dict["choices"][-1]["delta"]
            """
            Check if the delta contains content.
            """
            if delta.get("content", None):
                content_delta = delta["content"]
                content += delta["content"]
                message = ChatMessage(
                    role=role,
                    content=content,
                )
                yield ChatResponse(
                    message=message, raw=response_dict, delta=content_delta
                )

    # Asynchronous Methods
    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Asynchronously generate completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            CompletionResponse: The generated text completion.
        """
        payload = self._build_payload(prompt=prompt, **kwargs)

        result = await self._client.arequest(INFERENCE_ENDPOINT, payload)
        return CompletionResponse(text=maybe_extract_from_json(result), raw=result)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs
    ) -> CompletionResponseAsyncGen:
        """
        Asynchronously generate a streaming completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Yields:
            CompletionResponseAsyncGen: The streaming text completion.
        """
        payload = self._build_payload(prompt=prompt, **kwargs)

        async def gen():
            content = ""
            async for response_dict in self._client.arequest_stream(
                INFERENCE_ENDPOINT, payload
            ):
                content_delta = maybe_extract_from_json(response_dict)
                content += content_delta
                yield CompletionResponse(
                    text=content, delta=content_delta, raw=response_dict
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self, chat_messages: Sequence[ChatMessage], **kwargs
    ) -> ChatResponse:
        """
        Asynchronously generate a chat response for the given messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            ChatResponse: The chat response containing a sequence of messages.
        """
        messages = chat_messages_to_list(chat_messages)
        payload = self._build_payload(messages=messages, **kwargs)

        result = await self._client.arequest(CHAT_API_ENDPOINT, payload)
        mo = result["choices"][-1]["message"]
        additional_kwargs = {"tool_calls": mo.get("tool_calls", []) or []}
        return ChatResponse(
            message=ChatMessage(
                role=mo["role"],
                content=mo["content"],
                additional_kwargs=additional_kwargs,
            ),
            raw=result,
        )

    @llm_chat_callback()
    async def astream_chat(
        self, chat_messages: Sequence[ChatMessage], **kwargs
    ) -> ChatResponseAsyncGen:
        """
        Asynchronously generate a streaming chat response for the given messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API request.

        Yields:
            ChatResponseAsyncGen: The chat response containing a sequence of messages.
        """
        messages = chat_messages_to_list(chat_messages)
        payload = self._build_payload(messages=messages, **kwargs)

        async def gen():
            content = ""
            role = MessageRole.ASSISTANT
            async for response_dict in self._client.arequest_stream(
                CHAT_API_ENDPOINT, payload
            ):
                delta = response_dict["choices"][-1]["delta"]
                """
                Check if the delta contains content.
                """
                if delta.get("content", None):
                    content_delta = delta["content"]
                    content += delta["content"]
                    message = ChatMessage(
                        role=role,
                        content=content,
                    )
                    yield ChatResponse(
                        message=message, raw=response_dict, delta=content_delta
                    )

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Union[str, dict] = "auto",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tool_specs = [tool.metadata.to_openai_tool() for tool in tools]
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": TOOL_CHOICE,
            **kwargs,
        }

    def _validate_chat_with_tools_response(
        self,
        response: "ChatResponse",
        tools: List["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])
        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call_dict in tool_calls:
            tool_call = ToolCallMessage.parse_obj(tool_call_dict)
            argument_dict = json.loads(tool_call.function.arguments)

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    # Utility Methods
    def get_model_endpoint(self) -> str:
        """
        Get DeepInfra model endpoint.
        """
        return f"{INFERENCE_ENDPOINT}/{self.model}"

    def _build_payload(self, **kwargs) -> Dict[str, Any]:
        """
        Build the payload for the API request.
        The temperature and max_tokens parameters explicitly override
        the corresponding values in generate_kwargs.
        Any provided kwargs override all other parameters, including temperature and max_tokens.

        Args:
            prompt (str): The input prompt to generate completion for.
            stream (bool): Whether to stream the response.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            Dict[str, Any]: The API request payload.
        """
        return {
            **self.generate_kwargs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model": self.model,
            **kwargs,
        }
