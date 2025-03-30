import json
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Sequence,
    List,
    Union,
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
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    acompletion_to_chat_decorator,
    astream_chat_to_completion_decorator,
    astream_completion_to_chat_decorator,
    chat_to_completion_decorator,
    completion_to_chat_decorator,
    stream_chat_to_completion_decorator,
    stream_completion_to_chat_decorator,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.tools import ToolSelection
from llama_index.core.tools.types import BaseTool
from llama_index.llms.litellm.utils import (
    acompletion_with_retry,
    completion_with_retry,
    from_litellm_message,
    is_function_calling_model,
    openai_modelname_to_contextsize,
    to_openai_message_dicts,
    validate_litellm_api_key,
    update_tool_calls,
)


DEFAULT_LITELLM_MODEL = "gpt-3.5-turbo"


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


class LiteLLM(FunctionCallingLLM):
    """LiteLLM.

    Examples:
        `pip install llama-index-llms-litellm`

        ```python
        import os
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.litellm import LiteLLM

        # Set environment variables
        os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
        os.environ["COHERE_API_KEY"] = "your-cohere-api-key"

        # Define a chat message
        message = ChatMessage(role="user", content="Hey! how's it going?")

        # Initialize LiteLLM with the desired model
        llm = LiteLLM(model="gpt-3.5-turbo")

        # Call the chat method with the message
        chat_response = llm.chat([message])

        # Print the response
        print(chat_response)
        ```
    """

    model: str = Field(
        default=DEFAULT_LITELLM_MODEL,
        description=(
            "The LiteLLM model to use. "
            "For complete list of providers https://docs.litellm.ai/docs/providers"
        ),
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the LLM API.",
        # for all inputs https://docs.litellm.ai/docs/completion/input
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries."
    )

    _custom_llm_provider: Optional[str] = PrivateAttr(default=None)

    def __init__(
        self,
        model: str = DEFAULT_LITELLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        api_type: Optional[str] = None,
        api_base: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        **kwargs: Any,
    ) -> None:
        if "custom_llm_provider" in kwargs:
            if (
                kwargs["custom_llm_provider"] != "ollama"
                and kwargs["custom_llm_provider"] != "vllm"
            ):  # don't check keys for local models
                validate_litellm_api_key(api_key, api_type)
        else:  # by default assume it's a hosted endpoint
            validate_litellm_api_key(api_key, api_type)

        additional_kwargs = additional_kwargs or {}
        if api_key is not None:
            additional_kwargs["api_key"] = api_key
        if api_type is not None:
            additional_kwargs["api_type"] = api_type
        if api_base is not None:
            additional_kwargs["api_base"] = api_base

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            **kwargs,
        )

        self._custom_llm_provider = kwargs.get("custom_llm_provider", None)

    def _get_model_name(self) -> str:
        model_name = self.model
        if "ft-" in model_name:  # legacy fine-tuning
            model_name = model_name.split(":")[0]
        elif model_name.startswith("ft:"):
            model_name = model_name.split(":")[1]

        return model_name

    @classmethod
    def class_name(cls) -> str:
        return "litellm_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=openai_modelname_to_contextsize(self._get_model_name()),
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=is_function_calling_model(
                self._get_model_name(), self._custom_llm_provider
            ),
            model_name=self.model,
        )

    def _prepare_chat_with_tools(
        self,
        tools: List[BaseTool],
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
            "parallel_tool_calls": allow_parallel_tool_calls,
            **kwargs,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: List[BaseTool],
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
            if tool_call["type"] != "function" or "function" not in tool_call:
                raise ValueError(f"Invalid tool call of type {tool_call['type']}")
            function = tool_call["function"]
            argument_dict = json.loads(function["arguments"])
            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call["id"],
                    tool_name=function["name"],
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self._is_chat_model:
            chat_fn = self._chat
        else:
            chat_fn = completion_to_chat_decorator(self._complete)
        return chat_fn(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if self._is_chat_model:
            stream_chat_fn = self._stream_chat
        else:
            stream_chat_fn = stream_completion_to_chat_decorator(self._stream_complete)
        return stream_chat_fn(messages, **kwargs)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        # litellm assumes all llms are chat llms
        if self._is_chat_model:
            complete_fn = chat_to_completion_decorator(self._chat)
        else:
            complete_fn = self._complete

        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        if self._is_chat_model:
            stream_complete_fn = stream_chat_to_completion_decorator(self._stream_chat)
        else:
            stream_complete_fn = self._stream_complete
        return stream_complete_fn(prompt, **kwargs)

    @property
    def _is_chat_model(self) -> bool:
        # litellm assumes all llms are chat llms
        return True

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
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

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if not self._is_chat_model:
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        if "max_tokens" in all_kwargs and all_kwargs["max_tokens"] is None:
            all_kwargs.pop(
                "max_tokens"
            )  # don't send max_tokens == None, this throws errors for Non OpenAI providers

        response = completion_with_retry(
            is_chat_model=self._is_chat_model,
            max_retries=self.max_retries,
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        message_dict = response["choices"][0]["message"]
        message = from_litellm_message(message_dict)

        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if not self._is_chat_model:
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        if "max_tokens" in all_kwargs and all_kwargs["max_tokens"] is None:
            all_kwargs.pop(
                "max_tokens"
            )  # don't send max_tokens == None, this throws errors for Non OpenAI providers

        def gen() -> ChatResponseGen:
            content = ""
            tool_calls: List[dict] = []
            for response in completion_with_retry(
                is_chat_model=self._is_chat_model,
                max_retries=self.max_retries,
                messages=message_dicts,
                stream=True,
                **all_kwargs,
            ):
                delta = response["choices"][0]["delta"]
                role = delta.get("role") or MessageRole.ASSISTANT
                content_delta = delta.get("content", "") or ""
                content += content_delta

                # Handle tool_calls delta
                tool_call_delta = delta.get("tool_calls", None)
                if tool_call_delta is not None and len(tool_call_delta) > 0:
                    # Pass the entire list of tool call deltas
                    tool_calls = update_tool_calls(tool_calls, tool_call_delta)

                additional_kwargs = {}
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
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError("litellm assumes all llms are chat llms.")

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("litellm assumes all llms are chat llms.")

    def _get_max_token_for_prompt(self, prompt: str) -> int:
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Please install tiktoken to use the max_tokens=None feature."
            )
        context_window = self.metadata.context_window
        try:
            encoding = tiktoken.encoding_for_model(self._get_model_name())
        except KeyError:
            encoding = encoding = tiktoken.get_encoding(
                "cl100k_base"
            )  # default to using cl10k_base
        tokens = encoding.encode(prompt)
        max_token = context_window - len(tokens)
        if max_token <= 0:
            raise ValueError(
                f"The prompt is too long for the model. "
                f"Please use a prompt that is less than {context_window} tokens."
            )
        return max_token

    def _get_response_token_counts(self, raw_response: Any) -> dict:
        """Get the token usage reported by the response."""
        if not isinstance(raw_response, dict):
            return {}

        usage = raw_response.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    # ===== Async Endpoints =====
    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        achat_fn: Callable[..., Awaitable[ChatResponse]]
        if self._is_chat_model:
            achat_fn = self._achat
        else:
            achat_fn = acompletion_to_chat_decorator(self._acomplete)
        return await achat_fn(messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        astream_chat_fn: Callable[..., Awaitable[ChatResponseAsyncGen]]
        if self._is_chat_model:
            astream_chat_fn = self._astream_chat
        else:
            astream_chat_fn = astream_completion_to_chat_decorator(
                self._astream_complete
            )
        return await astream_chat_fn(messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        if self._is_chat_model:
            acomplete_fn = achat_to_completion_decorator(self._achat)
        else:
            acomplete_fn = self._acomplete
        return await acomplete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        if self._is_chat_model:
            astream_complete_fn = astream_chat_to_completion_decorator(
                self._astream_chat
            )
        else:
            astream_complete_fn = self._astream_complete
        return await astream_complete_fn(prompt, **kwargs)

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        if not self._is_chat_model:
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = await acompletion_with_retry(
            is_chat_model=self._is_chat_model,
            max_retries=self.max_retries,
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        message_dict = response["choices"][0]["message"]
        message = from_litellm_message(message_dict)

        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        if not self._is_chat_model:
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            tool_calls: List[dict] = []
            async for response in await acompletion_with_retry(
                is_chat_model=self._is_chat_model,
                max_retries=self.max_retries,
                messages=message_dicts,
                stream=True,
                **all_kwargs,
            ):
                delta = response["choices"][0]["delta"]
                role = delta.get("role") or MessageRole.ASSISTANT
                content_delta = delta.get("content", "") or ""
                content += content_delta

                # Handle tool_calls delta
                tool_call_delta = delta.get("tool_calls", None)
                if tool_call_delta is not None and len(tool_call_delta) > 0:
                    # Pass the entire list of tool call deltas
                    tool_calls = update_tool_calls(tool_calls, tool_call_delta)

                additional_kwargs = {}
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
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    async def _acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError("litellm assumes all llms are chat llms.")

    async def _astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError("litellm assumes all llms are chat llms.")
