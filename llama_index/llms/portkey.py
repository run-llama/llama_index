"""
Portkey integration with Llama_index for enhanced monitoring.
"""
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union, cast

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM
from llama_index.llms.generic_utils import (
    chat_to_completion_decorator,
    completion_to_chat_decorator,
    stream_chat_to_completion_decorator,
    stream_completion_to_chat_decorator,
)
from llama_index.llms.portkey_utils import (
    IMPORT_ERROR_MESSAGE,
    generate_llm_metadata,
    get_llm,
    is_chat_model,
)

if TYPE_CHECKING:
    from portkey import (
        LLMOptions,
        Modes,
        ModesLiteral,
        PortkeyResponse,
    )

DEFAULT_PORTKEY_MODEL = "gpt-3.5-turbo"


class Portkey(CustomLLM):
    """_summary_.

    Args:
        LLM (_type_): _description_
    """

    mode: Optional[Union["Modes", "ModesLiteral"]] = Field(
        description="The mode for using the Portkey integration"
    )

    model: Optional[str] = Field(default=DEFAULT_PORTKEY_MODEL)
    llm: "LLMOptions" = Field(description="LLM parameter", default_factory=dict)

    llms: List["LLMOptions"] = Field(description="LLM parameters", default_factory=list)

    _client: Any = PrivateAttr()

    def __init__(
        self,
        *,
        mode: Union["Modes", "ModesLiteral"],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize a Portkey instance.

        Args:
            mode (Optional[Modes]): The mode for using the Portkey integration
            (default: Modes.SINGLE).
            api_key (Optional[str]): The API key to authenticate with Portkey.
            base_url (Optional[str]): The Base url to the self hosted rubeus \
                (the opensource version of portkey) or any other self hosted server.
        """
        try:
            import portkey
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        super().__init__(
            base_url=base_url,
            api_key=api_key,
        )
        if api_key is not None:
            portkey.api_key = api_key

        if base_url is not None:
            portkey.base_url = base_url

        portkey.mode = mode

        self._client = portkey
        self.model = None
        self.mode = mode

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return generate_llm_metadata(self.llms[0])

    def add_llms(
        self, llm_params: Union["LLMOptions", List["LLMOptions"]]
    ) -> "Portkey":
        """
        Adds the specified LLM parameters to the list of LLMs. This may be used for
        fallbacks or load-balancing as specified in the mode.

        Args:
            llm_params (Union[LLMOptions, List[LLMOptions]]): A single LLM parameter \
            set or a list of LLM parameter sets. Each set should be an instance of \
            LLMOptions with
            the specified attributes.
                > provider: Optional[ProviderTypes]
                > model: str
                > temperature: float
                > max_tokens: Optional[int]
                > max_retries: int
                > trace_id: Optional[str]
                > cache_status: Optional[CacheType]
                > cache: Optional[bool]
                > metadata: Dict[str, Any]
                > weight: Optional[float]
                > **kwargs : Other additional parameters that are supported by \
                    LLMOptions in portkey-ai

            NOTE: User may choose to pass additional params as well.

        Returns:
            self
        """
        try:
            from portkey import LLMOptions
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc
        if isinstance(llm_params, LLMOptions):
            llm_params = [llm_params]
        self.llms.extend(llm_params)
        if self.model is None:
            self.model = self.llms[0].model
        return self

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint for LLM."""
        if self._is_chat_model:
            complete_fn = chat_to_completion_decorator(self._chat)
        else:
            complete_fn = self._complete
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self._is_chat_model:
            chat_fn = self._chat
        else:
            chat_fn = completion_to_chat_decorator(self._complete)
        return chat_fn(messages, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Completion endpoint for LLM."""
        if self._is_chat_model:
            complete_fn = stream_chat_to_completion_decorator(self._stream_chat)
        else:
            complete_fn = self._stream_complete
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if self._is_chat_model:
            stream_chat_fn = self._stream_chat
        else:
            stream_chat_fn = stream_completion_to_chat_decorator(self._stream_complete)
        return stream_chat_fn(messages, **kwargs)

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        try:
            from portkey import Config, Message
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc
        _messages = cast(
            List[Message],
            [{"role": i.role.value, "content": i.content} for i in messages],
        )
        config = Config(llms=self.llms)
        response = self._client.ChatCompletions.create(
            messages=_messages, config=config
        )
        self.llm = self._get_llm(response)

        message = response.choices[0].message
        return ChatResponse(message=message, raw=response)

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            from portkey import Config
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        config = Config(llms=self.llms)
        response = self._client.Completions.create(prompt=prompt, config=config)
        text = response.choices[0].text
        return CompletionResponse(text=text, raw=response)

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        try:
            from portkey import Config, Message
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc
        _messages = cast(
            List[Message],
            [{"role": i.role.value, "content": i.content} for i in messages],
        )
        config = Config(llms=self.llms)
        response = self._client.ChatCompletions.create(
            messages=_messages, config=config, stream=True, **kwargs
        )

        def gen() -> ChatResponseGen:
            content = ""
            function_call: Optional[dict] = {}
            for resp in response:
                if resp.choices is None:
                    continue
                delta = resp.choices[0].delta
                role = delta.get("role", "assistant")
                content_delta = delta.get("content", "") or ""
                content += content_delta

                function_call_delta = delta.get("function_call", None)
                if function_call_delta is not None:
                    if function_call is None:
                        function_call = function_call_delta
                        # ensure we do not add a blank function call
                        if (
                            function_call
                            and function_call.get("function_name", "") is None
                        ):
                            del function_call["function_name"]
                    else:
                        function_call["arguments"] += function_call_delta["arguments"]

                additional_kwargs = {}
                if function_call is not None:
                    additional_kwargs["function_call"] = function_call

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content_delta,
                    raw=resp,
                )

        return gen()

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        try:
            from portkey import Config
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        config = Config(llms=self.llms)
        response = self._client.Completions.create(
            prompt=prompt, config=config, stream=True, **kwargs
        )

        def gen() -> CompletionResponseGen:
            text = ""
            for resp in response:
                delta = resp.choices[0].text or ""
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                    raw=resp,
                )

        return gen()

    @property
    def _is_chat_model(self) -> bool:
        """Check if a given model is a chat-based language model.

        Returns:
            bool: True if the provided model is a chat-based language model,
            False otherwise.
        """
        return is_chat_model(self.model or "")

    def _get_llm(self, response: "PortkeyResponse") -> "LLMOptions":
        return get_llm(response, self.llms)
