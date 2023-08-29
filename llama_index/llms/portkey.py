"""_summary_

Returns:
    _type_: _description_
"""
from typing import Any, Optional, Sequence, Dict, Union, List

# from rubeus import Rubeus, LLMBase
from rubeus import (
    Rubeus,
    LLMBase,
    RubeusModes,
    RubeusResponse,
    RubeusModesLiteral,
    ProviderTypes,
    ProviderTypesLiteral,
    RubeusCacheType,
    RubeusCacheLiteral,
    Message,
)
from llama_index.llms.custom import CustomLLM
from llama_index.llms.base import (
    ChatMessage,
    LLMMetadata,
    ChatResponse,
    CompletionResponse,
    ChatResponseGen,
    llm_completion_callback,
    llm_chat_callback,
)
from llama_index.llms.portkey_utils import (
    is_chat_model,
    generate_llm_metadata,
    get_llm,
)
from llama_index.llms.generic_utils import (
    completion_to_chat_decorator,
    chat_to_completion_decorator,
    stream_completion_to_chat_decorator,
)

try:
    from pydantic.v1 import Field, PrivateAttr
except ImportError:
    from pydantic import Field, PrivateAttr


class Portkey(CustomLLM):
    """_summary_

    Args:
        LLM (_type_): _description_
    """

    mode: Optional[Union[RubeusModes, RubeusModesLiteral]] = Field(
        description="The mode for using the Portkey integration\
            (default: RubeusModes.PROXY)",
        default=RubeusModes.SINGLE,
    )

    model: str = Field(default="gpt-3.5-turbo")
    provider: Union[ProviderTypes, ProviderTypesLiteral] = Field(
        default=ProviderTypes.OPENAI
    )
    llm: LLMBase = Field(description="LLM parameter", default_factory=dict)

    llms: List[LLMBase] = Field(description="LLM parameters", default_factory=list)

    _client: Rubeus = PrivateAttr()
    _portkey_response: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(
        self,
        *,
        api_key: str = "",
        provider: Union[ProviderTypes, ProviderTypesLiteral] = ProviderTypes.OPENAI,
        mode: Optional[Union[RubeusModes, RubeusModesLiteral]] = RubeusModes.SINGLE,
        model: str = "gpt-3.5-turbo",
        model_api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        max_retries: int = 5,
        trace_id: Optional[str] = "",
        cache_status: Optional[Union[RubeusCacheType, RubeusCacheLiteral]] = None,
        cache: Optional[bool] = False,
        metadata: Optional[Dict[str, Any]] = {},
        weight: Optional[float] = 1.0,
        # **kwargs: Any,
    ) -> None:
        """
        Initialize a Portkey instance.

        Args:
            api_key (Optional[str]): The API key to authenticate with Portkey.
            mode (Optional[RubeusModes]): The mode for using the Portkey integration
            (default: RubeusModes.PROXY).
            provider (Optional[ProviderTypes]): The LLM provider to be used for the
                Portkey integration.
                Eg: openai, anthropic etc.
                NOTE: Check the ProviderTypes to see the supported list
                of LLMs.
            model (str): The name of the language model to use
            (default: "gpt-3.5-turbo").
            model_api_key (Optional[str]): The api key of the provider being used.
                Eg: api key of openai.
            temperature (float): The temperature parameter for text generation
            (default: 0.1).
            max_tokens (Optional[int]): The maximum number of tokens in the generated
            text.
            max_retries (int): The maximum number of retries for failed requests
            (default: 5).
            trace_id (Optional[str]): A unique identifier for tracing requests.
            cache_status (Optional[RubeusCacheType]): The type of cache to use
            (default: "").
                If cache_status is set, then cache is automatically set to True
            cache (Optional[bool]): Whether to use caching (default: False).
            metadata (Optional[Dict[str, Any]]): Metadata associated with the
            request (default: {}).
            weight (Optional[float]): The weight of the LLM in the ensemble
            (default: 1.0).
            **kwargs (Any): Additional keyword arguments.

        Raises:
            ValueError: If neither 'llm' nor 'llms' are provided during
            Portkey initialization.
        """
        self._client = Rubeus(api_key=api_key)
        self._portkey_response = None
        super().__init__(
            mode=mode,
            model=model,
            provider=provider,
            model_api_key=model_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            trace_id=trace_id,
            cache_status=cache_status,
            cache=cache,
            metadata=metadata,
            weight=weight,
        )

        self.model = model
        self.provider = provider

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return generate_llm_metadata(self.llm)

    def add_llms(self, llm_params: Union[LLMBase, List[LLMBase]]) -> "Portkey":
        """
        Adds the specified LLM parameters to the list of LLMs. This may be used for
        fallbacks or load-balancing as specified in the mode.

        Args:
            llm_params (Union[LLMBase, List[LLMBase]]): A single LLM parameter set or
            a list of LLM parameter sets. Each set should be an instance of LLMBase with
            the specified attributes.
                > provider: Optional[ProviderTypes]
                > model: str
                > temperature: float
                > max_tokens: Optional[int]
                > max_retries: int
                > trace_id: Optional[str]
                > cache_status: Optional[RubeusCacheType]
                > cache: Optional[bool]
                > metadata: Dict[str, Any]
                > weight: Optional[float]

            NOTE: User may choose to pass additional params as well.
        Returns:
            self
        """
        if isinstance(llm_params, LLMBase):
            llm_params = [llm_params]
        self.llms.extend(llm_params)
        return self

    def add_llm(self, llm_params: LLMBase) -> "Portkey":
        """
        Adds the specified LLM parameters to the list of fallback LLMs.

        Args:
            llm_params (LLMBase): The parameters representing the LLM to be added.
                Should be an instance of LLMBase with the following attributes:
                - provider: Optional[ProviderTypes]
                - model: str
                - temperature: float
                - max_tokens: Optional[int]
                - max_retries: int
                - trace_id: Optional[str]
                - cache_status: Optional[RubeusCacheType]
                - cache: Optional[bool]
                - metadata: Dict[str, Any]
                - weight: Optional[float]

        Returns:
            None
        """
        if self.llm is not None:
            raise ValueError(
                "Duplicate LLM: An LLM is already add as part of the configuration."
            )
        self.llm = llm_params
        return self

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint for LLM."""
        if self._is_chat_model:
            complete_fn = chat_to_completion_decorator(self.chat)
        else:
            complete_fn = self.complete
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self._is_chat_model:
            chat_fn = self._chat
        else:
            chat_fn = completion_to_chat_decorator(self._complete)
        return chat_fn(messages, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint for LLM."""
        if self._is_chat_model:
            complete_fn = chat_to_completion_decorator(self.chat)
        else:
            complete_fn = self.complete
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if self._is_chat_model:
            stream_chat_fn = self.stream_chat
        else:
            stream_chat_fn = stream_completion_to_chat_decorator(self.stream_complete)
        return stream_chat_fn(messages, **kwargs)

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messages_dict = [{"role": i.role.value, "content": i.content} for i in messages]
        self._client.default_params["messages"] = messages_dict  # type: ignore
        if self.mode == RubeusModes.FALLBACK:
            response = self._client.chat_completion.with_fallbacks(self.llms)
            # self.llm = self._get_llm(response)

        elif self.mode == RubeusModes.LOADBALANCE:
            response = self._client.chat_completion.with_loadbalancing(self.llms)
            # self.llm = self._get_llm(response)
        else:
            messages_input = [
                Message(role=i.role.value, content=i.content or "") for i in messages
            ]
            response = self._client.chat_completion.create(
                messages=messages_input, **kwargs
            )

        message = response.choices[0]["message"]
        raw = response.raw_body
        return ChatResponse(message=message, raw=raw)

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        self._client.default_params["prompt"] = prompt  # type: ignore
        if self.mode == RubeusModes.FALLBACK:
            response = self._client.completion.with_fallbacks(self.llms)
        elif self.mode == RubeusModes.LOADBALANCE:
            response = self._client.completion.with_loadbalancing(self.llms)
        else:
            response = self._client.completion.create(prompt=prompt, **kwargs)

        text = response.choices[0]["text"]
        raw = response.raw_body
        return CompletionResponse(text=text, raw=raw)

    @property
    def _is_chat_model(self) -> bool:
        """Check if a given model is a chat-based language model.

        Returns:
            bool: True if the provided model is a chat-based language model,
            False otherwise.
        """
        return is_chat_model(self.model)

    @property
    def _is_fallback_mode(self) -> bool:
        """Check if the suggested mode is fallback or not.

        Returns:
            bool: True if the provided mode is fallback type, False otherwise.
        """
        return self.mode == RubeusModes.FALLBACK

    # def _get_llm(self, response: RubeusResponse) -> LLMBase:
    #     return get_llm(response, self.llms)
