"""_summary_

Returns:
    _type_: _description_
"""
import json
from typing import Any, Optional, Sequence, Dict, Union, List
from llama_index.llms.custom import CustomLLM
from llama_index.llms.base import (
    LLM,
    ChatMessage,
    LLMMetadata,
    ChatResponse,
    CompletionResponse,
    ChatResponseGen,
    CompletionResponseGen,
    llm_completion_callback,
    llm_chat_callback,
)
from llama_index.llms.portkey_utils import (
    is_chat_model,
    generate_llm_metadata,
    get_fallback_llm,
    PortkeyParams
)
from llama_index.llms.generic_utils import (
    completion_to_chat_decorator,
    chat_to_completion_decorator,
    stream_completion_to_chat_decorator,
)
from llama_index.llms.rubeus_utils import RubeusModes, ProviderTypes, RubeusCacheType
from llama_index.llms.rubeus import Rubeus
from llama_index.llms.rubeus_utils import ProviderBase


class Portkey(CustomLLM):
    """_summary_

    Args:
        LLM (_type_): _description_
    """

    def __init__(
        self,
        api_key: Optional[str] = "",
        mode: Optional[RubeusModes] = RubeusModes.SINGLE,
        provider: ProviderTypes = None,
        model: str = "gpt-3.5-turbo",
        model_api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        max_retries: int = 5,
        trace_id: Optional[str] = "",
        cache_status: Optional[RubeusCacheType] = "",
        cache: Optional[bool] = False,
        metadata: Optional[Dict[str, Any]] = {},
        weight: Optional[float] = 1.0,
        **kwargs: Any,
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
                NOTE: Check the ProviderTypes to see the supported list of LLMs.
            model (str): The name of the language model to use (default: "gpt-3.5-turbo").
            model_api_key (Optional[str]): The api key of the provider being used.
                Eg: api key of openai.
            temperature (float): The temperature parameter for text generation (default: 0.1).
            max_tokens (Optional[int]): The maximum number of tokens in the generated text.
            max_retries (int): The maximum number of retries for failed requests (default: 5).
            trace_id (Optional[str]): A unique identifier for tracing requests.
            cache_status (Optional[RubeusCacheType]): The type of cache to use (default: "").
                If cache_status is set, then cache is automatically set to True
            cache (Optional[bool]): Whether to use caching (default: False).
            metadata (Optional[Dict[str, Any]]): Metadata associated with the request (default: {}).
            weight (Optional[float]): The weight of the LLM in the ensemble (default: 1.0).
            **kwargs (Any): Additional keyword arguments.

        Raises:
            ValueError: If neither 'llm' nor 'llms' are provided during Portkey initialization.
        """
        self._mode = mode
        self._llms = []
        self._llm = None

        self._client = Rubeus(api_key=api_key)

        self._model = self.metadata.model_name
        self._portkey_response = None
        self._provider = provider
        self._model_api_key = model_api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._trace_id = trace_id
        self._cache_status = cache_status
        self._cache = cache
        self._metadata = metadata
        self._weight = weight
        self._kwargs = kwargs

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return generate_llm_metadata(self._llm)

    def add_llms(
        self, llm_params: Union[PortkeyParams, List[PortkeyParams]]
    ) -> "Portkey":
        """
        Adds the specified LLM parameters to the list of LLMs. This may be used for fallbacks or
        load-balancing as specified in the mode.

        Args:
            llm_params (Union[PortkeyParams, List[PortkeyParams]]): A single LLM parameter set or
            a list of LLM parameter sets. Each set should be an instance of PortkeyParams with
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
        if isinstance(llm_params, PortkeyParams):
            llm_params = [llm_params]
        self._llms.extend(llm_params)
        return self

    def add_llm(self, llm_params: PortkeyParams) -> "Portkey":
        """
        Adds the specified LLM parameters to the list of fallback LLMs.

        Args:
            llm_params (PortkeyParams): The parameters representing the LLM to be added.
                Should be an instance of PortkeyParams with the following attributes:
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
        if self._llm is not None:
            raise ValueError(
                "Duplicate LLM: An LLM is already add as part of the configuration."
            )
        self._llm = llm_params
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
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint for LLM."""
        return self._fallback(prompt)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if self._is_chat_model:
            stream_chat_fn = self._stream_chat
        else:
            stream_chat_fn = stream_completion_to_chat_decorator(self._stream_complete)
        return stream_chat_fn(messages, **kwargs)

    def _fallback_chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        print('Comes to fallback chat...')
        base_providers = [ProviderBase(**i) for i in self._llms]
        # [(key, value) for d in list_of_dicts for key, value in d.items()]
        messages = [{"role": i.role.value, "content": i.content} for i in messages]
        print("messages: ", messages)
        self._client._default_params["messages"] = messages
        print("self._client.default_params: ", self._client._default_params)
        response = self._client.chat_completion.with_fallbacks(base_providers)
        self._portkey_response = response.json()
        print(self._portkey_response, response)
        message = self._portkey_response["choices"][0]["message"]
        self._llm = self._get_fallback_llm
        return ChatResponse(message=message, raw=self._portkey_response)

    def _fallback(self, prompt) -> CompletionResponse:
        base_providers = [ProviderBase(**i) for i in self._llms]
        response = self._client.completion.with_fallbacks(base_providers)
        self._portkey_response = response.json()
        text = self._portkey_response["choices"][0]["text"]
        self._llm = self._get_fallback_llm
        return CompletionResponse(text=text, raw=self._portkey_response)

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self._mode == RubeusModes.FALLBACK:
            return self._fallback_chat(messages)
        return self._client.chat_completion.create(messages=messages, **kwargs)

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if self._mode == RubeusModes.FALLBACK:
            return self._fallback(prompt)

        return self._client.completion.create(prompt=prompt, **kwargs)

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        pass

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        pass

    @property
    def _is_chat_model(self) -> bool:
        """Check if a given model is a chat-based language model.

        Returns:
            bool: True if the provided model is a chat-based language model, False otherwise.
        """
        return is_chat_model(self._model)

    @property
    def _is_fallback_mode(self) -> bool:
        """Check if the suggested mode is fallback or not.

        Returns:
            bool: True if the provided mode is fallback type, False otherwise.
        """
        return self._mode == RubeusModes.FALLBACK

    @property
    def _get_fallback_llm(self) -> LLM:
        return get_fallback_llm(self._portkey_response, self._llms)
