from typing import Any, Dict, Optional, Sequence, List

from aleph_alpha_client import Prompt, CompletionRequest, Client, AsyncClient
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatResponseGen,
    CompletionResponseAsyncGen,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.utils import Tokenizer

from llama_index.llms.alephalpha.utils import (
    alephalpha_modelname_to_contextsize,
    process_response,
)

DEFAULT_ALEPHALPHA_MODEL = "luminous-supreme-control"
DEFAULT_ALEPHALPHA_MAX_TOKENS = 128
DEFAULT_ALEPHALPHA_HOST = "https://api.aleph-alpha.com"


class AlephAlpha(LLM):
    """Aleph Alpha LLMs."""

    model: str = Field(
        default=DEFAULT_ALEPHALPHA_MODEL, description="The Aleph Alpha model to use."
    )
    token: str = Field(default=None, description="The Aleph Alpha API token.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_ALEPHALPHA_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    base_url: Optional[str] = Field(
        default=DEFAULT_ALEPHALPHA_HOST, description="The hostname of the API base_url."
    )
    timeout: Optional[float] = Field(
        default=None, description="The timeout to use in seconds.", ge=0
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", ge=0
    )
    hosting: Optional[str] = Field(default=None, description="The hosting to use.")
    nice: bool = Field(default=False, description="Whether to be nice to the API.")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Aleph Alpha API."
    )
    repetition_penalties_include_prompt: bool = Field(
        default=True,
        description="Whether presence penalty or frequency penalty are updated from the prompt",
    )
    repetition_penalties_include_completion: bool = Field(
        default=True,
        description="Whether presence penalty or frequency penalty are updated from the completion.",
    )
    sequence_penalty: float = Field(
        default=0.7,
        description="The sequence penalty to use. Increasing the sequence penalty reduces the likelihood of reproducing token sequences that already appear in the prompt",
        ge=0.0,
        le=1.0,
    )
    sequence_penalty_min_length: int = Field(
        default=3,
        description="Minimal number of tokens to be considered as sequence. Must be greater or equal 2.",
        ge=2,
    )
    stop_sequences: List[str] = Field(
        default=["\n\n"], description="The stop sequences to use."
    )
    log_probs: Optional[int] = Field(
        default=None,
        description="Number of top log probabilities to return for each token generated.",
        ge=0,
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling parameter controlling the cumulative probability threshold.",
        ge=0.0,
        le=1.0,
    )
    echo: Optional[bool] = Field(
        default=False, description="Echo the prompt in the completion."
    )
    penalty_exceptions: Optional[List[str]] = Field(
        default=None,
        description="List of strings that may be generated without penalty, regardless of other penalty settings.",
    )
    n: Optional[int] = Field(
        default=1,
        description="The number of completions to return. Useful for generating multiple alternatives.",
    )

    _client: Optional[Client] = PrivateAttr()
    _aclient: Optional[AsyncClient] = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_ALEPHALPHA_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_ALEPHALPHA_MAX_TOKENS,
        base_url: Optional[str] = DEFAULT_ALEPHALPHA_HOST,
        timeout: Optional[float] = None,
        max_retries: int = 10,
        token: Optional[str] = None,
        hosting: Optional[str] = None,
        nice: bool = False,
        verify_ssl: bool = True,
        log_probs: Optional[int] = None,
        top_p: Optional[float] = None,
        echo: Optional[bool] = False,
        penalty_exceptions: Optional[List[str]] = None,
        n: Optional[int] = 1,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            hosting=hosting,
            nice=nice,
            verify_ssl=verify_ssl,
        )

        self.token = get_from_param_or_env("aa_token", token, "AA_TOKEN", "")

        self.log_probs = log_probs
        self.top_p = top_p
        self.echo = echo
        self.penalty_exceptions = penalty_exceptions
        self.n = n

        self._client = None
        self._aclient = None

    @classmethod
    def class_name(cls) -> str:
        return "AlephAlpha"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=alephalpha_modelname_to_contextsize(self.model),
            num_output=self.max_tokens,
            is_chat_model=False,  # The Aleph Alpha API does not support chat yet
            model_name=self.model,
        )

    @property
    def tokenizer(self) -> Tokenizer:
        client = self._get_client()
        return client.tokenizer(model=self.model)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "maximum_tokens": self.max_tokens,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    @property
    def _completion_kwargs(self) -> Dict[str, Any]:
        completion_kwargs = {
            "maximum_tokens": self.max_tokens,
            "temperature": self.temperature,
            "log_probs": self.log_probs,
            "top_p": self.top_p,
            "echo": self.echo,
            "penalty_exceptions": self.penalty_exceptions,
            "n": self.n,
            "repetition_penalties_include_prompt": self.repetition_penalties_include_prompt,
            "repetition_penalties_include_completion": self.repetition_penalties_include_completion,
            "sequence_penalty": self.sequence_penalty,
            "sequence_penalty_min_length": self.sequence_penalty_min_length,
            "stop_sequences": self.stop_sequences,
        }

        return {k: v for k, v in completion_kwargs.items() if v is not None}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def _get_credential_kwargs(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "host": self.base_url,
            "hosting": self.hosting,
            "request_timeout_seconds": self.timeout,
            "total_retries": self.max_retries,
            "nice": self.nice,
            "verify_ssl": self.verify_ssl,
        }

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(**self._get_credential_kwargs())
        return self._client

    def _get_aclient(self) -> AsyncClient:
        if self._aclient is None:
            self._aclient = AsyncClient(**self._get_credential_kwargs())
        return self._aclient

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError("Aleph Alpha does not currently support chat.")

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        client = self._get_client()
        all_kwargs = {
            "prompt": Prompt.from_text(prompt),
            **self._completion_kwargs,
            **kwargs,
        }
        request = CompletionRequest(**all_kwargs)
        response = client.complete(request=request, model=self.model)
        completion = response.completions[0].completion if response.completions else ""
        return process_response(response, completion)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        client = self._get_aclient()
        all_kwargs = {
            "prompt": Prompt.from_text(prompt),
            **self._completion_kwargs,
            **kwargs,
        }
        request = CompletionRequest(**all_kwargs)
        async with client as aclient:
            response = await aclient.complete(request=request, model=self.model)
            completion = (
                response.completions[0].completion if response.completions else ""
            )
            return process_response(response, completion)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError("Aleph Alpha does not currently support streaming.")

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError("Aleph Alpha does not currently support chat.")

    def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError("Aleph Alpha does not currently support chat.")

    def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError("Aleph Alpha does not currently support chat.")

    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError("Aleph Alpha does not currently support streaming.")
