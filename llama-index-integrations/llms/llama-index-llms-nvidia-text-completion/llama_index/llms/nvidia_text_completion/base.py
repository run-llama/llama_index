from typing import Any, Dict, List, Optional
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)
import warnings
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.nvidia_text_completion.utils import (
    Model,
    COMPLETION_MODEL_TABLE,
    completions_arguments,
)
from urllib.parse import urlparse
from llama_index.core.base.llms.types import (
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
)

DEFAULT_MODEL = "bigcode/starcoder2-7b"
BASE_URL = "https://integrate.api.nvidia.com/v1"


class NVIDIATextCompletion(OpenAILike):
    _is_hosted: bool = PrivateAttr(True)
    _mode: str = PrivateAttr(default="nvidia")

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = BASE_URL,
        max_tokens: Optional[int] = 1024,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an instance of the NVIDIATextCompletion class.

        This class provides an interface to the NVIDIA NIM /completion endpoints.
        By default, it connects to a hosted NIM,
        but you can switch to an on-premises NIM by providing a `base_url`.

        Args:
            model (str, optional): The model to use for the NIM.
            nvidia_api_key (str, optional): The API key for the NVIDIA NIM. Defaults to None.
            api_key (str, optional): An alternative parameter for providing the API key. Defaults to None.
            base_url (str, optional): The base URL for the NIM. Use this to switch to an on-premises NIM.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            **kwargs: Additional keyword arguments.

        API Keys:
        - The recommended way to provide the API key is through the `NVIDIA_API_KEY` environment variable.

        Raises:
            DeprecationWarning: If an API key is not provided for a hosted NIM, a warning is issued. This will become an error in version 0.2.0.
        """
        api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        is_hosted = base_url == BASE_URL
        if not is_hosted:
            base_url = self._validate_url(base_url)

        if is_hosted and api_key == "NO_API_KEY_PROVIDED":
            warnings.warn(
                "An API key is required for the hosted NIM. This will become an error in 0.2.0.",
            )
        # kwargs = self.__check_kwargs(kwargs)

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=base_url,
            max_tokens=max_tokens,
            default_headers={"User-Agent": "llama-index-llms-nvidia"},
            **self.__check_kwargs(kwargs),
        )

    def _validate_url(self, base_url):
        """
        Base URL Validation.
        ValueError : url which do not have valid scheme and netloc.
        Warning : v1/completions routes.
        ValueError : Any other routes other than above.
        """
        expected_format = "Expected format is 'http://host:port'."
        result = urlparse(base_url)
        if not (result.scheme and result.netloc):
            raise ValueError(
                f"Invalid base_url, Expected format is 'http://host:port': {base_url}"
            )
        if base_url.endswith("v1/completions"):
            warnings.warn(f"{expected_format} Rest is Ignored.")
        return base_url.strip("/")

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIATextCompletion"

    @property
    def available_models(self) -> List[Model]:
        models = self._get_client().models.list().data
        if self._is_hosted:
            # only exclude models in hosted mode. in non-hosted mode, the administrator has control
            # over the model name and may deploy an excluded name that will work.
            models = list(COMPLETION_MODEL_TABLE.values())
        return models

    def __check_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check kwargs, warn for unknown keys, and return a copy recognized keys.
        """
        recognized_kwargs = {
            k: v for k, v in kwargs.items() if k in completions_arguments
        }
        unrecognized_kwargs = set(kwargs) - completions_arguments
        if len(unrecognized_kwargs) > 0:
            warnings.warn(f"Unrecognized, ignored arguments: {unrecognized_kwargs}")

        return recognized_kwargs

    async def achat(self, question: str) -> ChatResponse:
        raise NotImplementedError(
            "NVIDIATextCompletion does not currently support chat."
        )

    def chat(self, question: str) -> ChatResponse:
        raise NotImplementedError(
            "NVIDIATextCompletion does not currently support chat."
        )

    async def astream_complete(self, question: str) -> CompletionResponseAsyncGen:
        raise NotImplementedError(
            "NVIDIATextCompletion does not currently support streaming."
        )

    async def astream_chat(self, question: str) -> ChatResponseAsyncGen:
        raise NotImplementedError(
            "NVIDIATextCompletion does not currently support streaming."
        )

    def stream_chat(self, question: str) -> ChatResponseGen:
        raise NotImplementedError(
            "NVIDIATextCompletion does not currently support chat."
        )

    async def stream(self, question: str) -> ChatResponseAsyncGen:
        raise NotImplementedError(
            "NVIDIATextCompletion does not currently support streaming."
        )
