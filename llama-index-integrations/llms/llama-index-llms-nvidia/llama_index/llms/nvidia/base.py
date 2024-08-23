from typing import (
    Any,
    Optional,
    List,
    Literal,
)
from deprecated import deprecated
import warnings

from llama_index.core.bridge.pydantic import PrivateAttr, BaseModel
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)


from llama_index.llms.openai_like import OpenAILike
from urllib.parse import urlparse, urlunparse

DEFAULT_MODEL = "meta/llama3-8b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1/"

KNOWN_URLS = [
    BASE_URL,
    "https://integrate.api.nvidia.com/v1",
]


class Model(BaseModel):
    id: str


class NVIDIA(OpenAILike):
    """NVIDIA's API Catalog Connector."""

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
        Initialize an instance of the NVIDIA class.

        This class provides an interface to the NVIDIA NIM. By default, it connects to a hosted NIM,
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

        is_hosted = base_url in KNOWN_URLS
        if base_url not in KNOWN_URLS:
            base_url = self._validate_url(base_url)

        if is_hosted and api_key == "NO_API_KEY_PROVIDED":
            warnings.warn(
                "An API key is required for the hosted NIM. This will become an error in 0.2.0.",
            )

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=base_url,
            max_tokens=max_tokens,
            is_chat_model=True,
            default_headers={"User-Agent": "llama-index-llms-nvidia"},
            **kwargs,
        )
        self._is_hosted = is_hosted

    def _validate_url(self, base_url):
        """
        Base URL Validation.
        ValueError : url which do not have valid scheme and netloc.
        Warning : v1/chat/completions routes.
        ValueError : Any other routes other than above.
        """
        expected_format = "Expected format is 'http://host:port'."
        result = urlparse(base_url)
        if not (result.scheme and result.netloc):
            raise ValueError(
                f"Invalid base_url, Expected format is 'http://host:port': {base_url}"
            )
        if result.path:
            normalized_path = result.path.strip("/")
            if normalized_path == "v1":
                pass
            elif normalized_path == "v1/chat/completions":
                warnings.warn(f"{expected_format} Rest is Ignored.")
            else:
                raise ValueError(f"Base URL path is not recognized. {expected_format}")
        return urlunparse((result.scheme, result.netloc, "v1", "", "", ""))

    @property
    def available_models(self) -> List[Model]:
        models = self._get_client().models.list().data
        # only exclude models in hosted mode. in non-hosted mode, the administrator has control
        # over the model name and may deploy an excluded name that will work.
        if self._is_hosted:
            exclude = {
                "mistralai/mixtral-8x22b-v0.1",  # not a /chat/completion endpoint
            }
            models = [model for model in models if model.id not in exclude]
        return models

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIA"

    @deprecated(
        version="0.1.3",
        reason="Will be removed in 0.2. Construct with `base_url` instead.",
    )
    def mode(
        self,
        mode: Optional[Literal["nvidia", "nim"]] = "nvidia",
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "NVIDIA":
        """
        Deprecated: use NVIDIA(base_url="...") instead.
        """
        if mode == "nim":
            if not base_url:
                raise ValueError("base_url is required for nim mode")
        if mode == "nvidia":
            api_key = get_from_param_or_env(
                "api_key",
                api_key,
                "NVIDIA_API_KEY",
            )
            base_url = base_url or BASE_URL

        self._mode = mode
        if base_url:
            self.api_base = base_url
        if model:
            self.model = model
        if api_key:
            self.api_key = api_key

        return self
