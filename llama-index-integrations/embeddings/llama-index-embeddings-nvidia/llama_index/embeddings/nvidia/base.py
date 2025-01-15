"""NVIDIA embeddings file."""

from typing import Any, List, Literal, Optional
import warnings

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr, BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

from openai import OpenAI, AsyncOpenAI
from urllib.parse import urlparse

# integrate.api.nvidia.com is the default url for most models, any
# bespoke endpoints will need to be added to the MODEL_ENDPOINT_MAP
BASE_URL = "https://integrate.api.nvidia.com/v1/"
DEFAULT_MODEL = "nvidia/nv-embedqa-e5-v5"

# because MODEL_ENDPOINT_MAP is used to construct KNOWN_URLS, we need to
# include at least one model w/ https://integrate.api.nvidia.com/v1/
MODEL_ENDPOINT_MAP = {
    "NV-Embed-QA": "https://ai.api.nvidia.com/v1/retrieval/nvidia/",
    "snowflake/arctic-embed-l": "https://integrate.api.nvidia.com/v1/",
    "nvidia/nv-embed-v1": "https://integrate.api.nvidia.com/v1/",
    "nvidia/nv-embedqa-mistral-7b-v2": "https://integrate.api.nvidia.com/v1/",
    "nvidia/nv-embedqa-e5-v5": "https://integrate.api.nvidia.com/v1/",
    "baai/bge-m3": "https://integrate.api.nvidia.com/v1/",
    "nvidia/llama-3.2-nv-embedqa-1b-v1": "https://integrate.api.nvidia.com/v1/",
    "nvidia/llama-3.2-nv-embedqa-1b-v2": "https://integrate.api.nvidia.com/v1/",
}

KNOWN_URLS = list(MODEL_ENDPOINT_MAP.values())
KNOWN_URLS.append("https://ai.api.nvidia.com/v1/retrieval/snowflake/arctic-embed-l")


class Model(BaseModel):
    id: str
    base_model: Optional[str] = None


class NVIDIAEmbedding(BaseEmbedding):
    """NVIDIA embeddings."""

    model: Optional[str] = Field(
        description="Name of the NVIDIA embedding model to use.\n"
    )

    truncate: Literal["NONE", "START", "END"] = Field(
        default="NONE",
        description=(
            "Truncate input text if it exceeds the model's maximum token length. "
            "Default is 'NONE', which raises an error if an input is too long."
        ),
    )

    timeout: float = Field(
        default=120, description="The timeout for the API request in seconds.", ge=0
    )

    max_retries: int = Field(
        default=5,
        description="The maximum number of retries for the API request.",
        ge=0,
    )

    dimensions: Optional[int] = Field(
        default=None,
        description=(
            "The number of dimensions for the embeddings. This parameter is not "
            "supported by all models."
        ),
    )

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()
    _is_hosted: bool = PrivateAttr(True)

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: Optional[float] = 120,
        max_retries: Optional[int] = 5,
        dimensions: Optional[int] = 0,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,  # This could default to 50
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        """
        Construct an Embedding interface for NVIDIA NIM.

        This constructor initializes an instance of the NVIDIAEmbedding class, which provides
        an interface for embedding text using NVIDIA's NIM service.

        Parameters:
        - model (str, optional): The name of the model to use for embeddings.
        - timeout (float, optional): The timeout for requests to the NIM service, in seconds. Defaults to 120.
        - max_retries (int, optional): The maximum number of retries for requests to the NIM service. Defaults to 5.
        - dimensions (int, optional): The number of dimensions for the embeddings. This
                              parameter is not supported by all models.
        - nvidia_api_key (str, optional): The API key for the NIM service. This is required if using a hosted NIM.
        - api_key (str, optional): An alternative parameter for providing the API key.
        - base_url (str, optional): The base URL for the NIM service. If not provided, the service will default to a hosted NIM.
        - **kwargs: Additional keyword arguments.

        API Keys:
        - The recommended way to provide the API key is through the `NVIDIA_API_KEY` environment variable.

        Note:
        - Switch from a hosted NIM (default) to an on-premises NIM using the `base_url` parameter. An API key is required for hosted NIM.
        """
        super().__init__(
            model=model,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            dimensions=dimensions,
            **kwargs,
        )
        self.dimensions = dimensions

        if embed_batch_size > 259:
            raise ValueError("The batch size should not be larger than 259.")

        api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        base_url = base_url or BASE_URL
        self._is_hosted = base_url in KNOWN_URLS
        if self._is_hosted:  # hosted on API Catalog (build.nvidia.com)
            if api_key == "NO_API_KEY_PROVIDED":
                raise ValueError("An API key is required for hosted NIM.")
            # TODO: we should not assume unknown models are at the base url
            base_url = MODEL_ENDPOINT_MAP.get(model, BASE_URL)
        else:  # not hosted
            base_url = self._validate_url(base_url)

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._client._custom_headers = {"User-Agent": "llama-index-embeddings-nvidia"}

        self._aclient = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._aclient._custom_headers = {"User-Agent": "llama-index-embeddings-nvidia"}

        self.model = model
        if not self.model:
            if self._is_hosted:
                self.model = DEFAULT_MODEL
            else:
                self.__get_default_model()

        if not self.model.startswith("nvdev/"):
            self._validate_model(self.model)  ## validate model

    def __get_default_model(self) -> None:
        """Set default model."""
        if not self._is_hosted:
            valid_models = [
                model.id
                for model in self.available_models
                if not model.base_model or model.base_model == model.id
            ]
            self.model = next(iter(valid_models), None)
            if self.model:
                warnings.warn(
                    f"Default model is set as: {self.model}. \n"
                    "Set model using model parameter. \n"
                    "To get available models use available_models property.",
                    UserWarning,
                )
            else:
                raise ValueError("No locally hosted model was found.")
        else:
            self.model = self.model or DEFAULT_MODEL

    def _validate_url(self, base_url):
        """
        Base URL Validation.
        ValueError : url which do not have valid scheme and netloc.
        Warning : v1/embeddings routes.
        ValueError : Any other routes other than above.
        """
        expected_format = "Expected format is 'http://host:port'."
        result = urlparse(base_url)
        if not (result.scheme and result.netloc):
            raise ValueError(f"Invalid base_url, {expected_format}")
        if base_url.endswith("embeddings"):
            warnings.warn(f"{expected_format} Rest is ignored")
        return base_url.strip("/")

    def _validate_model(self, model_name: str) -> None:
        """
        Validates compatibility of the hosted model with the client.

        Args:
            model_name (str): The name of the model.

        Raises:
            ValueError: If the model is incompatible with the client.
        """
        if self._is_hosted:
            if model_name not in MODEL_ENDPOINT_MAP:
                if model_name in [model.id for model in self._client.models.list()]:
                    warnings.warn(f"Unable to determine validity of {model_name}")
                else:
                    raise ValueError(
                        f"Model {model_name} is incompatible with client {self.class_name()}. "
                        f"Please check `{self.class_name()}.available_models()`."
                    )
        else:
            if model_name not in [model.id for model in self.available_models]:
                raise ValueError(f"No locally hosted {model_name} was found.")

    @property
    def available_models(self) -> List[Model]:
        """Get available models."""
        # TODO: hosted now has a model listing, need to merge known and listed models
        if not self._is_hosted:
            return [
                Model(
                    id=model.id,
                    base_model=getattr(model, "params", {}).get("root", None),
                )
                for model in self._client.models.list()
            ]
        else:
            return [Model(id=id) for id in MODEL_ENDPOINT_MAP]

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIAEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        extra_body = {"input_type": "passage", "truncate": self.truncate}
        if self.dimensions:
            extra_body["dimensions"] = self.dimensions
        return (
            self._client.embeddings.create(
                input=[query],
                model=self.model,
                extra_body=extra_body,
            )
            .data[0]
            .embedding
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        extra_body = {"input_type": "passage", "truncate": self.truncate}
        if self.dimensions:
            extra_body["dimensions"] = self.dimensions
        return (
            self._client.embeddings.create(
                input=[text],
                model=self.model,
                extra_body=extra_body,
            )
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        assert len(texts) <= 259, "The batch size should not be larger than 259."
        extra_body = {"input_type": "passage", "truncate": self.truncate}
        if self.dimensions:
            extra_body["dimensions"] = self.dimensions
        data = self._client.embeddings.create(
            input=texts,
            model=self.model,
            extra_body=extra_body,
        ).data
        return [d.embedding for d in data]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronously get query embedding."""
        return (
            (
                await self._aclient.embeddings.create(
                    input=[query],
                    model=self.model,
                    extra_body={"input_type": "query", "truncate": self.truncate},
                )
            )
            .data[0]
            .embedding
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return (
            (
                await self._aclient.embeddings.create(
                    input=[text],
                    model=self.model,
                    extra_body={"input_type": "passage", "truncate": self.truncate},
                )
            )
            .data[0]
            .embedding
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        assert len(texts) <= 259, "The batch size should not be larger than 259."

        data = (
            await self._aclient.embeddings.create(
                input=texts,
                model=self.model,
                extra_body={"input_type": "passage", "truncate": self.truncate},
            )
        ).data
        return [d.embedding for d in data]
