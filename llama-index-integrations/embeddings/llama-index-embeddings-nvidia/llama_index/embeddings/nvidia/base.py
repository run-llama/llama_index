"""NVIDIA embeddings file."""

from typing import Any, List, Literal, Optional
import warnings
import os

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

from openai import OpenAI, AsyncOpenAI
from urllib.parse import urlparse, urlunparse
from .utils import (
    EMBEDDING_MODEL_TABLE,
    BASE_URL,
    KNOWN_URLS,
    DEFAULT_MODEL,
    Model,
    determine_model,
)


class NVIDIAEmbedding(BaseEmbedding):
    """NVIDIA embeddings."""

    base_url: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_BASE_URL", BASE_URL),
        description="Base url for model listing an invocation",
    )
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

        self._is_hosted = self.base_url in KNOWN_URLS
        if not self._is_hosted:
            self.base_url = self._validate_url(self.base_url)

        if self._is_hosted:  # hosted on API Catalog (build.nvidia.com)
            if api_key == "NO_API_KEY_PROVIDED":
                raise ValueError("An API key is required for hosted NIM.")
        else:  # not hosted
            self.base_url = self._validate_url(self.base_url)

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._client._custom_headers = {"User-Agent": "llama-index-embeddings-nvidia"}

        self._aclient = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
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
        validate the base_url.
        if the base_url is not a url, raise an error
        if the base_url does not end in /v1, e.g. /embeddings
        emit a warning. old documentation told users to pass in the full
        inference url, which is incorrect and prevents model listing from working.
        normalize base_url to end in /v1.
        """
        if base_url is not None:
            parsed = urlparse(base_url)

            # Ensure scheme and netloc (domain name) are present
            if not (parsed.scheme and parsed.netloc):
                expected_format = "Expected format is: http://host:port"
                raise ValueError(
                    f"Invalid base_url format. {expected_format} Got: {base_url}"
                )

            normalized_path = parsed.path.rstrip("/")
            if not normalized_path.endswith("/v1"):
                warnings.warn(
                    f"{base_url} does not end in /v1, you may "
                    "have inference and listing issues"
                )
                normalized_path += "/v1"

                base_url = urlunparse(
                    (parsed.scheme, parsed.netloc, normalized_path, None, None, None)
                )
        return base_url

    def _validate_model(self, model_name: str) -> None:
        """
        Validates compatibility of the hosted model with the client.
        Skipping the client validation for non-catalogue requests.

        Args:
            model_name (str): The name of the model.

        Raises:
            ValueError: If the model is incompatible with the client.
        """
        model = determine_model(model_name)
        if self._is_hosted:
            if not model:
                warnings.warn(f"Unable to determine validity of {model_name}")
            if model and model.endpoint:
                self.base_url = model.endpoint
        # TODO: handle locally hosted models

    @property
    def available_models(self) -> List[str]:
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
            return [Model(id=id) for id in EMBEDDING_MODEL_TABLE]

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
