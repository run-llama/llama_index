"""NVIDIA embeddings file."""

from typing import Any, List, Literal, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr, BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

from openai import OpenAI, AsyncOpenAI

BASE_RETRIEVAL_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia"
DEFAULT_MODEL = "NV-Embed-QA"


class Model(BaseModel):
    id: str


class NVIDIAEmbedding(BaseEmbedding):
    """NVIDIA embeddings."""

    model: str = Field(
        default=DEFAULT_MODEL,
        description="Name of the NVIDIA embedding model to use.\n"
        "Defaults to 'NV-Embed-QA'.",
    )

    truncate: Literal["NONE", "START", "END"] = Field(
        default="NONE",
        description=(
            "Truncate input text if it exceeds the model's maximum token length. "
            "Default is 'NONE', which raises an error if an input is too long."
        ),
    )

    timeout: float = Field(
        default=120, description="The timeout for the API request in seconds.", gte=0
    )

    max_retries: int = Field(
        default=5,
        description="The maximum number of retries for the API request.",
        gte=0,
    )

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()
    _mode: str = PrivateAttr("nvidia")

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: Optional[float] = 120,
        max_retries: Optional[int] = 5,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,  # This could default to 50
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        if embed_batch_size > 259:
            raise ValueError("The batch size should not be larger than 259.")

        api_key = get_from_param_or_env(
            "api_key", nvidia_api_key or api_key, "NVIDIA_API_KEY", "none"
        )

        self._client = OpenAI(
            api_key=api_key,
            base_url=BASE_RETRIEVAL_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._client._custom_headers = {"User-Agent": "llama-index-embeddings-nvidia"}

        self._aclient = AsyncOpenAI(
            api_key=api_key,
            base_url=BASE_RETRIEVAL_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._aclient._custom_headers = {"User-Agent": "llama-index-embeddings-nvidia"}

        super().__init__(
            model=model,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    @property
    def available_models(self) -> List[Model]:
        """Get available models."""
        ids = [DEFAULT_MODEL]
        if self._mode == "nim":
            ids = [model.id for model in self._client.models.list()]
        return [Model(id=id) for id in ids]

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIAEmbedding"

    def mode(
        self,
        mode: Optional[Literal["nvidia", "nim"]] = "nvidia",
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "NVIDIAEmbedding":
        if mode == "nim":
            if not base_url:
                raise ValueError("base_url is required for nim mode")
        if not base_url:
            base_url = BASE_RETRIEVAL_URL

        self._mode = mode
        if base_url:
            self._client.base_url = base_url
            self._aclient.base_url = base_url
        if model:
            self.model = model
            self._client.model = model
            self._aclient.model = model
        if api_key:
            self._client.api_key = api_key
            self._aclient.api_key = api_key

        return self

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return (
            self._client.embeddings.create(
                input=[query],
                model=self.model,
                extra_body={"input_type": "query", "truncate": self.truncate},
            )
            .data[0]
            .embedding
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return (
            self._client.embeddings.create(
                input=[text],
                model=self.model,
                extra_body={"input_type": "passage", "truncate": self.truncate},
            )
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        assert len(texts) <= 259, "The batch size should not be larger than 259."

        data = self._client.embeddings.create(
            input=texts,
            model=self.model,
            extra_body={"input_type": "passage", "truncate": self.truncate},
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
