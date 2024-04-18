"""NVIDIA embeddings file."""

from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

from openai import OpenAI, AsyncOpenAI

BASE_RETRIEVAL_PLAYGROUND_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia"


class NVIDIAEmbedding(BaseEmbedding):
    """NVIDIA embeddings."""

    model: str = Field(
        default="NV-Embed-QA",
        description="Name of the NVIDIA embedding model to use.\n"
        "Defaults to 'NV-Embed-QA'.",
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

    def __init__(
        self,
        model: str = "NV-Embed-QA",
        timeout: float = 120,
        max_retries: int = 5,
        api_key: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        api_key = get_from_param_or_env("api_key", api_key, "NVIDIA_API_KEY", "")

        self._client = OpenAI(
            api_key=api_key,
            base_url=BASE_RETRIEVAL_PLAYGROUND_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._client._custom_headers = {"User-Agent": "llama-index-embeddings-nvidia"}

        self._aclient = AsyncOpenAI(
            api_key=api_key,
            base_url=BASE_RETRIEVAL_PLAYGROUND_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._aclient._custom_headers = {"User-Agent": "llama-index-embeddings-nvidia"}

        super().__init__(
            model_name=model,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIAEmbedding"

    def mode(
        mode: Optional[("catalog", "nim")] = "catalog",
        base_url: Optional[str] = None,
        model: Optional[str] = "NV-Embed-QA",
        api_key: Optional[str] = None,
    ):
        out = self

        if mode == "catalog":
            if api_key is None:
                api_key = get_from_param_or_env(
                    "api_key", api_key, "NVIDIA_API_KEY", ""
                )

            if not api_key:
                raise ValueError(
                    "The NVIDIA API key must be provided as an environment variable or as a parameter to use the NVIDIA AI catalog."
                )

            out.model_name = model

            out._client.base_url = BASE_RETRIEVAL_PLAYGROUND_URL
            out._aclient.base_url = BASE_RETRIEVAL_PLAYGROUND_URL

            out._client.api_key = api_key
            out._aclient.api_key = api_key

        elif mode == "nim":
            if base_url is None:
                raise ValueError(
                    "The NIM base URL must be provided to connect to a local NIM"
                )

            out.model_name = model

            out._client.base_url = base_url
            out._aclient.base_url = base_url

        return out

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return (
            self._client.embeddings.create(
                input=[query], model=self.model, extra_body={"input_type": "query"}
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
                extra_body={"input_type": "passage"},
            )
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        assert len(texts) <= 259, "The batch size should not be larger than 299."

        data = self._client.embeddings.create(
            input=texts, model=self.model, extra_body={"input_type": "passage"}
        ).data
        return [d.embedding for d in data]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronously get query embedding."""
        return (
            (
                await self._aclient.embeddings.create(
                    input=[query],
                    model=self.model,
                    extra_body={"input_type": "query"},
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
                    extra_body={"input_type": "passage"},
                )
            )
            .data[0]
            .embedding
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        assert len(texts) <= 259, "The batch size should not be larger than 299."

        data = (
            await self._aclient.embeddings.create(
                input=texts, model=self.model, extra_body={"input_type": "passage"}
            )
        ).data
        return [d.embedding for d in data]
