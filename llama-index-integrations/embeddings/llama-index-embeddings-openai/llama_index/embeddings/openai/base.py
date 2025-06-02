"""OpenAI embeddings file."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.openai.utils import (
    DEFAULT_OPENAI_API_BASE,
    DEFAULT_OPENAI_API_VERSION,
    create_retry_decorator,
    resolve_openai_credentials,
)

from openai import AsyncOpenAI, OpenAI


class OpenAIEmbeddingMode(str, Enum):
    """OpenAI embedding mode."""

    SIMILARITY_MODE = "similarity"
    TEXT_SEARCH_MODE = "text_search"


class OpenAIEmbeddingModelType(str, Enum):
    """OpenAI embedding model type."""

    DAVINCI = "davinci"
    CURIE = "curie"
    BABBAGE = "babbage"
    ADA = "ada"
    TEXT_EMBED_ADA_002 = "text-embedding-ada-002"
    TEXT_EMBED_3_LARGE = "text-embedding-3-large"
    TEXT_EMBED_3_SMALL = "text-embedding-3-small"


class OpenAIEmbeddingModeModel(str, Enum):
    """OpenAI embedding mode model."""

    # davinci
    TEXT_SIMILARITY_DAVINCI = "text-similarity-davinci-001"
    TEXT_SEARCH_DAVINCI_QUERY = "text-search-davinci-query-001"
    TEXT_SEARCH_DAVINCI_DOC = "text-search-davinci-doc-001"

    # curie
    TEXT_SIMILARITY_CURIE = "text-similarity-curie-001"
    TEXT_SEARCH_CURIE_QUERY = "text-search-curie-query-001"
    TEXT_SEARCH_CURIE_DOC = "text-search-curie-doc-001"

    # babbage
    TEXT_SIMILARITY_BABBAGE = "text-similarity-babbage-001"
    TEXT_SEARCH_BABBAGE_QUERY = "text-search-babbage-query-001"
    TEXT_SEARCH_BABBAGE_DOC = "text-search-babbage-doc-001"

    # ada
    TEXT_SIMILARITY_ADA = "text-similarity-ada-001"
    TEXT_SEARCH_ADA_QUERY = "text-search-ada-query-001"
    TEXT_SEARCH_ADA_DOC = "text-search-ada-doc-001"

    # text-embedding-ada-002
    TEXT_EMBED_ADA_002 = "text-embedding-ada-002"

    # text-embedding-3-large
    TEXT_EMBED_3_LARGE = "text-embedding-3-large"

    # text-embedding-3-small
    TEXT_EMBED_3_SMALL = "text-embedding-3-small"


# convenient shorthand
OAEM = OpenAIEmbeddingMode
OAEMT = OpenAIEmbeddingModelType
OAEMM = OpenAIEmbeddingModeModel

EMBED_MAX_TOKEN_LIMIT = 2048


_QUERY_MODE_MODEL_DICT = {
    (OAEM.SIMILARITY_MODE, "davinci"): OAEMM.TEXT_SIMILARITY_DAVINCI,
    (OAEM.SIMILARITY_MODE, "curie"): OAEMM.TEXT_SIMILARITY_CURIE,
    (OAEM.SIMILARITY_MODE, "babbage"): OAEMM.TEXT_SIMILARITY_BABBAGE,
    (OAEM.SIMILARITY_MODE, "ada"): OAEMM.TEXT_SIMILARITY_ADA,
    (OAEM.SIMILARITY_MODE, "text-embedding-ada-002"): OAEMM.TEXT_EMBED_ADA_002,
    (OAEM.SIMILARITY_MODE, "text-embedding-3-small"): OAEMM.TEXT_EMBED_3_SMALL,
    (OAEM.SIMILARITY_MODE, "text-embedding-3-large"): OAEMM.TEXT_EMBED_3_LARGE,
    (OAEM.TEXT_SEARCH_MODE, "davinci"): OAEMM.TEXT_SEARCH_DAVINCI_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "curie"): OAEMM.TEXT_SEARCH_CURIE_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "babbage"): OAEMM.TEXT_SEARCH_BABBAGE_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "ada"): OAEMM.TEXT_SEARCH_ADA_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "text-embedding-ada-002"): OAEMM.TEXT_EMBED_ADA_002,
    (OAEM.TEXT_SEARCH_MODE, "text-embedding-3-large"): OAEMM.TEXT_EMBED_3_LARGE,
    (OAEM.TEXT_SEARCH_MODE, "text-embedding-3-small"): OAEMM.TEXT_EMBED_3_SMALL,
}

_TEXT_MODE_MODEL_DICT = {
    (OAEM.SIMILARITY_MODE, "davinci"): OAEMM.TEXT_SIMILARITY_DAVINCI,
    (OAEM.SIMILARITY_MODE, "curie"): OAEMM.TEXT_SIMILARITY_CURIE,
    (OAEM.SIMILARITY_MODE, "babbage"): OAEMM.TEXT_SIMILARITY_BABBAGE,
    (OAEM.SIMILARITY_MODE, "ada"): OAEMM.TEXT_SIMILARITY_ADA,
    (OAEM.SIMILARITY_MODE, "text-embedding-ada-002"): OAEMM.TEXT_EMBED_ADA_002,
    (OAEM.SIMILARITY_MODE, "text-embedding-3-small"): OAEMM.TEXT_EMBED_3_SMALL,
    (OAEM.SIMILARITY_MODE, "text-embedding-3-large"): OAEMM.TEXT_EMBED_3_LARGE,
    (OAEM.TEXT_SEARCH_MODE, "davinci"): OAEMM.TEXT_SEARCH_DAVINCI_DOC,
    (OAEM.TEXT_SEARCH_MODE, "curie"): OAEMM.TEXT_SEARCH_CURIE_DOC,
    (OAEM.TEXT_SEARCH_MODE, "babbage"): OAEMM.TEXT_SEARCH_BABBAGE_DOC,
    (OAEM.TEXT_SEARCH_MODE, "ada"): OAEMM.TEXT_SEARCH_ADA_DOC,
    (OAEM.TEXT_SEARCH_MODE, "text-embedding-ada-002"): OAEMM.TEXT_EMBED_ADA_002,
    (OAEM.TEXT_SEARCH_MODE, "text-embedding-3-large"): OAEMM.TEXT_EMBED_3_LARGE,
    (OAEM.TEXT_SEARCH_MODE, "text-embedding-3-small"): OAEMM.TEXT_EMBED_3_SMALL,
}


def get_embedding(client: OpenAI, text: str, engine: str, **kwargs: Any) -> List[float]:
    """
    Get embedding.

    NOTE: Copied from OpenAI's embedding utils:
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies
    like matplotlib, plotly, scipy, sklearn.

    """
    text = text.replace("\n", " ")

    return (
        client.embeddings.create(input=[text], model=engine, **kwargs).data[0].embedding
    )


async def aget_embedding(
    aclient: AsyncOpenAI, text: str, engine: str, **kwargs: Any
) -> List[float]:
    """
    Asynchronously get embedding.

    NOTE: Copied from OpenAI's embedding utils:
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies
    like matplotlib, plotly, scipy, sklearn.

    """
    text = text.replace("\n", " ")

    return (
        (await aclient.embeddings.create(input=[text], model=engine, **kwargs))
        .data[0]
        .embedding
    )


def get_embeddings(
    client: OpenAI, list_of_text: List[str], engine: str, **kwargs: Any
) -> List[List[float]]:
    """
    Get embeddings.

    NOTE: Copied from OpenAI's embedding utils:
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies
    like matplotlib, plotly, scipy, sklearn.

    """
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = client.embeddings.create(input=list_of_text, model=engine, **kwargs).data
    return [d.embedding for d in data]


async def aget_embeddings(
    aclient: AsyncOpenAI,
    list_of_text: List[str],
    engine: str,
    **kwargs: Any,
) -> List[List[float]]:
    """
    Asynchronously get embeddings.

    NOTE: Copied from OpenAI's embedding utils:
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies
    like matplotlib, plotly, scipy, sklearn.

    """
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = (
        await aclient.embeddings.create(input=list_of_text, model=engine, **kwargs)
    ).data
    return [d.embedding for d in data]


def get_engine(
    mode: str,
    model: str,
    mode_model_dict: Dict[Tuple[OpenAIEmbeddingMode, str], OpenAIEmbeddingModeModel],
) -> str:
    """Get engine."""
    key = (OpenAIEmbeddingMode(mode), OpenAIEmbeddingModelType(model))
    if key not in mode_model_dict:
        raise ValueError(f"Invalid mode, model combination: {key}")
    return mode_model_dict[key].value


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI class for embeddings.

    Args:
        mode (str): Mode for embedding.
            Defaults to OpenAIEmbeddingMode.TEXT_SEARCH_MODE.
            Options are:

            - OpenAIEmbeddingMode.SIMILARITY_MODE
            - OpenAIEmbeddingMode.TEXT_SEARCH_MODE

        model (str): Model for embedding.
            Defaults to OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002.
            Options are:

            - OpenAIEmbeddingModelType.DAVINCI
            - OpenAIEmbeddingModelType.CURIE
            - OpenAIEmbeddingModelType.BABBAGE
            - OpenAIEmbeddingModelType.ADA
            - OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002

    """

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )

    api_key: str = Field(description="The OpenAI API key.")
    api_base: Optional[str] = Field(
        default=DEFAULT_OPENAI_API_BASE, description="The base URL for OpenAI API."
    )
    api_version: Optional[str] = Field(
        default=DEFAULT_OPENAI_API_VERSION, description="The version for OpenAI API."
    )

    max_retries: int = Field(default=10, description="Maximum number of retries.", ge=0)
    timeout: float = Field(default=60.0, description="Timeout for each request.", ge=0)
    default_headers: Optional[Dict[str, str]] = Field(
        default=None, description="The default headers for API requests."
    )
    reuse_client: bool = Field(
        default=True,
        description=(
            "Reuse the OpenAI client between requests. When doing anything with large "
            "volumes of async API calls, setting this to false can improve stability."
        ),
    )
    dimensions: Optional[int] = Field(
        default=None,
        description=(
            "The number of dimensions on the output embedding vectors. "
            "Works only with v3 embedding models."
        ),
    )

    _query_engine: str = PrivateAttr()
    _text_engine: str = PrivateAttr()
    _client: Optional[OpenAI] = PrivateAttr()
    _aclient: Optional[AsyncOpenAI] = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()
    _async_http_client: Optional[httpx.AsyncClient] = PrivateAttr()

    def __init__(
        self,
        mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
        model: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
        embed_batch_size: int = 100,
        dimensions: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        if dimensions is not None:
            additional_kwargs["dimensions"] = dimensions

        api_key, api_base, api_version = self._resolve_credentials(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )

        query_engine = get_engine(mode, model, _QUERY_MODE_MODEL_DICT)
        text_engine = get_engine(mode, model, _TEXT_MODE_MODEL_DICT)

        if "model_name" in kwargs:
            model_name = kwargs.pop("model_name")
            query_engine = text_engine = model_name
        else:
            model_name = model

        super().__init__(
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            callback_manager=callback_manager,
            model_name=model_name,
            additional_kwargs=additional_kwargs,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            max_retries=max_retries,
            reuse_client=reuse_client,
            timeout=timeout,
            default_headers=default_headers,
            num_workers=num_workers,
            **kwargs,
        )
        self._query_engine = query_engine
        self._text_engine = text_engine

        self._client = None
        self._aclient = None
        self._http_client = http_client
        self._async_http_client = async_http_client

    def _resolve_credentials(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> Tuple[Optional[str], str, str]:
        return resolve_openai_credentials(api_key, api_base, api_version)

    def _get_client(self) -> OpenAI:
        if not self.reuse_client:
            return OpenAI(**self._get_credential_kwargs())

        if self._client is None:
            self._client = OpenAI(**self._get_credential_kwargs())
        return self._client

    def _get_aclient(self) -> AsyncOpenAI:
        if not self.reuse_client:
            return AsyncOpenAI(**self._get_credential_kwargs(is_async=True))

        if self._aclient is None:
            self._aclient = AsyncOpenAI(**self._get_credential_kwargs(is_async=True))
        return self._aclient

    def _create_retry_decorator(self):
        """Create a retry decorator using the instance's max_retries."""
        return create_retry_decorator(
            max_retries=self.max_retries,
            random_exponential=True,
            stop_after_delay_seconds=60,
            min_seconds=1,
            max_seconds=20,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OpenAIEmbedding"

    def _get_credential_kwargs(self, is_async: bool = False) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "default_headers": self.default_headers,
            "http_client": self._async_http_client if is_async else self._http_client,
        }

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        client = self._get_client()
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _retryable_get_embedding():
            return get_embedding(
                client,
                query,
                engine=self._query_engine,
                **self.additional_kwargs,
            )

        return _retryable_get_embedding()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        aclient = self._get_aclient()
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        async def _retryable_aget_embedding():
            return await aget_embedding(
                aclient,
                query,
                engine=self._query_engine,
                **self.additional_kwargs,
            )

        return await _retryable_aget_embedding()

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        client = self._get_client()
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _retryable_get_embedding():
            return get_embedding(
                client,
                text,
                engine=self._text_engine,
                **self.additional_kwargs,
            )

        return _retryable_get_embedding()

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        aclient = self._get_aclient()
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        async def _retryable_aget_embedding():
            return await aget_embedding(
                aclient,
                text,
                engine=self._text_engine,
                **self.additional_kwargs,
            )

        return await _retryable_aget_embedding()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get text embeddings.

        By default, this is a wrapper around _get_text_embedding.
        Can be overridden for batch queries.

        """
        client = self._get_client()
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _retryable_get_embeddings():
            return get_embeddings(
                client,
                texts,
                engine=self._text_engine,
                **self.additional_kwargs,
            )

        return _retryable_get_embeddings()

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        aclient = self._get_aclient()
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        async def _retryable_aget_embeddings():
            return await aget_embeddings(
                aclient,
                texts,
                engine=self._text_engine,
                **self.additional_kwargs,
            )

        return await _retryable_aget_embeddings()
