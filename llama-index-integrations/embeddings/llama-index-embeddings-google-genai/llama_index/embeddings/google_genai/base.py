"""Gemini embeddings file."""

import os
from importlib.metadata import PackageNotFoundError, version
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    AsyncRetrying,
)
from typing import Any, Dict, List, Optional, TypedDict, Callable, TypeVar, Awaitable

import requests
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager

import google.genai
import google.auth.credentials
import google.genai.types as types
from google.genai.errors import APIError

# Define generic types for functions that will be wrapped with retry
T = TypeVar("T")
R = TypeVar("R")


class VertexAIConfig(TypedDict):
    credentials: Optional[google.auth.credentials.Credentials] = None
    project: Optional[str] = None
    location: Optional[str] = None


def is_retryable_error(exception: BaseException) -> bool:
    """
    Checks if the exception is a retryable error based on the criteria.
    """
    if isinstance(exception, APIError):
        return exception.code in [429, 502, 503, 504]
    if isinstance(exception, requests.exceptions.ConnectionError):  # noqa: SIM103
        return True
    return False


def get_retryable_function(
    func: Callable[..., T],
    max_retries: int = 3,
    min_seconds: float = 1,
    max_seconds: float = 10,
    exponential_base: float = 2,
) -> Callable[..., T]:
    """
    Wraps a function with tenacity retry decorator based on configurable parameters.

    Args:
        func: The function to wrap with retry logic
        max_retries: Maximum number of retry attempts
        min_seconds: Minimum wait time between retries
        max_seconds: Maximum wait time between retries
        exponential_base: Base for exponential backoff calculation

    Returns:
        The wrapped function with retry logic

    """
    retry_decorator = retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(
            multiplier=min_seconds, max=max_seconds, exp_base=exponential_base
        ),
        retry=retry_if_exception(is_retryable_error),
    )

    return retry_decorator(func)


async def get_retryable_async_function(
    func: Callable[..., Awaitable[R]],
    max_retries: int = 3,
    min_seconds: float = 1,
    max_seconds: float = 10,
    exponential_base: float = 2,
) -> R:
    """
    Wraps an async function with tenacity retry logic based on configurable parameters.

    Args:
        func: The async function to wrap with retry logic
        max_retries: Maximum number of retry attempts
        min_seconds: Minimum wait time between retries
        max_seconds: Maximum wait time between retries
        exponential_base: Base for exponential backoff calculation

    Returns:
        The result of the async function after applying retry logic

    """
    retry_config = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(
            multiplier=min_seconds, max=max_seconds, exp_base=exponential_base
        ),
        retry=retry_if_exception(is_retryable_error),
    )

    async for attempt in retry_config:
        with attempt:
            return await func()

    raise Exception("Failed to get retryable async function")


class GoogleGenAIEmbedding(BaseEmbedding):
    """
    Google GenAI embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "text-embedding-005".
        api_key (Optional[str]): API key to access the model. Defaults to None.
        embedding_config (Optional[types.EmbedContentConfigOrDict]): Embedding config to access the model. Defaults to None.
        vertexai_config (Optional[VertexAIConfig]): Vertex AI config to access the model. Defaults to None.
        http_options (Optional[types.HttpOptions]): HTTP options to access the model. Defaults to None.
        debug_config (Optional[google.genai.client.DebugConfig]): Debug config to access the model. Defaults to None.
        embed_batch_size (int): Batch size for embedding. Defaults to 100.
        callback_manager (Optional[CallbackManager]): Callback manager to access the model. Defaults to None.
        retries (int): Maximum number of retries for API calls. Defaults to 3.
        timeout (int): Timeout for API calls in seconds. Defaults to 10.
        retry_min_seconds (float): Minimum wait time between retries. Defaults to 1.
        retry_max_seconds (float): Maximum wait time between retries. Defaults to 10.
        retry_exponential_base (float): Base for exponential backoff calculation. Defaults to 2.

    Examples:
        `pip install llama-index-embeddings-google-genai`

        ```python
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        embed_model = GoogleGenAIEmbedding(model_name="text-embedding-005", api_key="...")
        ```

    """

    _client: google.genai.Client = PrivateAttr()
    _embedding_config: types.EmbedContentConfigOrDict = PrivateAttr()

    embedding_config: Optional[types.EmbedContentConfigOrDict] = Field(
        default=None, description="""Used to override embedding config."""
    )
    retries: int = Field(default=3, description="Number of retries for embedding.")
    timeout: int = Field(default=10, description="Timeout for embedding.")
    retry_min_seconds: float = Field(
        default=1, description="Minimum wait time between retries."
    )
    retry_max_seconds: float = Field(
        default=10, description="Maximum wait time between retries."
    )
    retry_exponential_base: float = Field(
        default=2, description="Base for exponential backoff calculation."
    )

    def __init__(
        self,
        model_name: str = "text-embedding-004",
        api_key: Optional[str] = None,
        embedding_config: Optional[types.EmbedContentConfigOrDict] = None,
        vertexai_config: Optional[VertexAIConfig] = None,
        http_options: Optional[types.HttpOptions] = None,
        debug_config: Optional[google.genai.client.DebugConfig] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        retries: int = 3,
        timeout: int = 60,
        retry_min_seconds: float = 1,
        retry_max_seconds: float = 60,
        retry_exponential_base: float = 2,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            embedding_config=embedding_config,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            retries=retries,
            timeout=timeout,
            retry_min_seconds=retry_min_seconds,
            retry_max_seconds=retry_max_seconds,
            retry_exponential_base=retry_exponential_base,
            **kwargs,
        )

        # API keys are optional. The API can be authorised via OAuth (detected
        # environmentally) or by the GOOGLE_API_KEY environment variable.
        api_key = api_key or os.getenv("GOOGLE_API_KEY", None)
        vertexai = vertexai_config is not None or os.getenv(
            "GOOGLE_GENAI_USE_VERTEXAI", False
        )
        project = (vertexai_config or {}).get("project") or os.getenv(
            "GOOGLE_CLOUD_PROJECT", None
        )
        location = (vertexai_config or {}).get("location") or os.getenv(
            "GOOGLE_CLOUD_LOCATION", None
        )

        config_params: Dict[str, Any] = {
            "api_key": api_key,
        }

        if vertexai_config is not None:
            config_params.update(vertexai_config)
            config_params["api_key"] = None
            config_params["vertexai"] = True
        elif vertexai:
            config_params["project"] = project
            config_params["location"] = location
            config_params["api_key"] = None
            config_params["vertexai"] = True

        try:
            package_v = version("llama-index-embeddings-google-genai")
        except PackageNotFoundError:
            package_v = "0.0.0"
        client_hdr = {"x-goog-api-client": f"llamaindex/{package_v}"}

        if isinstance(http_options, dict):
            http_opts = http_options
        elif isinstance(http_options, types.HttpOptions):
            http_opts = http_options.to_json_dict()
        else:
            http_opts = {}
        http_opts["headers"] = http_opts.get("headers", {}) | client_hdr

        config_params["http_options"] = types.HttpOptions(**http_opts)

        if debug_config:
            config_params["debug_config"] = debug_config

        self._client = google.genai.Client(**config_params)

    @classmethod
    def class_name(cls) -> str:
        return "GeminiEmbedding"

    def _embed_texts(
        self, texts: List[str], task_type: Optional[str] = None
    ) -> List[List[float]]:
        """Embed texts."""
        # Set the task type if it is not already set
        if task_type and not self.embedding_config:
            embedding_config = types.EmbedContentConfig(task_type=task_type)
        elif task_type and self.embedding_config:
            embedding_config = dict(self.embedding_config)
            embedding_config["task_type"] = task_type
        else:
            embedding_config = self.embedding_config

        # Create the embedding function with retry logic
        def embed_with_client() -> List[List[float]]:
            results = self._client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=embedding_config,
            )
            return [result.values for result in results.embeddings]

        # Apply the retry decorator
        retryable_embed = get_retryable_function(
            embed_with_client,
            max_retries=self.retries,
            min_seconds=self.retry_min_seconds,
            max_seconds=self.retry_max_seconds,
            exponential_base=self.retry_exponential_base,
        )

        return retryable_embed()

    async def _aembed_texts(
        self, texts: List[str], task_type: Optional[str] = None
    ) -> List[List[float]]:
        """Asynchronously embed texts."""
        # Set the task type if it is not already set
        if task_type and not self.embedding_config:
            embedding_config = types.EmbedContentConfig(task_type=task_type)
        elif task_type and self.embedding_config:
            embedding_config = dict(self.embedding_config)
            embedding_config["task_type"] = task_type
        else:
            embedding_config = self.embedding_config

        # Create the async embedding function with retry logic
        async def aembed_with_client() -> List[List[float]]:
            results = await self._client.aio.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=embedding_config,
            )
            return [result.values for result in results.embeddings]

        # Apply the async retry helper
        return await get_retryable_async_function(
            aembed_with_client,
            max_retries=self.retries,
            min_seconds=self.retry_min_seconds,
            max_seconds=self.retry_max_seconds,
            exponential_base=self.retry_exponential_base,
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed_texts([text], task_type="RETRIEVAL_DOCUMENT")[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return (await self._aembed_texts([query], task_type="RETRIEVAL_QUERY"))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return (await self._aembed_texts([text], task_type="RETRIEVAL_DOCUMENT"))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await self._aembed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
