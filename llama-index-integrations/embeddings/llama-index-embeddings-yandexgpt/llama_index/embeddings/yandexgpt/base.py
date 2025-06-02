"""YandexGPT embeddings file."""

import time
import asyncio
import aiohttp
import requests
from typing import Any, List, Optional
from tenacity import Retrying, RetryError, stop_after_attempt, wait_fixed

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.embeddings.yandexgpt.util import YException

DEFAULT_YANDEXGPT_API_BASE = (
    "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
)


class YandexGPTEmbedding(BaseEmbedding):
    """
    A class representation for generating embeddings using the Yandex Cloud API.

    Args:
      api_key (Optional[str]): An API key for Yandex Cloud.
      model_name (str): The name of the model to be used for generating embeddings.
                         The class ensures that this model is supported. Defaults to "general:embedding".
      embed_batch_size (int): The batch size for embedding. Defaults to DEFAULT_EMBED_BATCH_SIZE.
      callback_manager (Optional[CallbackManager]): Callback manager for hooks.

    Example:
        . code-block:: python

            from llama_index.embeddings.yandexgpt import YandexGPTEmbedding

            embeddings = YandexGPTEmbedding(
                api_key="your-api-key",
                folder_id="your-folder-id",
            )

    """

    api_key: str = Field(description="The YandexGPT API key.")
    folder_id: str = Field(description="The folder id for YandexGPT API.")
    retries: int = 6
    sleep_interval: float = 0.1

    def __init__(
        self,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        model_name: str = "general:embedding",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        if not api_key:
            raise ValueError(
                "You must provide an API key or IAM token to use YandexGPT. "
                "You can either pass it in as an argument or set it `YANDEXGPT_API_KEY`."
            )
        if not folder_id:
            raise ValueError(
                "You must provide catalog_id to use YandexGPT. "
                "You can either pass it in as an argument or set it `YANDEXGPT_CATALOG_ID`."
            )

        api_key = get_from_param_or_env("api_key", api_key, "YANDEXGPT_KEY")
        folder_id = get_from_param_or_env(
            "folder_id", folder_id, "YANDEXGPT_CATALOG_ID"
        )

        super().__init__(
            model_name=model_name,
            api_key=api_key,
            folder_id=folder_id,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    def _getModelUri(self, is_document: bool = False) -> str:
        """Construct the model URI based on whether the text is a document or a query."""
        return f"emb://{self.folder_id}/text-search-{'doc' if is_document else 'query'}/latest"

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "YandexGPTEmbedding"

    def _embed(self, text: str, is_document: bool = False) -> List[float]:
        """
        Embeds text using the YandexGPT Cloud API synchronously.

        Args:
          text: The text to embed.
          is_document: Whether the text is a document (True) or a query (False).

        Returns:
          A list of floats representing the embedding.

        Raises:
          YException: If an error occurs during embedding.

        """
        payload = {"modelUri": self._getModelUri(is_document), "text": text}
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
            "x-data-logging-enabled": "false",
        }
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(self.retries),
                wait=wait_fixed(self.sleep_interval),
            ):
                with attempt:
                    response = requests.post(
                        DEFAULT_YANDEXGPT_API_BASE, json=payload, headers=header
                    )
                    response = response.json()
                    if "embedding" in response:
                        return response["embedding"]
                    raise YException(f"No embedding found, result returned: {response}")
        except RetryError:
            raise YException(
                f"Error computing embeddings after {self.retries} retries. Result returned:\n{response}"
            )

    async def _aembed(self, text: str, is_document: bool = False) -> List[float]:
        """
        Embeds text using the YandexGPT Cloud API asynchronously.

        Args:
          text: The text to embed.
          is_document: Whether the text is a document (True) or a query (False).

        Returns:
          A list of floats representing the embedding.

        Raises:
          YException: If an error occurs during embedding.

        """
        payload = {"modelUri": self._getModelUri(is_document), "text": text}
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
            "x-data-logging-enabled": "false",
        }
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(self.retries),
                wait=wait_fixed(self.sleep_interval),
            ):
                with attempt:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            DEFAULT_YANDEXGPT_API_BASE, json=payload, headers=header
                        ) as response:
                            result = await response.json()
                            if "embedding" in result:
                                return result["embedding"]
                            raise YException(
                                f"No embedding found, result returned: {result}"
                            )
        except RetryError:
            raise YException(
                f"Error computing embeddings after {self.retries} retries. Result returned:\n{result}"
            )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding sync."""
        return self._embed(text, is_document=True)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get list of texts embeddings sync."""
        embeddings = []
        for text in texts:
            embeddings.append(self._embed(text, is_document=True))
            time.sleep(self.sleep_interval)
        return embeddings

    def _get_query_embedding(self, text: str) -> List[float]:
        """Get query embedding sync."""
        return self._embed(text, is_document=False)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get query text async."""
        return await self._aembed(text, is_document=True)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get list of texts embeddings async."""
        embeddings = []
        for text in texts:
            embeddings.append(await self._aembed(text, is_document=True))
            await asyncio.sleep(self.sleep_interval)
        return embeddings

    async def _aget_query_embedding(self, text: str) -> List[float]:
        """Get query embedding async."""
        return await self._aembed(text, is_document=False)
