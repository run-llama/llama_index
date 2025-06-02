"""
TextEmbed: Embedding Inference Server.

TextEmbed offers a high-throughput, low-latency service for generating embeddings using various sentence-transformer models.
It now also supports image embedding models, providing flexibility and scalability for diverse applications.

Maintained by Keval Dekivadiya, TextEmbed is licensed under Apache-2.0.
"""

from typing import Callable, List, Optional, Union

import aiohttp
import requests

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager

DEFAULT_URL = "http://0.0.0.0:8000/v1"


class TextEmbedEmbedding(BaseEmbedding):
    """TextEmbedEmbedding is a class for interfacing with the TextEmbed: embedding inference server."""

    base_url: str = Field(
        default=DEFAULT_URL,
        description="Base URL for the text embeddings service.",
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for the request.",
    )
    auth_token: Optional[Union[str, Callable[[str], str]]] = Field(
        default=None,
        description="Authentication token or authentication token generating function for authenticated requests",
    )

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_URL,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        timeout: float = 60.0,
        callback_manager: Optional[CallbackManager] = None,
        auth_token: Optional[Union[str, Callable[[str], str]]] = None,
    ):
        """
        Initializes the TextEmbedEmbedding object with specified parameters.

        Args:
            model_name (str): The name of the model to be used for embeddings.
            base_url (str): The base URL of the embedding service.
            embed_batch_size (int): The batch size for embedding requests.
            timeout (float): Timeout for requests.
            callback_manager (Optional[CallbackManager]): Manager for handling callbacks.
            auth_token (Optional[Union[str, Callable[[str], str]]]): Authentication token or function for generating it.

        """
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            timeout=timeout,
            callback_manager=callback_manager,
            auth_token=auth_token,
        )

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Calls the TextEmbed API to get embeddings for a list of texts.

        Args:
            texts (List[str]): A list of texts to get embeddings for.

        Returns:
            List[List[float]]: A list of embeddings for the input texts.

        Raises:
            Exception: If the API responds with a status code other than 200.

        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}" if self.auth_token else None,
        }
        json_data = {"input": texts, "model": self.model_name}
        with requests.post(
            f"{self.base_url}/embedding",
            headers=headers,
            json=json_data,
            timeout=self.timeout,
        ) as response:
            if response.status_code != 200:
                raise Exception(
                    f"TextEmbed responded with an unexpected status message "
                    f"{response.status_code}: {response.text}"
                )
            return [e["embedding"] for e in response.json()["data"]]

    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously calls the TextEmbed API to get embeddings for a list of texts.

        Args:
            texts (List[str]): A list of texts to get embeddings for.

        Returns:
            List[List[float]]: A list of embeddings for the input texts.

        Raises:
            Exception: If the API responds with a status code other than 200.

        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}" if self.auth_token else None,
        }
        json_data = {"input": texts, "model": self.model_name}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/embedding",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"TextEmbed responded with an unexpected status message "
                        f"{response.status}: {response.text}"
                    )
                data = await response.json()
                return [e["embedding"] for e in data["data"]]

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Gets the embedding for a single query.

        Args:
            query (str): The query to get the embedding for.

        Returns:
            List[float]: The embedding for the query.

        """
        return self._call_api([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Gets the embedding for a single text.

        Args:
            text (str): The text to get the embedding for.

        Returns:
            List[float]: The embedding for the text.

        """
        return self._call_api([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Gets the embeddings for a list of texts.

        Args:
            texts (List[str]): The texts to get the embeddings for.

        Returns:
            List[List[float]]: A list of embeddings for the input texts.

        """
        return self._call_api(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Asynchronously gets the embedding for a single query.

        Args:
            query (str): The query to get the embedding for.

        Returns:
            List[float]: The embedding for the query.

        """
        return (await self._acall_api([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Asynchronously gets the embedding for a single text.

        Args:
            text (str): The text to get the embedding for.

        Returns:
            List[float]: The embedding for the text.

        """
        return (await self._acall_api([text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously gets the embeddings for a list of texts.

        Args:
            texts (List[str]): The texts to get the embeddings for.

        Returns:
            List[List[float]]: A list of embeddings for the input texts.

        """
        return await self._acall_api(texts)
