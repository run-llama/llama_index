import logging
from typing import Any, List

import requests
from requests.adapters import HTTPAdapter, Retry

from llama_index.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)


class LLMRailsEmbedding(BaseEmbedding):
    """LLMRails embedding models.

    This class provides an interface to generate embeddings using a model deployed
    in an LLMRails cluster. It requires a model_id of the model deployed in the cluster and api key you can obtain
    from https://console.llmrails.com/api-keys.

    """

    model_id: str
    api_key: str
    session: requests.Session

    @classmethod
    def class_name(self) -> str:
        return "LLMRailsEmbedding"

    def __init__(
        self,
        api_key: str,
        model_id: str = "embedding-english-v1",  # or embedding-multi-v1
        **kwargs: Any,
    ):
        retry = Retry(
            total=3,
            connect=3,
            read=2,
            allowed_methods=["POST"],
            backoff_factor=2,
            status_forcelist=[502, 503, 504],
        )
        session = requests.Session()
        session.mount("https://api.llmrails.com", HTTPAdapter(max_retries=retry))
        session.headers = {"X-API-KEY": api_key}
        super().__init__(model_id=model_id, api_key=api_key, session=session, **kwargs)

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text (str): The query text to generate an embedding for.

        Returns:
            List[float]: The embedding for the input query text.
        """
        try:
            response = self.session.post(
                "https://api.llmrails.com/v1/embeddings",
                json={"input": [text], "model": self.model_id},
            )

            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

        except requests.exceptions.HTTPError as e:
            logger.error(f"Error while embedding text {e}.")
            raise ValueError(f"Unable to embed given text {e}")

    async def _aget_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text (str): The query text to generate an embedding for.

        Returns:
            List[float]: The embedding for the input query text.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "The httpx library is required to use the async version of "
                "this function. Install it with `pip install httpx`."
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.llmrails.com/v1/embeddings",
                    headers={"X-API-KEY": self.api_key},
                    json={"input": [text], "model": self.model_id},
                )

                response.raise_for_status()

            return response.json()["data"][0]["embedding"]

        except httpx._exceptions.HTTPError as e:
            logger.error(f"Error while embedding text {e}.")
            raise ValueError(f"Unable to embed given text {e}")

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aget_embedding(query)

    async def _aget_text_embedding(self, query: str) -> List[float]:
        return await self._aget_embedding(query)


LLMRailsEmbeddings = LLMRailsEmbedding
