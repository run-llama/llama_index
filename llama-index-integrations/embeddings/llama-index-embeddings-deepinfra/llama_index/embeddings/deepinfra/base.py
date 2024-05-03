import logging
import os

import requests
from typing import List, Union

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr


logger = logging.getLogger(__name__)

"""DeepInfra Inference API URL."""
INFERENCE_URL = "https://api.deepinfra.com/v1/inference"
"""Environment variable name of DeepInfra API token."""
ENV_VARIABLE = "DEEPINFRA_API_TOKEN"
"""Default model ID for DeepInfra embeddings."""
DEFAULT_MODEL_ID = "sentence-transformers/clip-ViT-B-32"
"""Maximum batch size for embedding requests."""
MAX_BATCH_SIZE = 1024


class DeepInfraEmbeddingModel(BaseEmbedding):
    """
    A wrapper class for accessing embedding models available via the DeepInfra API. This class allows for easy integration
    of DeepInfra embeddings into your projects, supporting both synchronous and asynchronous retrieval of text embeddings.

    Args:
        model_id (str): Identifier for the model to be used for embeddings. Defaults to 'sentence-transformers/clip-ViT-B-32'.
        normalize (bool): Flag to normalize embeddings post retrieval. Defaults to False.
        api_token (str): DeepInfra API token. If not provided,
        the token is fetched from the environment variable 'DEEPINFRA_API_TOKEN'.

    Examples:
        >>> from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
        >>> model = DeepInfraEmbeddingModel()
        >>> print(model.get_text_embedding("Hello, world!"))
        [0.1, 0.2, 0.3, ...]
    """

    _model_id: str = PrivateAttr()
    _normalize: bool = PrivateAttr()
    _api_token: str = PrivateAttr()

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        normalize: bool = False,
        api_token: str = None,
    ) -> None:
        """
        Init params.
        """
        self._model_id = model_id
        self._normalize = normalize
        self._api_token = os.getenv(ENV_VARIABLE, api_token)

    def _post(self, data: Union[str, list[str]]):
        """
        Sends a POST request to the DeepInfra Inference API with the given data and returns the API response.

        Args:
            data (str | list[str]): Text or list of texts to be embedded.

        Returns:
            dict: A dictionary containing embeddings from the API.
        """
        url = f"{INFERENCE_URL}/{self._model_id}"
        resp = requests.post(
            url,
            json={
                "inputs": data,
            },
            headers={
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
            },
        )
        return resp.json()

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get query embedding.
        """
        return self._post([query])["embeddings"][0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Async get query embedding.
        """
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get text embedding.
        """
        return self._get_query_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Async get text embedding.
        """
        return self._get_query_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get text embedding.
        """
        return self._post(texts)["embeddings"]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Async get text embeddings.
        """
        return self._get_text_embeddings(texts)
