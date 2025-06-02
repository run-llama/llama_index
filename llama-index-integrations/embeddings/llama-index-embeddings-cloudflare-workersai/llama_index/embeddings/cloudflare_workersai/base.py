"""Cloudflare embeddings file."""

from typing import Any, List, Optional
import requests
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

MAX_BATCH_SIZE = 100  # As per Cloudflare's maxItems limit for batch processing

API_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}"


class CloudflareEmbedding(BaseEmbedding):
    """
    Cloudflare Workers AI class for generating text embeddings.

    This class allows for the generation of text embeddings using Cloudflare Workers AI with the BAAI general embedding models.

    Args:
    account_id (str): The Cloudflare Account ID.
    auth_token (str, Optional): The Cloudflare Auth Token. Alternatively, set up environment variable `CLOUDFLARE_AUTH_TOKEN`.
    model (str): The model ID for the embedding service. Cloudflare provides different models for embeddings, check https://developers.cloudflare.com/workers-ai/models/#text-embeddings. Defaults to "@cf/baai/bge-base-en-v1.5".
    embed_batch_size (int): The batch size for embedding generation. Cloudflare's current limit is 100 at max. Defaults to llama_index's default.

    Note:
    Ensure you have a valid Cloudflare account and have access to the necessary AI services and models. The account ID and authorization token are sensitive details; secure them appropriately.

    """

    account_id: str = Field(default=None, description="The Cloudflare Account ID.")
    auth_token: str = Field(default=None, description="The Cloudflare Auth Token.")
    model: str = Field(
        default="@cf/baai/bge-base-en-v1.5",
        description="The model to use when calling Cloudflare AI API",
    )

    _session: Any = PrivateAttr()

    def __init__(
        self,
        account_id: str,
        auth_token: Optional[str] = None,
        model: str = "@cf/baai/bge-base-en-v1.5",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model=model,
            **kwargs,
        )
        self.account_id = account_id
        self.auth_token = get_from_param_or_env(
            "auth_token", auth_token, "CLOUDFLARE_AUTH_TOKEN", ""
        )
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.auth_token}"})

    @classmethod
    def class_name(cls) -> str:
        return "CloudflareEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self._aget_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        result = await self._aget_text_embeddings([text])
        return result[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        response = self._session.post(
            API_URL_TEMPLATE.format(self.account_id, self.model), json={"text": texts}
        ).json()

        if "result" not in response:
            print(response)
            raise RuntimeError("Failed to fetch embeddings")

        return response["result"]["data"]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        import aiohttp

        async with aiohttp.ClientSession(trust_env=True) as session:
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Accept-Encoding": "identity",
            }
            async with session.post(
                API_URL_TEMPLATE.format(self.account_id, self.model),
                json={"text": texts},
                headers=headers,
            ) as response:
                resp = await response.json()
                if "result" not in resp:
                    raise RuntimeError("Failed to fetch embeddings asynchronously")

                return resp["result"]["data"]
