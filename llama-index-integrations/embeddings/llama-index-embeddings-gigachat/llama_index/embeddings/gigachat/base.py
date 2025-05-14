from typing import Any, List, Optional

from gigachat import GigaChat  # Install GigaChat API library via 'pip install gigachat'
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager


class GigaChatEmbedding(BaseEmbedding):
    """
    GigaChat encoder class for generating embeddings.

    Attributes:
        _client (Optional[GigaChat]): Instance of the GigaChat client.
        type (str): Type identifier for the encoder, which is "gigachat".

    Example:
        .. code-block:: python
            from llama_index.embeddings.gigachat import GigaChatEmbeddings

            embeddings = GigaChatEmbeddings(
                credentials=..., scope=..., verify_ssl_certs=False
            )

    """

    _client: Optional[GigaChat] = PrivateAttr()
    type: str = "gigachat"

    def __init__(
        self,
        name: Optional[str] = "Embeddings",
        auth_data: Optional[str] = None,
        scope: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        auth_data = get_from_param_or_env(
            "auth_data", auth_data, "GIGACHAT_AUTH_DATA", ""
        )
        if not auth_data:
            raise ValueError(
                "You must provide an AUTH DATA to use GigaChat. "
                "You can either pass it in as an argument or set it `GIGACHAT_AUTH_DATA`."
            )
        if scope is None:
            raise ValueError(
                """
                GigaChat scope cannot be 'None'.
                Set 'GIGACHAT_API_PERS' for personal use or 'GIGACHAT_API_CORP' for corporate use.
                """
            )
        super().__init__(
            model_name=name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )
        try:
            self._client = GigaChat(
                scope=scope, credentials=auth_data, verify_ssl_certs=False
            )
        except Exception as e:
            raise ValueError(f"GigaChat client failed to initialize. Error: {e}") from e

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "GigaChatEmbedding"

    def _get_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        """
        Synchronously Embed documents using a GigaChat embeddings model.

        Args:
            queries: The list of documents to embed.

        Returns:
            List of embeddings, one for each document.

        """
        embeddings = self._client.embeddings(queries).data
        return [embeds_obj.embedding for embeds_obj in embeddings]

    async def _aget_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        """
        Asynchronously embed documents using a GigaChat embeddings model.

        Args:
            queries: The list of documents to embed.

        Returns:
            List of embeddings, one for each document.

        """
        embeddings = (await self._client.aembeddings(queries)).data
        return [embeds_obj.embedding for embeds_obj in embeddings]

    def _get_query_embedding(self, query: List[str]) -> List[float]:
        """
        Synchronously embed a document using GigaChat embeddings model.

        Args:
            query: The document to embed.

        Returns:
            Embeddings for the document.

        """
        return self._client.embeddings(query).data[0].embedding

    async def _aget_query_embedding(self, query: List[str]) -> List[float]:
        """
        Asynchronously embed a query using GigaChat embeddings model.

        Args:
            query: The document to embed.

        Returns:
            Embeddings for the document.

        """
        return (await self._client.aembeddings(query)).data[0].embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Synchronously embed a text using GigaChat embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.

        """
        return self._client.embeddings([text]).data[0].embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Asynchronously embed a text using GigaChat embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.

        """
        return (await self._client.aembeddings([text])).data[0].embedding
