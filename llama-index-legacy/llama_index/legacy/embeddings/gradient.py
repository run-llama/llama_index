import logging
from typing import Any, List, Optional

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.core.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding,
)

logger = logging.getLogger(__name__)


# For bge models that Gradient AI provides, it is suggested to add the instruction for retrieval.
# Reference: https://huggingface.co/BAAI/bge-large-en-v1.5#model-list
QUERY_INSTRUCTION_FOR_RETRIEVAL = (
    "Represent this sentence for searching relevant passages:"
)

GRADIENT_EMBED_BATCH_SIZE: int = 32_768


class GradientEmbedding(BaseEmbedding):
    """GradientAI embedding models.

    This class provides an interface to generate embeddings using a model
    deployed in Gradient AI. At the initialization it requires a model_id
    of the model deployed in the cluster.

    Note:
        Requires `gradientai` package to be available in the PYTHONPATH. It can be installed with
        `pip install gradientai`.
    """

    embed_batch_size: int = Field(default=GRADIENT_EMBED_BATCH_SIZE, gt=0)

    _gradient: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "GradientEmbedding"

    def __init__(
        self,
        *,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        gradient_model_slug: str,
        gradient_access_token: Optional[str] = None,
        gradient_workspace_id: Optional[str] = None,
        gradient_host: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initializes the GradientEmbedding class.

        During the initialization the `gradientai` package is imported. Using the access token,
        workspace id and the slug of the model, the model is fetched from Gradient AI and prepared to use.

        Args:
            embed_batch_size (int, optional): The batch size for embedding generation. Defaults to 10,
                must be > 0 and <= 100.
            gradient_model_slug (str): The model slug of the model in the Gradient AI account.
            gradient_access_token (str, optional): The access token of the Gradient AI account, if
                `None` read from the environment variable `GRADIENT_ACCESS_TOKEN`.
            gradient_workspace_id (str, optional): The workspace ID of the Gradient AI account, if `None`
                read from the environment variable `GRADIENT_WORKSPACE_ID`.
            gradient_host (str, optional): The host of the Gradient AI API. Defaults to None, which
              means the default host is used.

        Raises:
            ImportError: If the `gradientai` package is not available in the PYTHONPATH.
            ValueError: If the model cannot be fetched from Gradient AI.
        """
        if embed_batch_size <= 0:
            raise ValueError(f"Embed batch size {embed_batch_size}  must be > 0.")

        try:
            import gradientai
        except ImportError:
            raise ImportError("GradientEmbedding requires `pip install gradientai`.")

        self._gradient = gradientai.Gradient(
            access_token=gradient_access_token,
            workspace_id=gradient_workspace_id,
            host=gradient_host,
        )

        try:
            self._model = self._gradient.get_embeddings_model(slug=gradient_model_slug)
        except gradientai.openapi.client.exceptions.UnauthorizedException as e:
            logger.error(f"Error while loading model {gradient_model_slug}.")
            self._gradient.close()
            raise ValueError("Unable to fetch the requested embeddings model") from e

        super().__init__(
            embed_batch_size=embed_batch_size, model_name=gradient_model_slug, **kwargs
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input sequence of text asynchronously.
        """
        inputs = [{"input": text} for text in texts]

        result = await self._model.aembed(inputs=inputs).embeddings

        return [e.embedding for e in result]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input sequence of text.
        """
        inputs = [{"input": text} for text in texts]

        result = self._model.embed(inputs=inputs).embeddings

        return [e.embedding for e in result]

    def _get_text_embedding(self, text: str) -> Embedding:
        """Alias for _get_text_embeddings() with single text input."""
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Alias for _aget_text_embeddings() with single text input."""
        embedding = await self._aget_text_embeddings([text])
        return embedding[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        embedding = await self._aget_text_embeddings(
            [f"{QUERY_INSTRUCTION_FOR_RETRIEVAL} {query}"]
        )
        return embedding[0]

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embeddings(
            [f"{QUERY_INSTRUCTION_FOR_RETRIEVAL} {query}"]
        )[0]
