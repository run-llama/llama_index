"""GradientAI embeddings wrapper """

import logging
from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.embeddings.base import BaseEmbedding, Embedding

logger = logging.getLogger(__name__)


# For bge models that Gradient AI provides, it is suggested to add the instruction for retrieval. Reference:
QUERY_INSTRUCTION_FOR_RETRIEVAL = (
    "Represent this sentence for searching relevant passages:"
)


class GradientEmbedding(BaseEmbedding):
    """GradientAI embedding models.

    This class provides an interface to generate embeddings using a model
    deployed in Gradient AI. At the initialization it requires a model_id
    of the model deployed in the cluster.
    """

    _gradient: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "GradientEmbedding"

    def __init__(
        self,
        gradient_access_token: str,
        gradient_workspace_id: str,
        gradient_model_slug: str,
        gradient_host: Optional[str] = None,
        **kwargs: Any,
    ):
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

        super().__init__(model_name=gradient_model_slug, **kwargs)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        inputs = [{"input": text} for text in texts]

        result = self._model.generate_embeddings(inputs=inputs).embeddings

        return [e.embedding for e in result]

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(f"{QUERY_INSTRUCTION_FOR_RETRIEVAL}{query}")
