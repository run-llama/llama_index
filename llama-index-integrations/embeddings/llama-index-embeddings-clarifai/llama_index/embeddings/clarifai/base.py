import logging
import os
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE

from clarifai.client.model import Model

logger = logging.getLogger(__name__)

EXAMPLE_URL = "https://clarifai.com/anthropic/completion/models/claude-v2"


class ClarifaiEmbedding(BaseEmbedding):
    """
    Clarifai embeddings class.

    Clarifai uses Personal Access Tokens(PAT) to validate requests.
    You can create and manage PATs under your Clarifai account security settings.
    Export your PAT as an environment variable by running `export CLARIFAI_PAT={PAT}`
    """

    model_url: Optional[str] = Field(
        description=f"Full URL of the model. e.g. `{EXAMPLE_URL}`"
    )
    model_id: Optional[str] = Field(description="Model ID.")
    model_version_id: Optional[str] = Field(description="Model Version ID.")
    app_id: Optional[str] = Field(description="Clarifai application ID of the model.")
    user_id: Optional[str] = Field(description="Clarifai user ID of the model.")
    pat: Optional[str] = Field(
        description="Personal Access Tokens(PAT) to validate requests."
    )

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_url: Optional[str] = None,
        model_version_id: Optional[str] = "",
        app_id: Optional[str] = None,
        user_id: Optional[str] = None,
        pat: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ):
        embed_batch_size = min(128, embed_batch_size)

        if pat is None and os.environ.get("CLARIFAI_PAT") is not None:
            pat = os.environ.get("CLARIFAI_PAT")

        if not pat and os.environ.get("CLARIFAI_PAT") is None:
            raise ValueError(
                "Set `CLARIFAI_PAT` as env variable or pass `pat` as constructor argument"
            )

        if model_url is not None and model_name is not None:
            raise ValueError("You can only specify one of model_url or model_name.")
        if model_url is None and model_name is None:
            raise ValueError("You must specify one of model_url or model_name.")

        if model_name is not None:
            if app_id is None or user_id is None:
                raise ValueError(
                    f"Missing one app ID or user ID of the model: {app_id=}, {user_id=}"
                )
            else:
                model = Model(
                    user_id=user_id,
                    app_id=app_id,
                    model_id=model_name,
                    model_version={"id": model_version_id},
                    pat=pat,
                )

        if model_url is not None:
            model = Model(model_url, pat=pat)
            model_name = model.id

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
        )
        self._model = model

    @classmethod
    def class_name(cls) -> str:
        return "ClarifaiEmbedding"

    def _embed(self, sentences: List[str]) -> List[List[float]]:
        """Embed sentences."""
        try:
            from clarifai.client.input import Inputs
        except ImportError:
            raise ImportError("ClarifaiEmbedding requires `pip install clarifai`.")

        embeddings = []
        try:
            for i in range(0, len(sentences), self.embed_batch_size):
                batch = sentences[i : i + self.embed_batch_size]
                input_batch = [
                    Inputs.get_text_input(input_id=str(id), raw_text=inp)
                    for id, inp in enumerate(batch)
                ]
                predict_response = self._model.predict(input_batch)
                embeddings.extend(
                    [
                        list(output.data.embeddings[0].vector)
                        for output in predict_response.outputs
                    ]
                )
        except Exception as e:
            logger.error(f"Predict failed, exception: {e}")

        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed(texts)
