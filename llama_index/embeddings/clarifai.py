import logging
from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding

logger = logging.getLogger(__name__)

EXAMPLE_URL = "https://clarifai.com/anthropic/completion/models/claude-v2"


class ClarifaiEmbedding(BaseEmbedding):
    """Clarifai embeddings class.

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

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_url: Optional[str] = None,
        model_version_id: Optional[str] = "",
        app_id: Optional[str] = None,
        user_id: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ):
        try:
            from clarifai.client.model import Model
        except ImportError:
            raise ImportError("ClarifaiEmbedding requires `pip install clarifai`.")

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
                self._model = Model(
                    user_id=user_id,
                    app_id=app_id,
                    model_id=model_name,
                    model_version={"id": model_version_id},
                )

        if model_url is not None:
            self._model = Model(model_url)
            model_name = self._model.id

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name,
        )

    @classmethod
    def class_name(cls) -> str:
        return "ClarifaiEmbedding"

    def _embed(self, sentences: List[str]) -> List[List[float]]:
        """Embed sentences."""
        try:
            from clarifai_grpc.grpc.api import resources_pb2
            from clarifai_grpc.grpc.api.status import status_code_pb2
        except ImportError:
            raise ImportError("ClarifaiEmbedding requires `pip install clarifai`.")

        embeddings = []
        for i in range(0, len(sentences), self.embed_batch_size):
            batch = sentences[i : i + self.embed_batch_size]
            inputs = [
                resources_pb2.Input(
                    data=resources_pb2.Data(text=resources_pb2.Text(raw=t))
                )
                for t in batch
            ]
            post_model_outputs_response = self._model.predict(inputs)

            if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
                logger.error(post_model_outputs_response.status)
                first_output_failure = (
                    post_model_outputs_response.outputs[0].status
                    if len(post_model_outputs_response.outputs)
                    else None
                )
                raise Exception(
                    f"Model prediction failed. Status: "
                    f"{post_model_outputs_response.status}. First output failure: "
                    f"{first_output_failure}"
                )
            embeddings.extend(
                [
                    list(o.data.embeddings[0].vector)
                    for o in post_model_outputs_response.outputs
                ]
            )
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
