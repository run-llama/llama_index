"""Azure AI model inference embeddings client."""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

from azure.ai.inference import EmbeddingsClient
from azure.ai.inference.aio import EmbeddingsClient as EmbeddingsClientAsync
from azure.core.credentials import AzureKeyCredential

DEFAULT_AZUREAI_ENDPOINT = "https://inference.ai.azure.com"


class AzureAIModelInference(BaseEmbedding):
    """Azure AI model inference for embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "mistral-embed".

        api_key (Optional[str]): API key to access the model. Defaults to None.
    """

    model: Optional[str] = Field(default=None, description="The model id to use.")
    max_retries: int = Field(
        default=5, description="The maximum number of API retries.", gte=0
    )
    seed: str = Field(default=None, description="The random seed to use for sampling.")
    model_extras: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs model parameters."
    )

    _client: EmbeddingsClient = PrivateAttr()
    _async_client: EmbeddingsClientAsync = PrivateAttr()

    def __init__(
        self,
        endpoint: str = None,
        credential: Union[str, AzureKeyCredential, "TokenCredential"] = None,
        model: str = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        endpoint = get_from_param_or_env(
            "endpoint", endpoint, "AZUREAI_ENDPOINT_URL", DEFAULT_AZUREAI_ENDPOINT
        )
        credential = get_from_param_or_env(
            "credential", credential, "AZUREAI_ENDPOINT_CREDENTIAL", None
        )
        credential = (
            AzureKeyCredential(credential)
            if isinstance(credential, str)
            else credential
        )

        if not credential:
            raise ValueError(
                "You must provide an credential to use the Azure AI model inference LLM."
            )

        self._client = EmbeddingsClient(
            endpoint=endpoint,
            credential=credential,
            **kwargs,
        )

        self._async_client = EmbeddingsClientAsync(
            endpoint=endpoint,
            credential=credential,
            **kwargs,
        )

        super().__init__(
            model_name=model,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "AzureAIModelInferenceEmbeddings"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._client.embed(model=self.model, input=[query]).data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return (
            (await self._async_client.embed(model=self.model_name, input=[query]))
            .data[0]
            .embedding
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._client.embed(model=self.model_name, input=[text]).data[0].embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return (
            (await self._async_client.embed(model=self.model_name, input=[text]))
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embedding_response = self._client.embed(model=self.model_name, input=texts).data
        return [embed.embedding for embed in embedding_response]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        embedding_response = await self._async_client.embed(
            model=self.model_name, input=texts
        )
        return [embed.embedding for embed in embedding_response.data]
