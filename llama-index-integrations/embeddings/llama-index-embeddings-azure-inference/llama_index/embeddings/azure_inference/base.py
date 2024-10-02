"""Azure AI model inference embeddings client."""

import logging
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
from azure.core.exceptions import HttpResponseError

logger = logging.getLogger(__name__)


class AzureAIEmbeddingsModel(BaseEmbedding):
    """Azure AI model inference for embeddings.

    Examples:
        ```python
        from llama_index.core import Settings
        from llama_index.embeddings.azure_inference import AzureAIEmbeddingsModel

        llm = AzureAIEmbeddingsModel(
            endpoint="https://[your-endpoint].inference.ai.azure.com",
            credential="your-api-key",
        )

        # # If using Microsoft Entra ID authentication, you can create the
        # # client as follows
        #
        # from azure.identity import DefaultAzureCredential
        #
        # embed_model = AzureAIEmbeddingsModel(
        #     endpoint="https://[your-endpoint].inference.ai.azure.com",
        #     credential=DefaultAzureCredential()
        # )
        #
        # # If you plan to use asynchronous calling, make sure to use the async
        # # credentials as follows
        #
        # from azure.identity.aio import DefaultAzureCredential as DefaultAzureCredentialAsync
        #
        # embed_model = AzureAIEmbeddingsModel(
        #     endpoint="https://[your-endpoint].inference.ai.azure.com",
        #     credential=DefaultAzureCredentialAsync()
        # )

        # Once the client is instantiated, you can set the context to use the model
        Settings.embed_model = embed_model

        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        ```
    """

    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs model parameters."
    )

    _client: EmbeddingsClient = PrivateAttr()
    _async_client: EmbeddingsClientAsync = PrivateAttr()

    def __init__(
        self,
        endpoint: str = None,
        credential: Union[str, AzureKeyCredential, "TokenCredential"] = None,
        model_name: str = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        num_workers: Optional[int] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        client_kwargs = client_kwargs or {}

        endpoint = get_from_param_or_env(
            "endpoint", endpoint, "AZURE_INFERENCE_ENDPOINT", None
        )
        credential = get_from_param_or_env(
            "credential", credential, "AZURE_INFERENCE_CREDENTIAL", None
        )
        credential = (
            AzureKeyCredential(credential)
            if isinstance(credential, str)
            else credential
        )

        if not endpoint:
            raise ValueError(
                "You must provide an endpoint to use the Azure AI model inference LLM."
                "Pass the endpoint as a parameter or set the AZURE_INFERENCE_ENDPOINT"
                "environment variable."
            )

        if not credential:
            raise ValueError(
                "You must provide an credential to use the Azure AI model inference LLM."
                "Pass the credential as a parameter or set the AZURE_INFERENCE_CREDENTIAL"
            )

        client = EmbeddingsClient(
            endpoint=endpoint,
            credential=credential,
            user_agent="llamaindex",
            **client_kwargs,
        )

        async_client = EmbeddingsClientAsync(
            endpoint=endpoint,
            credential=credential,
            user_agent="llamaindex",
            **client_kwargs,
        )

        if not model_name:
            try:
                # Get model info from the endpoint. This method may not be supported by all
                # endpoints.
                model_info = client.get_model_info()
                model_name = model_info.get("model_name", None)
            except HttpResponseError:
                logger.warning(
                    f"Endpoint '{self._client._config.endpoint}' does not support model metadata retrieval. "
                    "Unable to populate model attributes."
                )

        super().__init__(
            model_name=model_name or "unknown",
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            num_workers=num_workers,
            **kwargs,
        )

        self._client = client
        self._async_client = async_client

    @classmethod
    def class_name(cls) -> str:
        return "AzureAIEmbeddingsModel"

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        additional_kwargs = {}
        if self.model_name and self.model_name != "unknown":
            additional_kwargs["model"] = self.model_name
        if self.model_kwargs:
            # pass any extra model parameters
            additional_kwargs.update(self.model_kwargs)

        return additional_kwargs

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._client.embed(input=[query], **self._model_kwargs).data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return (
            (await self._async_client.embed(input=[query], **self._model_kwargs))
            .data[0]
            .embedding
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._client.embed(input=[text], **self._model_kwargs).data[0].embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return (
            (await self._async_client.embed(input=[text], **self._model_kwargs))
            .data[0]
            .embedding
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embedding_response = self._client.embed(input=texts, **self._model_kwargs).data
        return [embed.embedding for embed in embedding_response]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        embedding_response = await self._async_client.embed(
            input=texts, **self._model_kwargs
        )
        return [embed.embedding for embed in embedding_response.data]
