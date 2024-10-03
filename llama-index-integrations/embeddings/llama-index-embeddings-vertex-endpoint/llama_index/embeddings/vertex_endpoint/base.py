from typing import Any, Dict, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from google.cloud import aiplatform
from google.oauth2 import service_account
from llama_index.embeddings.vertex_endpoint.utils import (
    BaseIOHandler,
    IOHandler,
)

DEFAULT_IO_HANDLER = IOHandler()


class VertexEndpointEmbedding(BaseEmbedding):
    endpoint_id: str = Field(description="Vertex AI endpoint ID")
    project_id: str = Field(description="GCP Project ID")
    location: str = Field(description="GCP Region for Vertex AI")
    endpoint_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the predict request.",
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="kwargs to pass to the model.",
    )
    content_handler: BaseIOHandler = Field(
        default=DEFAULT_IO_HANDLER,
        description="used to format input/output",
    )
    service_account_file: Optional[str] = Field(
        default=None, description="Path to the service account JSON file."
    )
    service_account_info: Optional[Dict[str, str]] = Field(
        default=None, description="Directly provide service account credentials."
    )
    timeout: Optional[float] = Field(
        default=60.0,
        description="Timeout for API requests in seconds.",
        ge=0,
    )
    _client: aiplatform.Endpoint = PrivateAttr()
    _verbose: bool = PrivateAttr()

    def __init__(
        self,
        endpoint_id: str,
        project_id: str,
        location: str,
        content_handler: BaseIOHandler = DEFAULT_IO_HANDLER,
        endpoint_kwargs: Optional[Dict[str, Any]] = {},
        model_kwargs: Optional[Dict[str, Any]] = {},
        service_account_file: Optional[str] = None,
        service_account_info: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = 60.0,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ):
        super().__init__(
            endpoint_id=endpoint_id,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            project_id=project_id,
            location=location,
            content_handler=content_handler,
            endpoint_kwargs=endpoint_kwargs or {},
            model_kwargs=model_kwargs or {},
            timeout=timeout,
        )

        # Initialize the client
        if service_account_file:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file
            )
        elif service_account_info:
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info
            )
        else:
            credentials = None  # Use default application credentials if not provided
        try:
            self._client = aiplatform.Endpoint(
                endpoint_name=endpoint_id,
                project=project_id,
                location=location,
                credentials=credentials,
            )
        except Exception as e:
            raise ValueError("Please verify the provided credentials.") from (e)

        self._verbose = verbose

    @classmethod
    def class_name(cls) -> str:
        return "VertexEndpointEmbedding"

    def _get_embedding(self, payload: List[str], **kwargs: Any) -> List[Embedding]:
        # Combine model kwargs with any additional kwargs passed to the function
        endpoint_kwargs = {**self.endpoint_kwargs, **{"timeout": self.timeout}}
        model_kwargs = {**self.model_kwargs, **kwargs}

        # Directly send the input payload to the endpoint
        response = self._client.predict(
            instances=self.content_handler.serialize_input(payload),
            parameters=model_kwargs,
            **endpoint_kwargs
        )

        # Assuming response contains the embeddings in a field called 'predictions'
        return self.content_handler.deserialize_output(response)

    async def _aget_embedding(
        self, payload: List[str], **kwargs: Any
    ) -> List[Embedding]:
        # Combine model kwargs with any additional kwargs passed to the function
        endpoint_kwargs = {**self.endpoint_kwargs, **{"timeout": self.timeout}}
        model_kwargs = {**self.model_kwargs, **kwargs}

        # Directly send the input payload to the endpoint
        response = await self._client.predict_async(
            instances=self.content_handler.serialize_input(payload),
            parameters=model_kwargs,
            **endpoint_kwargs
        )

        # Assuming response contains the embeddings in a field called 'predictions'
        return self.content_handler.deserialize_output(response)

    def _get_query_embedding(self, query: str, **kwargs: Any) -> Embedding:
        query = query.replace("\n", " ")
        return self._get_embedding([query], **kwargs)[0]

    def _get_text_embedding(self, text: str, **kwargs: Any) -> Embedding:
        text = text.replace("\n", " ")
        return self._get_embedding([text], **kwargs)[0]

    def _get_text_embeddings(self, texts: List[str], **kwargs: Any) -> List[Embedding]:
        texts = [text.replace("\n", " ") for text in texts]
        return self._get_embedding(texts, **kwargs)

    async def _aget_query_embedding(self, query: str, **kwargs: Any) -> Embedding:
        query = query.replace("\n", " ")
        return await self._aget_embedding([query], **kwargs)[0]

    async def _aget_text_embedding(self, text: str, **kwargs: Any) -> Embedding:
        text = text.replace("\n", " ")
        return await self._aget_embedding([text], **kwargs)[0]

    async def _aget_text_embeddings(
        self, texts: List[str], **kwargs: Any
    ) -> List[Embedding]:
        texts = [text.replace("\n", " ") for text in texts]
        return await self._aget_embedding(texts, **kwargs)
