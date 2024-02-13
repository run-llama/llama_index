from typing import Any, Dict, List, Optional

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.legacy.core.embeddings.base import BaseEmbedding, Embedding
from llama_index.legacy.embeddings.sagemaker_embedding_endpoint_utils import (
    BaseIOHandler,
    IOHandler,
)
from llama_index.legacy.types import PydanticProgramMode
from llama_index.legacy.utilities.aws_utils import get_aws_service_client

DEFAULT_IO_HANDLER = IOHandler()


class SageMakerEmbedding(BaseEmbedding):
    endpoint_name: str = Field(description="SageMaker Embedding endpoint name")
    endpoint_kwargs: Dict[str, Any] = Field(
        default={},
        description="Additional kwargs for the invoke_endpoint request.",
    )
    model_kwargs: Dict[str, Any] = Field(
        default={},
        description="kwargs to pass to the model.",
    )
    content_handler: BaseIOHandler = Field(
        default=DEFAULT_IO_HANDLER,
        description="used to serialize input, deserialize output, and remove a prefix.",
    )

    profile_name: Optional[str] = Field(
        description="The name of aws profile to use. If not given, then the default profile is used."
    )
    aws_access_key_id: Optional[str] = Field(description="AWS Access Key ID to use")
    aws_secret_access_key: Optional[str] = Field(
        description="AWS Secret Access Key to use"
    )
    aws_session_token: Optional[str] = Field(description="AWS Session Token to use")
    aws_region_name: Optional[str] = Field(
        description="AWS region name to use. Uses region configured in AWS CLI if not passed"
    )
    max_retries: Optional[int] = Field(
        default=3,
        description="The maximum number of API retries.",
        gte=0,
    )
    timeout: Optional[float] = Field(
        default=60.0,
        description="The timeout, in seconds, for API requests.",
        gte=0,
    )
    _client: Any = PrivateAttr()
    _verbose: bool = PrivateAttr()

    def __init__(
        self,
        endpoint_name: str,
        endpoint_kwargs: Optional[Dict[str, Any]] = {},
        model_kwargs: Optional[Dict[str, Any]] = {},
        content_handler: BaseIOHandler = DEFAULT_IO_HANDLER,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        max_retries: Optional[int] = 3,
        timeout: Optional[float] = 60.0,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        verbose: bool = False,
    ):
        if not endpoint_name:
            raise ValueError(
                "Missing required argument:`endpoint_name`"
                " Please specify the endpoint_name"
            )
        endpoint_kwargs = endpoint_kwargs or {}
        model_kwargs = model_kwargs or {}
        content_handler = content_handler
        self._client = get_aws_service_client(
            service_name="sagemaker-runtime",
            profile_name=profile_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._verbose = verbose

        super().__init__(
            endpoint_name=endpoint_name,
            endpoint_kwargs=endpoint_kwargs,
            model_kwargs=model_kwargs,
            content_handler=content_handler,
            embed_batch_size=embed_batch_size,
            pydantic_program_mode=pydantic_program_mode,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(self) -> str:
        return "SageMakerEmbedding"

    def _get_embedding(self, payload: List[str], **kwargs: Any) -> List[Embedding]:
        model_kwargs = {**self.model_kwargs, **kwargs}

        request_body = self.content_handler.serialize_input(
            request=payload, model_kwargs=model_kwargs
        )

        response = self._client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=request_body,
            ContentType=self.content_handler.content_type,
            Accept=self.content_handler.accept,
            **self.endpoint_kwargs,
        )["Body"]

        return self.content_handler.deserialize_output(response=response)

    def _get_query_embedding(self, query: str, **kwargs: Any) -> Embedding:
        query = query.replace("\n", " ")
        return self._get_embedding([query], **kwargs)[0]

    def _get_text_embedding(self, text: str, **kwargs: Any) -> Embedding:
        text = text.replace("\n", " ")
        return self._get_embedding([text], **kwargs)[0]

    def _get_text_embeddings(self, texts: List[str], **kwargs: Any) -> List[Embedding]:
        """
        Embed the input sequence of text synchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        texts = [text.replace("\n", " ") for text in texts]

        # Default implementation just loops over _get_text_embedding
        return self._get_embedding(texts, **kwargs)

    async def _aget_query_embedding(self, query: str, **kwargs: Any) -> Embedding:
        raise NotImplementedError

    async def _aget_text_embedding(self, text: str, **kwargs: Any) -> Embedding:
        raise NotImplementedError

    async def _aget_text_embeddings(
        self, texts: List[str], **kwargs: Any
    ) -> List[Embedding]:
        raise NotImplementedError
