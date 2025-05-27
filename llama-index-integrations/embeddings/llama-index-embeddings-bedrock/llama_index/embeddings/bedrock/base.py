import json
import os
import warnings
from enum import Enum
from deprecated import deprecated
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.types import BaseOutputParser, PydanticProgramMode


class PROVIDERS(str, Enum):
    AMAZON = "amazon"
    COHERE = "cohere"


class Models(str, Enum):
    TITAN_EMBEDDING = "amazon.titan-embed-text-v1"
    TITAN_EMBEDDING_V2_0 = "amazon.titan-embed-text-v2:0"
    TITAN_EMBEDDING_G1_TEXT_02 = "amazon.titan-embed-g1-text-02"
    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"


PROVIDER_SPECIFIC_IDENTIFIERS = {
    PROVIDERS.AMAZON.value: {
        "get_embeddings_func": lambda r, isbatch: r.get("embedding"),
    },
    PROVIDERS.COHERE.value: {
        "get_embeddings_func": lambda r, isbatch: (
            r.get("embeddings") if isbatch else r.get("embeddings")[0]
        ),
    },
}


class BedrockEmbedding(BaseEmbedding):
    model_name: str = Field(description="The modelId of the Bedrock model to use.")
    profile_name: Optional[str] = Field(
        default=None,
        description="The name of aws profile to use. If not given, then the default profile is used.",
    )
    aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS Access Key ID to use"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS Secret Access Key to use"
    )
    aws_session_token: Optional[str] = Field(
        default=None, description="AWS Session Token to use"
    )
    region_name: Optional[str] = Field(
        default=None,
        description="AWS region name to use. Uses region configured in AWS CLI if not passed",
    )
    botocore_session: Optional[Any] = Field(
        default=None,
        description="Use this Botocore session instead of creating a new default one.",
        exclude=True,
    )
    botocore_config: Optional[Any] = Field(
        default=None,
        description="Custom configuration object to use instead of the default generated one.",
        exclude=True,
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries.", gt=0
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout for the Bedrock API request in seconds. It will be used for both connect and read timeouts.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the bedrock client."
    )

    _config: Any = PrivateAttr()
    _client: Any = PrivateAttr()
    _asession: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = Models.TITAN_EMBEDDING,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        client: Optional[Any] = None,
        botocore_session: Optional[Any] = None,
        botocore_config: Optional[Any] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        callback_manager: Optional[CallbackManager] = None,
        # base class
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        **kwargs: Any,
    ):
        additional_kwargs = additional_kwargs or {}

        session_kwargs = {
            "profile_name": profile_name,
            "region_name": region_name,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
            "botocore_session": botocore_session,
        }

        super().__init__(
            model_name=model_name,
            max_retries=max_retries,
            timeout=timeout,
            botocore_config=botocore_config,
            profile_name=profile_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            botocore_session=botocore_session,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            **kwargs,
        )

        try:
            import boto3
            import aioboto3
            from botocore.config import Config

            self._config = (
                Config(
                    retries={"max_attempts": max_retries, "mode": "standard"},
                    connect_timeout=timeout,
                    read_timeout=timeout,
                )
                if botocore_config is None
                else botocore_config
            )
            session = boto3.Session(**session_kwargs)
            self._asession = aioboto3.Session(**session_kwargs)
        except ImportError:
            raise ImportError(
                "boto3 and/or aioboto3 package not found, install with"
                "'pip install boto3 aioboto3"
            )

        # Prior to general availability, custom boto3 wheel files were
        # distributed that used the bedrock service to invokeModel.
        # This check prevents any services still using those wheel files
        # from breaking
        if client is not None:
            self._client = client
        elif "bedrock-runtime" in session.get_available_services():
            self._client = session.client("bedrock-runtime", config=self._config)
        else:
            self._client = session.client("bedrock", config=self._config)

    @staticmethod
    def list_supported_models() -> Dict[str, List[str]]:
        list_models = {}
        for provider in PROVIDERS:
            list_models[provider.value] = [
                m.value for m in Models if provider.value in m.value
            ]
        return list_models

    @classmethod
    def class_name(self) -> str:
        return "BedrockEmbedding"

    @deprecated(
        version="0.9.48",
        reason=(
            "Use the provided kwargs in the constructor, "
            "set_credentials will be removed in future releases."
        ),
        action="once",
    )
    def set_credentials(
        self,
        aws_region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_profile: Optional[str] = None,
    ) -> None:
        aws_region = aws_region or os.getenv("AWS_REGION")
        aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")

        if aws_region is None:
            warnings.warn(
                "AWS_REGION not found. Set environment variable AWS_REGION or set aws_region"
            )

        if aws_access_key_id is None:
            warnings.warn(
                "AWS_ACCESS_KEY_ID not found. Set environment variable AWS_ACCESS_KEY_ID or set aws_access_key_id"
            )
            assert aws_access_key_id is not None

        if aws_secret_access_key is None:
            warnings.warn(
                "AWS_SECRET_ACCESS_KEY not found. Set environment variable AWS_SECRET_ACCESS_KEY or set aws_secret_access_key"
            )
            assert aws_secret_access_key is not None

        if aws_session_token is None:
            warnings.warn(
                "AWS_SESSION_TOKEN not found. Set environment variable AWS_SESSION_TOKEN or set aws_session_token"
            )
            assert aws_session_token is not None

        session_kwargs = {
            "profile_name": aws_profile,
            "region_name": aws_region,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
        }

        try:
            import boto3

            session = boto3.Session(**session_kwargs)
        except ImportError:
            raise ImportError(
                "boto3 package not found, install with'pip install boto3'"
            )

        if "bedrock-runtime" in session.get_available_services():
            self._client = session.client("bedrock-runtime")
        else:
            self._client = session.client("bedrock")

    @classmethod
    @deprecated(
        version="0.9.48",
        reason=(
            "Use the provided kwargs in the constructor, "
            "set_credentials will be removed in future releases."
        ),
        action="once",
    )
    def from_credentials(
        cls,
        model_name: str = Models.TITAN_EMBEDDING,
        aws_region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_profile: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> "BedrockEmbedding":
        """
        Instantiate using AWS credentials.

        Args:
            model_name (str) : Name of the model
            aws_access_key_id (str): AWS access key ID
            aws_secret_access_key (str): AWS secret access key
            aws_session_token (str): AWS session token
            aws_region (str): AWS region where the service is located
            aws_profile (str): AWS profile, when None, default profile is chosen automatically

        Example:
                .. code-block:: python

                    from llama_index.embeddings import BedrockEmbedding

                    # Define the model name
                    model_name = "your_model_name"

                    embeddings = BedrockEmbedding.from_credentials(
                        model_name,
                        aws_access_key_id,
                        aws_secret_access_key,
                        aws_session_token,
                        aws_region,
                        aws_profile,
                    )

        """
        session_kwargs = {
            "profile_name": aws_profile,
            "region_name": aws_region,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
        }

        try:
            import boto3

            session = boto3.Session(**session_kwargs)
        except ImportError:
            raise ImportError(
                "boto3 package not found, install with'pip install boto3'"
            )

        if "bedrock-runtime" in session.get_available_services():
            client = session.client("bedrock-runtime")
        else:
            client = session.client("bedrock")
        return cls(
            client=client,
            model=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    def _get_embedding(
        self, payload: Union[str, List[str]], type: Literal["text", "query"]
    ) -> Union[Embedding, List[Embedding]]:
        """
        Get the embedding for the given payload.

        Args:
            payload (Union[str, List[str]]): The text or list of texts for which the embeddings are to be obtained.
            type (Literal[&quot;text&quot;, &quot;query&quot;]): The type of the payload. It can be either "text" or "query".

        Returns:
            Union[Embedding, List[Embedding]]: The embedding or list of embeddings for the given payload. If the payload is a list of strings, then the response will be a list of embeddings.

        """
        if self._client is None:
            self.set_credentials()

        if self._client is None:
            raise ValueError("Client not set")

        provider = self.model_name.split(".")[0]
        request_body = self._get_request_body(provider, payload, type)

        response = self._client.invoke_model(
            body=request_body,
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
        )

        resp = json.loads(response.get("body").read().decode("utf-8"))
        identifiers = PROVIDER_SPECIFIC_IDENTIFIERS.get(provider)
        if identifiers is None:
            raise ValueError("Provider not supported")
        return identifiers["get_embeddings_func"](resp, isinstance(payload, list))

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_embedding(query, "query")

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_embedding(text, "text")

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        provider = self.model_name.split(".")[0]
        if provider == PROVIDERS.COHERE:
            return self._get_embedding(texts, "text")
        return super()._get_text_embeddings(texts)

    def _get_request_body(
        self,
        provider: str,
        payload: Union[str, List[str]],
        input_type: Literal["text", "query"],
    ) -> Any:
        """
        Build the request body as per the provider.
        Currently supported providers are amazon, cohere.

        amazon:
            Sample Payload of type str
            "Hello World!"

        cohere:
            Sample Payload of type dict of following format
            {
                'texts': ["This is a test document", "This is another document"],
                'input_type': 'search_document'
            }

        """
        if provider == PROVIDERS.AMAZON:
            if isinstance(payload, list):
                raise ValueError("Amazon provider does not support list of texts")

            titan_body_request = {"inputText": payload}

            # Titan Embedding V2.0 has additional body parameters to check.
            if "dimensions" in self.additional_kwargs:
                if self.model_name == Models.TITAN_EMBEDDING_V2_0:
                    titan_body_request["dimensions"] = self.additional_kwargs[
                        "dimensions"
                    ]
                else:
                    raise ValueError(
                        "'dimensions' param not supported outside of 'titan-embed-text-v2:0' model."
                    )
            if "normalize" in self.additional_kwargs:
                if self.model_name == Models.TITAN_EMBEDDING_V2_0:
                    titan_body_request["normalize"] = self.additional_kwargs[
                        "normalize"
                    ]
                else:
                    raise ValueError(
                        "'normalize' param not supported outside of 'titan-embed-text-v2:0' model."
                    )

            request_body = json.dumps(titan_body_request)

        elif provider == PROVIDERS.COHERE:
            input_types = {
                "text": "search_document",
                "query": "search_query",
            }
            payload = [payload] if isinstance(payload, str) else payload
            payload = [p[:2048] if len(p) > 2048 else p for p in payload]
            request_body = json.dumps(
                {
                    "texts": payload,
                    "input_type": input_types[input_type],
                }
            )
        else:
            raise ValueError("Provider not supported")
        return request_body

    async def _aget_embedding(
        self, payload: Union[str, List[str]], type: Literal["text", "query"]
    ) -> Union[Embedding, List[Embedding]]:
        """
        Get the embedding asynchronously for the given payload.

        Args:
            payload (Union[str, List[str]]): The text or list of texts for which the embeddings are to be obtained.
            type (Literal[&quot;text&quot;, &quot;query&quot;]): The type of the payload. It can be either "text" or "query".

        Returns:
            Union[Embedding, List[Embedding]]: The embedding or list of embeddings for the given payload. If the payload is a list of strings, then the response will be a list of embeddings.

        """
        if self._asession is None:
            raise ValueError("Client not set")

        provider = self.model_name.split(".")[0]
        request_body = self._get_request_body(provider, payload, type)

        async with self._asession.client(
            "bedrock-runtime", config=self._config
        ) as client:
            response = await client.invoke_model(
                body=request_body,
                modelId=self.model_name,
                accept="application/json",
                contentType="application/json",
            )
            streaming_body = await response.get("body").read()
            resp = json.loads(streaming_body.decode("utf-8"))

        identifiers = PROVIDER_SPECIFIC_IDENTIFIERS.get(provider)
        if identifiers is None:
            raise ValueError("Provider not supported")
        return identifiers["get_embeddings_func"](resp, isinstance(payload, list))

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await self._aget_embedding(query, "query")

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await self._aget_embedding(text, "text")
