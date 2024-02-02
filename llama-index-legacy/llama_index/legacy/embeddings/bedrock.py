import json
import os
import warnings
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.callbacks.base import CallbackManager
from llama_index.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.embeddings.base import BaseEmbedding, Embedding


class PROVIDERS(str, Enum):
    AMAZON = "amazon"
    COHERE = "cohere"


class Models(str, Enum):
    TITAN_EMBEDDING = "amazon.titan-embed-text-v1"
    TITAN_EMBEDDING_G1_TEXT_02 = "amazon.titan-embed-g1-text-02"
    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"


PROVIDER_SPECIFIC_IDENTIFIERS = {
    PROVIDERS.AMAZON.value: {
        "get_embeddings_func": lambda r: r.get("embedding"),
    },
    PROVIDERS.COHERE.value: {
        "get_embeddings_func": lambda r: r.get("embeddings")[0],
    },
}


class BedrockEmbedding(BaseEmbedding):
    _client: Any = PrivateAttr()
    _verbose: bool = PrivateAttr()

    def __init__(
        self,
        model_name: str = Models.TITAN_EMBEDDING,
        client: Any = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ):
        self._client = client
        self._verbose = verbose

        super().__init__(
            model_name=model_name,
            client=client,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
        )

    @staticmethod
    def list_supported_models() -> Dict[str, List[str]]:
        list_models = {}
        for provider in PROVIDERS:
            list_models[provider.value] = [m.value for m in Models]
        return list_models

    @classmethod
    def class_name(self) -> str:
        return "BedrockEmbedding"

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
                "boto3 package not found, install with" "'pip install boto3'"
            )

        if "bedrock-runtime" in session.get_available_services():
            self._client = session.client("bedrock-runtime")
        else:
            self._client = session.client("bedrock")

    @classmethod
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
                "boto3 package not found, install with" "'pip install boto3'"
            )

        if "bedrock-runtime" in session.get_available_services():
            client = session.client("bedrock-runtime")
        else:
            client = session.client("bedrock")
        return cls(
            client=client,
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    def _get_embedding(self, payload: str, type: Literal["text", "query"]) -> Embedding:
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
        identifiers = PROVIDER_SPECIFIC_IDENTIFIERS.get(provider, None)
        if identifiers is None:
            raise ValueError("Provider not supported")
        return identifiers["get_embeddings_func"](resp)

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_embedding(query, "query")

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_embedding(text, "text")

    def _get_request_body(
        self, provider: str, payload: str, type: Literal["text", "query"]
    ) -> Any:
        """Build the request body as per the provider.
        Currently supported providers are amazon, cohere.

        amazon:
            Sample Payload of type str
            "Hello World!"

        cohere:
            Sample Payload of type dict of following format
            {
                'texts': ["This is a test document", "This is another document"],
                'input_type': 'search_document',
                'truncate': 'NONE'
            }

        """
        if self._verbose:
            print("provider: ", provider, PROVIDERS.AMAZON)
        if provider == PROVIDERS.AMAZON:
            request_body = json.dumps({"inputText": payload})
        elif provider == PROVIDERS.COHERE:
            input_types = {
                "text": "search_document",
                "query": "search_query",
            }
            request_body = json.dumps(
                {
                    "texts": [payload],
                    "input_type": input_types[type],
                    "truncate": "NONE",
                }
            )
        else:
            raise ValueError("Provider not supported")
        return request_body

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_embedding(query, "query")

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._get_embedding(text, "text")
