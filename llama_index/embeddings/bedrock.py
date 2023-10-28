from typing import Any, List, Optional
import warnings
import json
import os
from llama_index.bridge.pydantic import PrivateAttr
from llama_index.embeddings.base import BaseEmbedding

Embedding = List[float]


class Models(str):
    TITAN_EMBEDDING = "amazon.titan-embed-text-v1"


# To be used when AWS Bedrock comes with more embedding models
# class ModelParams(str):
# TITAN_EMBEDDING = {"inputText":""}


class BedrockEmbeddings(BaseEmbedding):
    _client: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = Models.TITAN_EMBEDDING,
        max_length: int = 8000,
        client: Any = None,
    ):
        self._client = client

        super().__init__(model_name=model_name, client=client)

    @classmethod
    def class_name(self) -> str:
        return "BedrockEmbeddings"

    def set_credentials(
        self,
        model_name: str = Models.TITAN_EMBEDDING,
        aws_region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_profile: str = "default",
    ):

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
        aws_profile: str = "default",
    ):
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
        return cls(client=client, model_name=model_name)

    """
    Instantiate using AWS credentials

    Args:
        model_name (str) : Name of the model
        aws_access_key_id (str): AWS access key ID
        aws_secret_access_key (str): AWS secret access key
        aws_session_token (str): AWS session token
        aws_region (str): AWS region where the service is located
        aws_profile (str): AWS profile 

    Example:
            .. code-block:: python

                from llama_index.embeddings import BedrockEmbeddings

                # Define the model name
                model_name = "your_model_name"

                embeddings = BedrockEmbeddings.from_credentials(
                    model_name,
                    aws_access_key_id,
                    aws_secret_access_key,
                    aws_session_token,
                    aws_region,
                    aws_profile,
                )
    
    """

    def _get_query_embedding(self, query: str) -> Embedding:
        if self._client is None:
            self.set_credentials(self.model_name)

        client = self._client
        body = json.dumps({"inputText": query})
        response = client.invoke_model(
            body=body,
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
        )
        embeddings = json.loads(response.get("body").read())["embedding"]
        return embeddings

    def _aget_query_embedding(self, query: str) -> Embedding:
        raise NotImplementedError("Use _get_embedding ")

    def _get_text_embedding(self, text: str) -> Embedding:
        return _get_query_embedding(text)

    def _aget_text_embedding(self, text: str) -> Embedding:
        raise NotImplementedError("Use _get_test_embedding ")
