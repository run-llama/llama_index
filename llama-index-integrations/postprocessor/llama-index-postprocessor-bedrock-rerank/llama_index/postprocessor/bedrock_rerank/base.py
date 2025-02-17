from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

dispatcher = get_dispatcher(__name__)


class Models(str, Enum):
    COHERE_RERANK_V3_5 = "cohere.rerank-v3-5:0"


class AWSBedrockRerank(BaseNodePostprocessor):
    top_n: int = Field(default=2, description="Top N nodes to return.")
    rerank_model_name: str = Field(
        default=Models.COHERE_RERANK_V3_5.value,
        description="The modelId of the Bedrock model to use.",
    )
    rerank_model_arn: Optional[str] = Field(
        default=None,
        description="Optional custom model ARN to use.",
    )
    profile_name: Optional[str] = Field(
        default=None,
        description=(
            "The name of AWS profile to use. "
            "If not given, then the default profile is used."
        ),
    )
    aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS Access Key ID to use."
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS Secret Access Key to use."
    )
    aws_session_token: Optional[str] = Field(
        default=None, description="AWS Session Token to use."
    )
    region_name: Optional[str] = Field(
        default=None,
        description=(
            "AWS region name to use. "
            "Uses region configured in AWS CLI if not passed."
        ),
    )
    botocore_session: Optional[Any] = Field(
        default=None,
        description="Use this Botocore session instead of creating a new default one.",
        exclude=True,
    )
    botocore_config: Optional[Any] = Field(
        default=None,
        description=(
            "Custom configuration object to use instead of the default generated one."
        ),
        exclude=True,
    )
    max_retries: int = Field(
        default=10,
        description="The maximum number of API retries.",
        gt=0,
    )
    timeout: float = Field(
        default=60.0,
        description=(
            "The timeout for the Bedrock API request in seconds. "
            "It will be used for both connect and read timeouts."
        ),
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the Bedrock client.",
    )
    _client: Any = PrivateAttr()
    _model_package_arn: str = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        rerank_model_name: str = Models.COHERE_RERANK_V3_5.value,
        rerank_model_arn: Optional[str] = None,
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
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.top_n = top_n
        self.rerank_model_name = rerank_model_name
        self.rerank_model_arn = rerank_model_arn
        self.profile_name = profile_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.region_name = region_name
        self.botocore_session = botocore_session
        self.botocore_config = botocore_config
        self.max_retries = max_retries
        self.timeout = timeout
        self.additional_kwargs = additional_kwargs or {}

        session_kwargs = {
            "profile_name": self.profile_name,
            "region_name": self.region_name,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "aws_session_token": self.aws_session_token,
            "botocore_session": self.botocore_session,
        }

        try:
            import boto3
            from botocore.config import Config

            config = (
                Config(
                    retries={"max_attempts": self.max_retries, "mode": "standard"},
                    connect_timeout=self.timeout,
                    read_timeout=self.timeout,
                )
                if self.botocore_config is None
                else self.botocore_config
            )
            session = boto3.Session(**session_kwargs)
        except ImportError:
            raise ImportError(
                "The 'boto3' package was not found. Install it with 'pip install boto3'"
            )

        self.region_name = self.region_name or session.region_name

        if client is not None:
            self._client = client
        else:
            try:
                self._client = session.client("bedrock-agent-runtime", config=config)
            except Exception as e:
                raise ValueError(f"Failed to create Bedrock Agent Runtime client: {e}")

        if self.rerank_model_arn:
            self._model_package_arn = self.rerank_model_arn
        else:
            self._model_package_arn = f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.rerank_model_name}"

    @classmethod
    def class_name(cls) -> str:
        return "AWSBedrockRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if dispatcher:
            dispatcher.event(
                ReRankStartEvent(
                    query=query_bundle,
                    nodes=nodes,
                    top_n=self.top_n,
                    model_name=self.rerank_model_name,
                )
            )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.rerank_model_name,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in nodes
            ]

            # Prepare the text sources for AWS Bedrock
            text_sources = []
            for text in texts:
                text_sources.append(
                    {
                        "type": "INLINE",
                        "inlineDocumentSource": {
                            "type": "TEXT",
                            "textDocument": {"text": text},
                        },
                    }
                )
            # change top_n if the number of nodes is less than top_n
            if len(nodes) < self.top_n:
                self.top_n = len(nodes)

            queries = [
                {
                    "type": "TEXT",
                    "textQuery": {"text": query_bundle.query_str},
                }
            ]

            rerankingConfiguration = {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": self.top_n,
                    "modelConfiguration": {
                        "modelArn": self._model_package_arn,
                    },
                },
            }

            try:
                response = self._client.rerank(
                    queries=queries,
                    sources=text_sources,
                    rerankingConfiguration=rerankingConfiguration,
                )

                results = response["results"]
            except Exception as e:
                raise RuntimeError(f"Failed to invoke AWS Bedrock model: {e}")

            new_nodes = []
            for result in results:
                index = result["index"]
                relevance_score = result.get("relevanceScore", 0.0)
                new_node_with_score = NodeWithScore(
                    node=nodes[index].node,
                    score=relevance_score,
                )
                new_nodes.append(new_node_with_score)

            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
