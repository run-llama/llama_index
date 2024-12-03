from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode

dispatcher = get_dispatcher(__name__)


class AWSBedrockRerank(BaseNodePostprocessor):
    top_n: int = Field(description="Top N nodes to return.")
    model_id: str = Field(description="AWS Bedrock model ID.")
    region_name: str = Field(description="AWS region name for the Bedrock service.")

    _client: Any = PrivateAttr()
    _model_package_arn: str = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model_id: str = "cohere.rerank-v3-5:0",
        region_name: str = "us-west-2",
        **kwargs: Any,
    ):
        super().__init__(top_n=top_n, model_id=model_id, region_name=region_name)
        try:
            import boto3

            self.region_name = region_name or boto3.Session().region_name
            self._client = boto3.client(
                "bedrock-agent-runtime", region_name=self.region_name
            )
            self._model_package_arn = (
                f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.model_id}"
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize AWS Bedrock client: {e}")

    @classmethod
    def class_name(cls) -> str:
        return "AWSBedrockRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model_id,
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
                EventPayload.MODEL_NAME: self.model_id,
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
