import os
import json
import requests
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode

dispatcher = get_dispatcher(__name__)


class AIMonRerank(BaseNodePostprocessor):

    model: str = Field(description="AIMon's reranking model name.")
    top_n: int = Field(description="Top N nodes to return.")
    task_definition: str = Field(
        default="Determine the relevance of context documents with respect to the user query.",
        description="The task definition for the AIMon reranker.",
    )

    _api_key: str = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "retrieval_relevance",
        api_key: Optional[str] = None,
        task_definition: Optional[str] = None,
    ):
        super().__init__(top_n=top_n, model=model)
        self.task_definition = task_definition or (
            "Determine the relevance of context documents with respect to the user query."
        )
        try:
            self._api_key = api_key or os.environ["AIMON_API_KEY"]
        except IndexError:
            raise ValueError(
                "Must pass in AIMon api key or specify via AIMON_API_KEY environment variable"
            )

    @classmethod
    def class_name(cls) -> str:
        return "AIMonRerank"

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
                model_name=self.model,
            )
        )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        texts = [
            node.node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
        ]

        lengths = {"Document lengths": [len(text.split()) for text in texts]}
        
        print("\n")
        print(json.dumps(lengths, indent=4))

        # Construct the AIMon POST request payload
        payload = [
            {
                "task_definition": self.task_definition,
                "context": texts,
                "user_query": query_bundle.query_str,
                "config": {
                    "retrieval_relevance": {
                        "detector_name": "default"
                    }
                }
            }
        ]

        ## Printing AIMon Payload for debugging
        print("\n")
        print("AIMon Request Payload:")
        print(json.dumps(payload, indent=2))

        # Define the request headers
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Send the request
        url = "https://pbe-api.aimon.ai/v2/detect"
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # Validate response
        if response.status_code != 200:
            raise ValueError(
                f"AIMon API request failed with status code {response.status_code}: {response.text}"
            )

        response_data = response.json()

        # Print the full response data for debugging
        print("\n")
        print("AIMon API Response:")
        print(json.dumps(response_data, indent=2))

        relevance_scores = response_data[0]["retrieval_relevance"][0]["relevance_scores"]

        # Attach scores to nodes
        scored_nodes = [
            NodeWithScore(node=nodes[i].node, score=relevance_scores[i])
            for i in range(len(nodes))
        ]

        # Sort nodes by score in descending order
        scored_nodes.sort(key=lambda x: x.score, reverse=True)

        # Keep only top N nodes
        new_nodes = scored_nodes[: self.top_n]

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes
