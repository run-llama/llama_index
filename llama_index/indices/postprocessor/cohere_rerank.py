import os
from typing import Dict, List, Optional, cast
from llama_index.data_structs.node import NodeWithScore
from llama_index.indices.postprocessor.node import BaseNodePostprocessor
from llama_index.indices.query.schema import QueryBundle


class CohereRerank(BaseNodePostprocessor):
    def __init__(
        self,
        top_n: int = 2,
        model: str = "rerank-english-v2.0",
        api_key: Optional[str] = None,
    ):
        try:
            api_key = api_key or os.environ["COHERE_API_KEY"]
        except IndexError:
            raise ValueError(
                "Must pass in cohere api key or "
                "specify via COHERE_API_KEY environment variable "
            )
        try:
            from cohere import Client
        except ImportError:
            raise ImportError(
                "Cannot import cohere package, please `pip install cohere`."
            )

        self._client = Client(api_key=api_key)
        self._top_n = top_n
        self._model = model

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        extra_info: Optional[Dict],
    ) -> List[NodeWithScore]:
        if extra_info is None or "query_bundle" not in extra_info:
            raise ValueError("Missing query bundle in extra info.")

        query_bundle = cast(QueryBundle, extra_info["query_bundle"])

        texts = [node.node.get_text() for node in nodes]
        results = self._client.rerank(
            model=self._model,
            top_n=self._top_n,
            query=query_bundle.query_str,
            documents=texts,
        )

        new_nodes = []
        for result in results:
            new_node_with_score = NodeWithScore(
                nodes[result.index].node, result.relevance_score
            )
            new_nodes.append(new_node_with_score)
        return new_nodes
