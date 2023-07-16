import os
from typing import List, Optional
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import NodeWithScore


class SentenceTransformerRerank(BaseNodePostprocessor):
    def __init__(
        self,
        top_n: int = 2,
        model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Cannot import sentence-transformers package,",
                "please `pip install sentence-transformers`",
            )

        self._top_n = top_n
        self._model = CrossEncoder(model, max_length=512)

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        query_and_nodes = [
            (query_bundle.query_str, node.node.get_content()) for node in nodes
        ]

        scores = self._model.predict(query_and_nodes)

        for node, score in zip(nodes, scores):
            node.score = score

        new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[: self._top_n]

        return new_nodes
