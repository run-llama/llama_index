from typing import Any, List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.labelled_property_graph.base import (
    LabelledPropertyGraphIndex,
)
from llama_index.core.schema import NodeWithScore


class BaseLPGRetriever(BaseRetriever):
    def __init__(
        self,
        index: LabelledPropertyGraphIndex,
        include_text: bool = True,
        **kwargs: Any
    ) -> None:
        self._storage_context = index.storage_context
        self._include_text = include_text
        super().__init__(callback_manager=kwargs.get("callback_manager", None))

    def _parse_results(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        if not self._include_text:
            return nodes

        og_nodes = self._storage_context.lpg_graph_store.get_nodes(
            [x.node.node_id for x in nodes]
        )
        node_map = {node.node_id: node for node in og_nodes}

        result_nodes = []
        for node_with_score in nodes:
            node = node_map.get(node_with_score.node.node_id, None)
            if node:
                result_nodes.append(
                    NodeWithScore(
                        node=node,
                        score=node_with_score.score,
                    )
                )

        return result_nodes
