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

        result_nodes = []
        for node in nodes:
            if self._storage_context.lpg_graph_store.supports_nodes:
                og_nodes = self._storage_context.lpg_graph_store.get_by_ids(
                    [node.metadata["id_"]]
                )
                if len(og_nodes) > 0:
                    node.node.set_content(og_nodes[0].get_content(metadata_mode="none"))
            else:
                og_node = self._storage_context.docstore.get_document(
                    node.metadata["id_"]
                )
                if og_node:
                    node.node.set_content(og_node.get_content(metadata_mode="none"))

            result_nodes.append(node)

        return result_nodes
