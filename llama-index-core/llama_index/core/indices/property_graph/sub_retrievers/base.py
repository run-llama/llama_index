from abc import abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.graph_stores.types import LabelledPropertyGraphStore
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle

DEFAULT_PREAMBLE = "Here are some facts extracted from the provided text:\n\n"


class BaseLPGRetriever(BaseRetriever):
    def __init__(
        self,
        graph_store: LabelledPropertyGraphStore,
        include_text: bool = True,
        include_text_preamble: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self._graph_store = graph_store
        self.include_text = include_text
        self._include_text_preamble = include_text_preamble or DEFAULT_PREAMBLE
        super().__init__(callback_manager=kwargs.get("callback_manager", None))

    def _add_source_text(
        self, retrieved_nodes: List[NodeWithScore], og_node_map: Dict[str, BaseNode]
    ) -> List[NodeWithScore]:
        """Combine retrieved nodes/triplets with their source text, using provided preamble."""
        # map of ref doc id to triplets/retrieved labelled nodes
        graph_node_map = {}
        for node in retrieved_nodes:
            if node.node.ref_doc_id not in graph_node_map:
                graph_node_map[node.node.ref_doc_id] = []

            graph_node_map[node.node.ref_doc_id].append(node.get_content())

        result_nodes = []
        for node_with_score in retrieved_nodes:
            node = og_node_map.get(node_with_score.node.ref_doc_id, None)
            if node:
                graph_content = graph_node_map.get(node.node_id, [])
                if len(graph_content) > 0:
                    graph_content_str = "\n".join(graph_content)
                    cur_content = node.get_content()
                    node.set_content(
                        self._include_text_preamble
                        + graph_content_str
                        + "\n\n"
                        + cur_content
                    )
                result_nodes.append(
                    NodeWithScore(
                        node=node,
                        score=node_with_score.score,
                    )
                )

        return result_nodes

    def add_source_text(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Combine retrieved nodes/triplets with their source text."""
        if not self.include_text:
            return nodes

        og_nodes = self._graph_store.get_llama_nodes([x.node.ref_doc_id for x in nodes])
        node_map = {node.node_id: node for node in og_nodes}

        return self._add_source_text(nodes, node_map)

    async def async_add_source_text(
        self, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Combine retrieved nodes/triplets with their source text."""
        if not self.include_text:
            return nodes

        og_nodes = await self._graph_store.aget_llama_nodes(
            [x.node.ref_doc_id for x in nodes]
        )
        og_node_map = {node.node_id: node for node in og_nodes}

        return self._add_source_text(nodes, og_node_map)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self.retrieve_from_graph(query_bundle)
        if self.include_text:
            nodes = self.add_source_text(nodes)
        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self.aretrieve_from_graph(query_bundle)
        if self.include_text:
            nodes = await self.async_add_source_text(nodes)
        return nodes

    @abstractmethod
    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from the labelled property graph."""
        ...

    @abstractmethod
    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """Retrieve nodes from the labelled property graph."""
        ...
