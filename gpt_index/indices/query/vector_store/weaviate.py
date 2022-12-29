"""Weaviate vector store index query."""


from typing import Any, List, Optional, Tuple, cast

from gpt_index.data_structs.data_structs import Node, WeaviateIndexStruct
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.response.builder import ResponseBuilder
from gpt_index.indices.utils import truncate_text
from gpt_index.readers.weaviate.data_structs import WeaviateNode


class GPTWeaviateIndexQuery(BaseGPTIndexQuery[WeaviateIndexStruct]):
    """Base vector store query."""

    def __init__(
        self,
        index_struct: WeaviateIndexStruct,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: Optional[int] = 1,
        weaviate_client: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        self._embed_model = embed_model or OpenAIEmbedding()
        self.similarity_top_k = similarity_top_k
        import_err_msg = (
            "`weaviate` package not found, please run `pip install weaviate-client`"
        )
        try:
            import weaviate  # noqa: F401
            from weaviate import Client  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)
        self.client = cast(Client, weaviate_client)

    def _give_response_for_nodes(
        self, query_str: str, nodes: List[Node], verbose: bool = False
    ) -> str:
        """Give response for nodes."""
        response_builder = ResponseBuilder(
            self._prompt_helper,
            self._llm_predictor,
            self.text_qa_template,
            self.refine_template,
        )
        for node in nodes:
            text = self._get_text_from_node(query_str, node, verbose=verbose)
            response_builder.add_text_chunks([text])
        response = response_builder.get_response(
            query_str, verbose=verbose, mode=self._response_mode
        )

        return response or ""

    def _get_nodes_for_response(
        self, query_str: str, verbose: bool = False
    ) -> Tuple[List[str], List[Node]]:
        """Get nodes for response."""
        query_embedding = self._embed_model.get_query_embedding(query_str)
        nodes = WeaviateNode.to_gpt_index_list(
            self.client,
            self._index_struct.get_class_prefix(),
            vector=query_embedding,
            object_limit=self.similarity_top_k,
        )
        nodes = nodes[: self.similarity_top_k]
        node_idxs = [str(i) for i in range(len(nodes))]
        return node_idxs, nodes

    def get_nodes_for_response(
        self, query_str: str, verbose: bool = False
    ) -> Tuple[List[str], List[Node]]:
        """Get nodes for response."""
        ids, nodes = self._get_nodes_for_response(query_str, verbose=verbose)
        filtered_ids = []
        filtered_nodes = []
        for id, node in zip(ids, nodes):
            if self._should_use_node(node):
                filtered_ids.append(id)
                filtered_nodes.append(node)
        return filtered_ids, filtered_nodes

    def _query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        node_idxs, top_k_nodes = self.get_nodes_for_response(query_str, verbose=verbose)
        # print verbose output
        if verbose:
            fmt_txts = []
            for node_idx, node in zip(node_idxs, top_k_nodes):
                fmt_txt = f"> [Node {node_idx}] {truncate_text(node.get_text(), 100)}"
                fmt_txts.append(fmt_txt)
            top_k_node_text = "\n".join(fmt_txts)
            print(f"> Top {len(top_k_nodes)} nodes:\n{top_k_node_text}")
        return self._give_response_for_nodes(query_str, top_k_nodes, verbose=verbose)
