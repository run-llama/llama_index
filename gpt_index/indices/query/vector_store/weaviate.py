"""Weaviate vector store index query."""


import logging
from typing import Any, List, Optional, cast

from gpt_index.data_structs.data_structs import Node, WeaviateIndexStruct
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
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
        super().__init__(index_struct=index_struct, embed_model=embed_model, **kwargs)
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

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        query_embedding = self._embed_model.get_agg_embedding_from_queries(
            query_bundle.embedding_strs
        )
        nodes = WeaviateNode.to_gpt_index_list(
            self.client,
            self._index_struct.get_class_prefix(),
            vector=query_embedding,
            object_limit=self.similarity_top_k,
        )
        nodes = nodes[: self.similarity_top_k]
        node_idxs = [str(i) for i in range(len(nodes))]

        if logging.getLogger(__name__).getEffectiveLevel() == logging.DEBUG:
            fmt_txts = []
            for node_idx, node in zip(node_idxs, nodes):
                fmt_txt = f"> [Node {node_idx}] {truncate_text(node.get_text(), 100)}"
                fmt_txts.append(fmt_txt)
            top_k_node_text = "\n".join(fmt_txts)
            logging.debug(f"> Top {len(nodes)} nodes:\n{top_k_node_text}")
        return nodes
