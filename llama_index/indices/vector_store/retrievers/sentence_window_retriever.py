"""Base vector store index query."""
from typing import List

from llama_index.indices.vector_store.sentence_window import SentenceWindowVectorIndex
from llama_index.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.schema import NodeWithScore, MetadataMode
from llama_index.vector_stores.types import VectorStoreQueryResult


class SentenceWindowVectorRetriever(VectorIndexRetriever):
    """Base vector store index query."""

    def _build_node_list_from_query_result(
        self, query_result: VectorStoreQueryResult
    ) -> List[NodeWithScore]:
        nodes_with_scores = super()._build_node_list_from_query_result(query_result)

        # replace node content with window
        for n in nodes_with_scores:
            n.node.set_content(
                n.node.metadata.get(
                    SentenceWindowVectorIndex.window_metadata_key,
                    n.node.get_content(metadata_mode=MetadataMode.NONE),
                )
            )

        return nodes_with_scores
