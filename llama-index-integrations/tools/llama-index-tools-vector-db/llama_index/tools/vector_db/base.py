"""Vector DB tool spec."""

from typing import List

from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.vector_stores.types import ExactMatchFilter, MetadataFilters


class VectorDBToolSpec(BaseToolSpec):
    """Vector DB tool spec."""

    spec_functions = ["auto_retrieve_fn"]

    def __init__(
        self,
        index: BaseIndex,  # TODO typing
    ) -> None:
        """Initialize with parameters."""
        self._index = index

    def auto_retrieve_fn(
        self,
        query: str,
        top_k: int,
        filter_key_list: List[str],
        filter_value_list: List[str],
    ) -> str:
        """
        Auto retrieval function.

        Performs auto-retrieval from a vector database, and then applies a set of filters.

        Args:
            query (str): The query to search
            top_k (int): The number of results to retrieve
            filter_key_list (List[str]): The list of filter keys
            filter_value_list (List[str]): The list of filter values

        """
        exact_match_filters = [
            ExactMatchFilter(key=k, value=v)
            for k, v in zip(filter_key_list, filter_value_list)
        ]
        retriever = VectorIndexRetriever(
            self._index,
            filters=MetadataFilters(filters=exact_match_filters),
            top_k=top_k,
        )
        query_engine = RetrieverQueryEngine.from_args(retriever)

        response = query_engine.query(query)
        return str(response)


# backwards compatibility
VectorDB = VectorDBToolSpec
