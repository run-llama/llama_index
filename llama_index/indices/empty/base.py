"""Empty index.

An index that doesn't contain any documents. Can only be used for
pure LLM calls.

"""

from typing import Any, Dict, Optional, Sequence

from llama_index.data_structs.data_structs import EmptyIndex
from llama_index.data_structs.node import Node
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.service_context import ServiceContext
from llama_index.storage.docstore.types import RefDocInfo


class GPTEmptyIndex(BaseGPTIndex[EmptyIndex]):
    """GPT Empty Index.

    An index that doesn't contain any documents. Used for
    pure LLM calls.
    NOTE: this exists because an empty index it allows certain properties,
    such as the ability to be composed with other indices + token
    counting + others.

    """

    index_struct_cls = EmptyIndex

    def __init__(
        self,
        index_struct: Optional[EmptyIndex] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            nodes=None,
            index_struct=index_struct or EmptyIndex(),
            service_context=service_context,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        # NOTE: lazy import
        from llama_index.indices.empty.retrievers import EmptyIndexRetriever

        return EmptyIndexRetriever(self)

    def _build_index_from_nodes(self, nodes: Sequence[Node]) -> EmptyIndex:
        """Build the index from documents.

        Args:
            documents (List[BaseDocument]): A list of documents.

        Returns:
            IndexList: The created list index.
        """
        del nodes  # Unused
        index_struct = EmptyIndex()
        return index_struct

    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert a document."""
        del nodes  # Unused
        raise NotImplementedError("Cannot insert into an empty index.")

    def _delete_node(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError("Cannot delete from an empty index.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        raise NotImplementedError("ref_doc_info not supported for an empty index.")
