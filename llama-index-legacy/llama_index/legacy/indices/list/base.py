"""Summary index.

A simple data structure where LlamaIndex iterates through document chunks
in sequence in order to answer a given query.

"""

from enum import Enum
from typing import Any, Dict, Optional, Sequence, Union

from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.legacy.data_structs.data_structs import IndexList
from llama_index.legacy.indices.base import BaseIndex
from llama_index.legacy.schema import BaseNode, IndexNode
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.storage.docstore.types import RefDocInfo
from llama_index.legacy.utils import get_tqdm_iterable


class ListRetrieverMode(str, Enum):
    DEFAULT = "default"
    EMBEDDING = "embedding"
    LLM = "llm"


class SummaryIndex(BaseIndex[IndexList]):
    """Summary Index.

    The summary index is a simple data structure where nodes are stored in
    a sequence. During index construction, the document texts are
    chunked up, converted to nodes, and stored in a list.

    During query time, the summary index iterates through the nodes
    with some optional filter parameters, and synthesizes an
    answer from all the nodes.

    Args:
        text_qa_template (Optional[BasePromptTemplate]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
            NOTE: this is a deprecated field.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

    """

    index_struct_cls = IndexList

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[IndexList] = None,
        service_context: Optional[ServiceContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            show_progress=show_progress,
            objects=objects,
            **kwargs,
        )

    def as_retriever(
        self,
        retriever_mode: Union[str, ListRetrieverMode] = ListRetrieverMode.DEFAULT,
        **kwargs: Any,
    ) -> BaseRetriever:
        from llama_index.legacy.indices.list.retrievers import (
            SummaryIndexEmbeddingRetriever,
            SummaryIndexLLMRetriever,
            SummaryIndexRetriever,
        )

        if retriever_mode == ListRetrieverMode.DEFAULT:
            return SummaryIndexRetriever(self, object_map=self._object_map, **kwargs)
        elif retriever_mode == ListRetrieverMode.EMBEDDING:
            return SummaryIndexEmbeddingRetriever(
                self, object_map=self._object_map, **kwargs
            )
        elif retriever_mode == ListRetrieverMode.LLM:
            return SummaryIndexLLMRetriever(self, object_map=self._object_map, **kwargs)
        else:
            raise ValueError(f"Unknown retriever mode: {retriever_mode}")

    def _build_index_from_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False
    ) -> IndexList:
        """Build the index from documents.

        Args:
            documents (List[BaseDocument]): A list of documents.

        Returns:
            IndexList: The created summary index.
        """
        index_struct = IndexList()
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Processing nodes"
        )
        for n in nodes_with_progress:
            index_struct.add_node(n)
        return index_struct

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        for n in nodes:
            self._index_struct.add_node(n)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        cur_node_ids = self._index_struct.nodes
        cur_nodes = self._docstore.get_nodes(cur_node_ids)
        nodes_to_keep = [n for n in cur_nodes if n.node_id != node_id]
        self._index_struct.nodes = [n.node_id for n in nodes_to_keep]

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        node_doc_ids = self._index_struct.nodes
        nodes = self.docstore.get_nodes(node_doc_ids)

        all_ref_doc_info = {}
        for node in nodes:
            ref_node = node.source_node
            if not ref_node:
                continue

            ref_doc_info = self.docstore.get_ref_doc_info(ref_node.node_id)
            if not ref_doc_info:
                continue

            all_ref_doc_info[ref_node.node_id] = ref_doc_info
        return all_ref_doc_info


# Legacy
GPTListIndex = SummaryIndex

# New name
ListIndex = SummaryIndex
