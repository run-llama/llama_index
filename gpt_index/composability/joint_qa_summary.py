"""Joint QA Summary graph."""


from typing import Sequence, Optional

from gpt_index.indices.service_context import ServiceContext
from gpt_index.composability import ComposableGraph
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.indices.vector_store.vector_indices import GPTSimpleVectorIndex
from gpt_index.readers.schema.base import Document

DEFAULT_SUMMARY_TEXT = "Use this index for summarization queries"
DEFAULT_QA_TEXT = (
    "Use this index for queries that require retrieval of specific "
    "context from documents."
)


class QASummaryGraphBuilder:
    """Joint QA Summary graph builder."""

    def build_graph_from_documents(
        self,
        documents: Sequence[Document],
        verbose: bool = False,
        summary_text: str = DEFAULT_SUMMARY_TEXT,
        qa_text: str = DEFAULT_QA_TEXT,
        service_context: Optional[ServiceContext] = None,
    ) -> "ComposableGraph":
        """Build graph from index."""

        # used for QA
        vector_index = GPTSimpleVectorIndex.from_documents(
            documents,
            service_context=service_context,
        )
        # used for summarization
        list_index = GPTListIndex.from_documents(
            documents, service_context=service_context
        )

        vector_index.index_struct.summary = qa_text
        list_index.index_struct.summary = summary_text

        graph = ComposableGraph.from_indices(
            GPTTreeIndex, [vector_index, list_index], service_context=service_context
        )

        return graph
