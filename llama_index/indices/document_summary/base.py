"""Document summary index.

A data structure where LlamaIndex stores the summary per document, and maps
the summary to the underlying Nodes.
This summary can be used for retrieval.

"""
from enum import Enum
import logging
from collections import defaultdict
from typing import Optional, Sequence, Any, Dict, Union, cast


from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.base import BaseGPTIndex
from llama_index.data_structs.document_summary import IndexDocumentSummary
from llama_index.data_structs.node import Node, DocumentRelationship, NodeWithScore
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.query.response_synthesis import ResponseSynthesizer
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.response.schema import Response
from llama_index.storage.docstore.types import RefDocInfo


logger = logging.getLogger(__name__)


DEFAULT_SUMMARY_QUERY = (
    "Give a concise summary of this document. Also describe some of the questions "
    "that this document can answer. "
)


class DocumentSummaryRetrieverMode(str, Enum):
    DEFAULT = "default"
    EMBEDDING = "embedding"


DSRM = DocumentSummaryRetrieverMode


class GPTDocumentSummaryIndex(BaseGPTIndex[IndexDocumentSummary]):
    """GPT Document Summary Index.

    Args:
        summary_template (Optional[SummaryPrompt]): A Summary Prompt
            (see :ref:`Prompt-Templates`).

    """

    index_struct_cls = IndexDocumentSummary

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[IndexDocumentSummary] = None,
        service_context: Optional[ServiceContext] = None,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        summary_query: str = DEFAULT_SUMMARY_QUERY,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._response_synthesizer = (
            response_synthesizer
            or ResponseSynthesizer.from_args(service_context=service_context)
        )
        self._summary_query = summary_query or "summarize:"
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            **kwargs,
        )

    def as_retriever(
        self,
        retriever_mode: Union[str, DSRM] = DSRM.DEFAULT,
        **kwargs: Any,
    ) -> BaseRetriever:
        """Get retriever.

        Args:
            retriever_mode (Union[str, DocumentSummaryRetrieverMode]): A retriever mode.

        """
        from llama_index.indices.document_summary.retrievers import (
            DocumentSummaryIndexEmbeddingRetriever,
            DocumentSummaryIndexRetriever,
        )

        DSIR = DocumentSummaryIndexRetriever
        DSIER = DocumentSummaryIndexEmbeddingRetriever

        if retriever_mode == DSRM.DEFAULT:
            return DSIR(self, **kwargs)
        elif retriever_mode == DSRM.EMBEDDING:
            return DSIER(self, **kwargs)
        else:
            raise ValueError(f"Unknown retriever mode: {retriever_mode}")

    def get_document_summary(self, doc_id: str) -> str:
        """Get document summary by doc id.

        Args:
            doc_id (str): A document id.

        """
        if doc_id not in self._index_struct.doc_id_to_summary_id:
            raise ValueError(f"doc_id {doc_id} not in index")
        summary_id = self._index_struct.doc_id_to_summary_id[doc_id]
        return self.docstore.get_node(summary_id).get_text()

    def _add_nodes_to_index(
        self, index_struct: IndexDocumentSummary, nodes: Sequence[Node]
    ) -> None:
        """Add nodes to index."""
        doc_id_to_nodes = defaultdict(list)
        for node in nodes:
            if node.ref_doc_id is None:
                raise ValueError(
                    "ref_doc_id of node cannot be None when building a document "
                    "summary index"
                )
            doc_id_to_nodes[node.ref_doc_id].append(node)

        summary_node_dict = {}
        for doc_id, nodes in doc_id_to_nodes.items():
            print(f"current doc id: {doc_id}")
            nodes_with_scores = [NodeWithScore(n) for n in nodes]
            # get the summary for each doc_id
            summary_response = self._response_synthesizer.synthesize(
                query_bundle=QueryBundle(self._summary_query),
                nodes=nodes_with_scores,
            )
            summary_response = cast(Response, summary_response)
            summary_node_dict[doc_id] = Node(
                summary_response.response,
                relationships={DocumentRelationship.SOURCE: doc_id},
            )
            self.docstore.add_documents([summary_node_dict[doc_id]])
            logger.info(
                f"> Generated summary for doc {doc_id}: " f"{summary_response.response}"
            )

        for doc_id, nodes in doc_id_to_nodes.items():
            index_struct.add_summary_and_nodes(summary_node_dict[doc_id], nodes)

    def _build_index_from_nodes(self, nodes: Sequence[Node]) -> IndexDocumentSummary:
        """Build index from nodes."""
        # first get doc_id to nodes_dict, generate a summary for each doc_id,
        # then build the index struct
        index_struct = IndexDocumentSummary()
        self._add_nodes_to_index(index_struct, nodes)
        return index_struct

    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_nodes_to_index(self._index_struct, nodes)

    def _delete_node(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        if doc_id not in self._index_struct.doc_id_to_summary_id:
            raise ValueError(f"doc_id {doc_id} not in index")
        summary_id = self._index_struct.doc_id_to_summary_id[doc_id]

        # delete summary node from docstore
        self.docstore.delete_document(summary_id)

        # delete from index struct
        self._index_struct.delete(doc_id)

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        ref_doc_ids = list(self._index_struct.doc_id_to_summary_id.keys())

        all_ref_doc_info = {}
        for ref_doc_id in ref_doc_ids:
            ref_doc_info = self.docstore.get_ref_doc_info(ref_doc_id)
            if not ref_doc_info:
                continue

            all_ref_doc_info[ref_doc_id] = ref_doc_info
        return all_ref_doc_info
