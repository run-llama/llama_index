"""Document summary index.

A data structure where LlamaIndex stores the summary per document, and maps
the summary to the underlying Nodes.
This summary can be used for retrieval.

"""
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Union, cast

from llama_index.data_structs.document_summary import IndexDocumentSummary
from llama_index.indices.base import BaseIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.utils import embed_nodes
from llama_index.response.schema import Response
from llama_index.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.schema import (
    BaseNode,
    NodeRelationship,
    NodeWithScore,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.storage.docstore.types import RefDocInfo
from llama_index.storage.storage_context import StorageContext
from llama_index.utils import get_tqdm_iterable
from llama_index.vector_stores.types import VectorStore

logger = logging.getLogger(__name__)


DEFAULT_SUMMARY_QUERY = (
    "Describe what the provided text is about. "
    "Also describe some of the questions that this text can answer. "
)


class DocumentSummaryRetrieverMode(str, Enum):
    EMBEDDING = "embedding"
    LLM = "llm"


_RetrieverMode = DocumentSummaryRetrieverMode


class DocumentSummaryIndex(BaseIndex[IndexDocumentSummary]):
    """Document Summary Index.

    Args:
        response_synthesizer (BaseSynthesizer): A response synthesizer for generating
            summaries.
        summary_query (str): The query to use to generate the summary for each document.
        show_progress (bool): Whether to show tqdm progress bars.
            Defaults to False.
        embed_summaries (bool): Whether to embed the summaries.
            This is required for running the default embedding-based retriever.
            Defaults to True.

    """

    index_struct_cls = IndexDocumentSummary

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[IndexDocumentSummary] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        summary_query: str = DEFAULT_SUMMARY_QUERY,
        show_progress: bool = False,
        embed_summaries: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(
            service_context=service_context, response_mode=ResponseMode.TREE_SUMMARIZE
        )
        self._summary_query = summary_query
        self._embed_summaries = embed_summaries

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs,
        )

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    def as_retriever(
        self,
        retriever_mode: Union[str, _RetrieverMode] = _RetrieverMode.EMBEDDING,
        **kwargs: Any,
    ) -> BaseRetriever:
        """Get retriever.

        Args:
            retriever_mode (Union[str, DocumentSummaryRetrieverMode]): A retriever mode.
                Defaults to DocumentSummaryRetrieverMode.EMBEDDING.

        """
        from llama_index.indices.document_summary.retrievers import (
            DocumentSummaryIndexEmbeddingRetriever,
            DocumentSummaryIndexLLMRetriever,
        )

        LLMRetriever = DocumentSummaryIndexLLMRetriever
        EmbeddingRetriever = DocumentSummaryIndexEmbeddingRetriever

        if retriever_mode == _RetrieverMode.EMBEDDING:
            if not self._embed_summaries:
                raise ValueError(
                    "Cannot use embedding retriever if embed_summaries is False"
                )

            if "service_context" not in kwargs:
                kwargs["service_context"] = self._service_context
            return EmbeddingRetriever(self, **kwargs)
        if retriever_mode == _RetrieverMode.LLM:
            return LLMRetriever(self, **kwargs)
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
        return self.docstore.get_node(summary_id).get_content()

    def _add_nodes_to_index(
        self,
        index_struct: IndexDocumentSummary,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
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
        items = doc_id_to_nodes.items()
        iterable_with_progress = get_tqdm_iterable(
            items, show_progress, "Summarizing documents"
        )

        for doc_id, nodes in iterable_with_progress:
            print(f"current doc id: {doc_id}")
            nodes_with_scores = [NodeWithScore(node=n) for n in nodes]
            # get the summary for each doc_id
            summary_response = self._response_synthesizer.synthesize(
                query=self._summary_query,
                nodes=nodes_with_scores,
            )
            summary_response = cast(Response, summary_response)
            summary_node_dict[doc_id] = TextNode(
                text=summary_response.response,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)
                },
            )
            self.docstore.add_documents([summary_node_dict[doc_id]])
            logger.info(
                f"> Generated summary for doc {doc_id}: " f"{summary_response.response}"
            )

        for doc_id, nodes in doc_id_to_nodes.items():
            index_struct.add_summary_and_nodes(summary_node_dict[doc_id], nodes)

        if self._embed_summaries:
            embed_model = self._service_context.embed_model
            summary_nodes = list(summary_node_dict.values())
            id_to_embed_map = embed_nodes(
                summary_nodes, embed_model, show_progress=show_progress
            )

            summary_nodes_with_embedding = []
            for node in summary_nodes:
                node_with_embedding = node.copy()
                node_with_embedding.embedding = id_to_embed_map[node.node_id]
                summary_nodes_with_embedding.append(node_with_embedding)

            self._vector_store.add(summary_nodes_with_embedding)

    def _build_index_from_nodes(
        self, nodes: Sequence[BaseNode]
    ) -> IndexDocumentSummary:
        """Build index from nodes."""
        # first get doc_id to nodes_dict, generate a summary for each doc_id,
        # then build the index struct
        index_struct = IndexDocumentSummary()
        self._add_nodes_to_index(index_struct, nodes, self._show_progress)
        return index_struct

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_nodes_to_index(self._index_struct, nodes)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        if node_id not in self._index_struct.doc_id_to_summary_id:
            raise ValueError(f"node_id {node_id} not in index")
        summary_id = self._index_struct.doc_id_to_summary_id[node_id]

        # delete summary node from docstore
        self.docstore.delete_document(summary_id)

        # delete from index struct
        self._index_struct.delete(node_id)

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


# legacy
GPTDocumentSummaryIndex = DocumentSummaryIndex
