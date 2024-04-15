from typing import Dict, Optional

from llama_index.core.vector_stores.types import (
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    VectorStoreQuery,
)

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import IndexNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.retrievers import BaseRetriever

from llama_index.core.readers import Document

from dataclasses import dataclass
from typing import List

from llama_index.packs.docugami_kg_rag.config import (
    RETRIEVER_K,
    FULL_DOC_SUMMARY_ID_KEY,
    SOURCE_KEY,
    PARENT_DOC_ID_KEY,
    EMBEDDINGS,
)
from llama_index.core import QueryBundle

from llama_index.core.schema import NodeWithScore

DOCUMENT_SUMMARY_TEMPLATE: str = """
--------------------------------
**** DOCUMENT NAME: {doc_name}

**** DOCUMENT SUMMARY:
{summary}

**** RELEVANT FRAGMENTS:
{fragments}
--------------------------------
"""


@dataclass
class FusedDocumentElements:
    score: float
    summary: str
    fragments: List[str]
    source: str


class FusedSummaryRetriever(BaseRetriever):
    """
    Retrieves a fused document that includes pre-calculated summaries.

    - Full document summaries are included in the fused document to give
      broader context to the LLM, which may not be in the retrieved chunks

    - Chunk summaries are using to improve retrieval, i.e. "big-to-small"
      retrieval which is a common use case with the multi-vector retriever
    """

    vectorstore: ChromaVectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors."""

    full_doc_summary_store: Dict[str, Document]
    """The storage layer for the parent document summaries."""

    parent_doc_store: Dict[str, Document]
    """The storage layer for the parent (original) docs for summaries in
    the vector store."""

    parent_id_key: str = PARENT_DOC_ID_KEY
    """Metadata key for parent doc ID (maps chunk summaries in the vector
    store to parent docs)."""

    full_doc_summary_id_key: str = FULL_DOC_SUMMARY_ID_KEY
    """Metadata key for full doc summary ID (maps chunk summaries in the
    vector store to full doc summaries)."""

    source_key: str = SOURCE_KEY
    """Metadata key for source document of chunks."""

    search_type: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT
    """Type of search to perform (similarity (default)/ mmr / etc.)"""

    def __init__(
        self,
        vectorstore: ChromaVectorStore,
        full_doc_summary_store: Dict[str, Document],
        parent_doc_store: Dict[str, Document],
        search_type: VectorStoreQueryMode,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[Dict] = None,
        objects: Optional[List[IndexNode]] = None,
        verbose: bool = False,
    ):
        super().__init__(
            callback_manager,
            object_map,
            objects,
            verbose,
        )

        self.vectorstore = vectorstore
        self.full_doc_summary_store = full_doc_summary_store
        self.parent_doc_store = parent_doc_store
        self.search_type = search_type

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        query = VectorStoreQuery(
            query_embedding=EMBEDDINGS.get_text_embedding(query_bundle.query_str),
            similarity_top_k=RETRIEVER_K,
            query_str=query_bundle.query_str,
            mode=self.search_type,
        )

        query_result: VectorStoreQueryResult = self.vectorstore.query(query)
        fused_doc_elements: Dict[str, FusedDocumentElements] = {}

        for i in range(RETRIEVER_K):
            node = query_result.nodes[i]

            parent_id = node.metadata.get(self.parent_id_key)
            full_doc_summary_id = node.metadata.get(self.full_doc_summary_id_key)

            if parent_id and full_doc_summary_id:
                parent_in_store = self.parent_doc_store.get(parent_id)
                full_doc_summary_in_store = self.full_doc_summary_store.get(
                    full_doc_summary_id
                )
                if parent_in_store and full_doc_summary_in_store:
                    parent: Document = parent_in_store
                    full_doc_summary: str = full_doc_summary_in_store

                else:
                    raise Exception(
                        f"No parent or full doc summary found for retrieved doc {node},"
                        "please pre-load parent and full doc summaries."
                    )

                source = node.metadata.get(self.source_key)
                if not source:
                    raise Exception(
                        f"No source doc name found in metadata for: {node}."
                    )

                if full_doc_summary_id not in fused_doc_elements:
                    # Init fused parent with information from most relevant sub-doc
                    fused_doc_elements[full_doc_summary_id] = FusedDocumentElements(
                        score=query_result.similarities[i],
                        summary=full_doc_summary,
                        fragments=[parent.text],
                        source=source,
                    )
                else:
                    fused_doc_elements[full_doc_summary_id].fragments.append(
                        parent.text
                    )

        fused_docs: List[NodeWithScore] = []
        for element in sorted(fused_doc_elements.values(), key=lambda x: x.score):
            fragments_str = "\n\n".join([d.strip() for d in element.fragments])
            fused_doc = Document(
                text=DOCUMENT_SUMMARY_TEMPLATE.format(
                    doc_name=element.source,
                    summary=element.summary,
                    fragments=fragments_str,
                )
            )

            fused_docs.append(NodeWithScore(node=fused_doc, score=element.score))

        return fused_docs
