"""Vertex AI Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

from typing import Any, Optional, Sequence

from google.cloud.aiplatform_v1beta1 import ImportRagFilesResponse
import vertexai
from vertexai.preview import rag

import re

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.managed.base import BaseManagedIndex


class VertexAIIndex(BaseManagedIndex):
    """Vertex AI Index.

    The Vertex AI RAG index implements a managed index that uses Vertex AI as the backend.
    Vertex AI performs a lot of the functions in traditional indexes in the backend:
    - breaks down a document into chunks (nodes)
    - Creates the embedding for each chunk (node)
    - Performs the search for the top k most similar nodes to a query
    - Optionally can perform summarization of the top k nodes

    Args:
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
    """

    def __init__(
        self,
        project_id: str,
        location: Optional[str] = None,
        corpus_id: Optional[str] = None,
        corpus_display_name: Optional[str] = None,
        corpus_description: Optional[str] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Vertex AI API."""

        if corpus_id and (corpus_display_name or corpus_description):
            raise ValueError(
                "Cannot specify both corpus_id and corpus_display_name or corpus_description"
            )

        self.project_id = project_id
        self.location = location
        self.show_progress = show_progress

        vertexai.init(project=self.project_id, location=self.location)

        # If a corpus is not specified, create a new one.
        if corpus_id:
            # Make sure corpus exists
            self.corpus_name = rag.get_corpus(name=corpus_id).name
        else:
            self.corpus_name = rag.create_corpus(
                display_name=corpus_display_name, description=corpus_description
            ).name

    def import_files(
        self,
        uris: Sequence[str],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> ImportRagFilesResponse:
        """Import Google Cloud Storage or Google Drive files into the index."""
        # Convert https://storage.googleapis.com URLs to gs:// format
        uris = [
            re.sub(r"^https://storage\.googleapis\.com/", "gs://", uri) for uri in uris
        ]

        return rag.import_files(
            corpus=self.corpus_name,
            uris=uris,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            timeout=timeout,
            **kwargs,
        )

    def insert_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        **insert_kwargs: Any,
    ) -> Optional[str]:
        """Insert a local file into the index."""
        if metadata:
            display_name = metadata.get("display_name")
            description = metadata.get("description")

        rag_file = rag.upload_file(
            corpus_name=self.corpus_name,
            path=file_path,
            display_name=display_name,
            description=description,
            **insert_kwargs,
        )

        return rag_file.name if rag_file else None

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        from llama_index.core.query_engine.retriever_query_engine import (
            RetrieverQueryEngine,
        )

        kwargs["retriever"] = self.as_retriever(**kwargs)
        return RetrieverQueryEngine.from_args(**kwargs)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a Retriever for this managed index."""
        from llama_index.indices.managed.vertexai.retriever import (
            VertexAIRetriever,
        )

        similarity_top_k = kwargs.pop("similarity_top_k", None)
        vector_distance_threshold = kwargs.pop("vector_distance_threshold", None)

        return VertexAIRetriever(
            self.corpus_name, similarity_top_k, vector_distance_threshold, **kwargs
        )
