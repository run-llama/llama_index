"""
Vertex AI Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

import re
from importlib import metadata
from typing import Any, Optional, Sequence

from google.cloud.aiplatform_v1beta1 import ImportRagFilesResponse
from google.cloud.aiplatform import telemetry

import vertexai
from vertexai.preview import rag

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.core.schema import BaseNode, Document


def get_user_agent(module: Optional[str] = None) -> str:
    r"""
    Returns a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.

    Returns:
        Tuple[str, str]

    """
    try:
        llama_index_version = metadata.version("llama-index")
    except metadata.PackageNotFoundError:
        llama_index_version = "0.0.0"
    client_library_version = (
        f"{llama_index_version}-{module}" if module else llama_index_version
    )
    return f"llama-index/{client_library_version}"


class VertexAIIndex(BaseManagedIndex):
    """
    Vertex AI Index.

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
        self._user_agent = get_user_agent("vertexai-rag")

        vertexai.init(project=self.project_id, location=self.location)

        with telemetry.tool_context_manager(self._user_agent):
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

        with telemetry.tool_context_manager(self._user_agent):
            return rag.import_files(
                self.corpus_name,
                paths=uris,
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

        with telemetry.tool_context_manager(self._user_agent):
            rag_file = rag.upload_file(
                corpus_name=self.corpus_name,
                path=file_path,
                display_name=display_name,
                description=description,
                **insert_kwargs,
            )

        return rag_file.name if rag_file else None

    def list_files(self) -> Sequence[str]:
        """List all files in the index."""
        files = []
        with telemetry.tool_context_manager(self._user_agent):
            for file in rag.list_files(corpus_name=self.corpus_name):
                files.append(file.name)
        return files

    def delete_file(self, file_name: str) -> None:
        """Delete file from the index."""
        with telemetry.tool_context_manager(self._user_agent):
            rag.delete_file(name=file_name, corpus_name=self.corpus_name)

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
            self.corpus_name,
            similarity_top_k,
            vector_distance_threshold,
            self._user_agent,
            **kwargs,
        )

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a set of documents (each a node)."""
        raise NotImplementedError("Node insertion is not supported.")

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        if delete_from_docstore:
            with telemetry.tool_context_manager(self._user_agent):
                rag.delete_file(
                    name=ref_doc_id,
                    corpus_name=self.corpus_name,
                )

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document and it's corresponding nodes."""
        raise NotImplementedError("Document update is not supported.")
