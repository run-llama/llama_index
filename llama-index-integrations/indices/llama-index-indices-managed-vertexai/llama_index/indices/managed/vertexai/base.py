"""Vertex AI Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from hashlib import blake2b
from typing import Any, Dict, List, Optional, Sequence, Type

import requests
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.data_structs.data_structs import IndexDict, IndexStructType
from llama_index.core.indices.managed.base import BaseManagedIndex, IndexType
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.schema import (
    BaseNode,
    Document,
    MetadataMode,
    TextNode,
    TransformComponent,
)
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import Settings
from llama_index.core.storage.storage_context import StorageContext

from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer

_logger = logging.getLogger(__name__)


class VertexAIIndexStruct(IndexDict):
    """Vertex AI Index Struct."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get index struct type."""
        return IndexStructType.VERTEX_AI


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
        location: str,
        corpus_id: str,
        nodes: Optional[Sequence[BaseNode]] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Vertex AI API."""
        corpus_name = (
            f"projects/{project_id}/locations/{location}/ragCorpora/{corpus_id}"
        )
        index_struct = VertexAIIndexStruct(
            index_id=corpus_name,
            summary="Vertex AI Index",
        )

        super().__init__(
            show_progress=show_progress,
            index_struct=index_struct,
            service_context=ServiceContext.from_defaults(
                llm=None, llm_predictor=None, embed_model=None
            ),
            **kwargs,
        )

        # if nodes is specified, consider each node as a single document
        # and use _build_index_from_nodes() to add them to the index
        if nodes is not None:
            self._build_index_from_nodes(nodes)

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a set of documents (each a node)."""

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        docs = [
            Document(
                text=node.get_content(metadata_mode=MetadataMode.NONE),
                metadata=node.metadata,  # type: ignore
                id_=node.id_,  # type: ignore
            )
            for node in nodes
        ]
        self.add_documents(docs)
        return self.index_struct

    def add_documents(
        self,
        docs: Sequence[Document],
    ) -> None:
        nodes = [
            TextNode(text=doc.get_content(), metadata=doc.metadata) for doc in docs  # type: ignore
        ]
        self._insert(nodes)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a Retriever for this managed index."""
        from llama_index.indices.managed.vertexai.retriever import (
            VertexAIRetriever,
        )

        return VertexAIRetriever(self, **kwargs)

    def as_query_engine(
        self, llm: Optional[LLMType] = None, **kwargs: Any
    ) -> BaseQueryEngine:
        if kwargs.get("summary_enabled", True):
            from llama_index.indices.managed.vertexai.query import (
                VertexAIQueryEngine,
            )

            kwargs["summary_enabled"] = True
            retriever = self.as_retriever(**kwargs)
            return VertexAIQueryEngine.from_args(retriever, **kwargs)  # type: ignore
        else:
            from llama_index.core.query_engine.retriever_query_engine import (
                RetrieverQueryEngine,
            )

            llm = (
                resolve_llm(llm, callback_manager=self._callback_manager)
                or Settings.llm
            )

            retriever = self.as_retriever(**kwargs)
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT,
                llm=llm,
            )
            return RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                **kwargs,
            )

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        # deprecated
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> IndexType:
        """Build a Vertex AI index from a sequence of documents."""
        nodes = [
            TextNode(text=document.get_content(), metadata=document.metadata)  # type: ignore
            for document in documents
        ]
        return cls(
            nodes=nodes,
            show_progress=show_progress,
            **kwargs,
        )
