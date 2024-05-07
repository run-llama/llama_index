"""Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

import logging
import asyncio
from llama_index.core.async_utils import run_async_tasks
from concurrent.futures import ThreadPoolExecutor
from hashlib import blake2b
from typing import Any, Dict, List, Optional, Sequence, Type

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

from pgml import Collection, Pipeline

_logger = logging.getLogger(__name__)


class PostgresMLIndexStruct(IndexDict):
    """PostgresML Index Struct."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get index struct type."""
        return IndexStructType.POSTGRESML


class PostgresMLIndex(BaseManagedIndex):
    """PostgresML Index.

    The PostgresML index implements a managed index that uses PostgresML as the backend.
    PostgresML performs a lot of the functions in traditional indexes in the backend:
    - breaks down a document into chunks (nodes)
    - Creates the embedding for each chunk (node)
    - Performs the search for the top k most similar nodes to a query
    - Optionally can perform text-generation or chat completion

    Args:
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

    """

    def __init__(
        self,
        collection_name: str,
        pipeline_name: Optional[str] = None,
        pipeline_schema: Optional[Dict[str, Any]] = None,
        pgml_database_url: Optional[str] = None,
        show_progress: bool = False,
        nodes: Optional[Sequence[BaseNode]] = None,
        parallelize_ingest: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the PostgresML SDK."""
        self.parallelize_ingest = parallelize_ingest
        index_struct = PostgresMLIndexStruct(
            index_id=collection_name,
            summary="PostgresML Index",
        )

        super().__init__(
            show_progress=show_progress,
            index_struct=index_struct,
            service_context=ServiceContext.from_defaults(
                llm=None, llm_predictor=None, embed_model=None
            ),
            **kwargs,
        )

        # Create our Collection and Pipeline
        self.collection = Collection(collection_name, pgml_database_url)
        if pipeline_name == None:
            pipeline_name = "v1"
        if pipeline_schema == None:
            pipeline_schema = {
                "content": {
                    "splitter": {
                        "model": "recursive_character",
                        "parameters": {"chunk_size": 1500},
                    },
                    "semantic_search": {
                        "model": "intfloat/e5-small-v2",
                        "parameters": {"prompt": "passage: "},
                    },
                }
            }
        self.pipeline = Pipeline(pipeline_name, pipeline_schema)

        async def add_pipeline():
            await self.collection.add_pipeline(self.pipeline)

        run_async_tasks([add_pipeline()])

        if nodes:
            self._insert(nodes)

    # NOTE: Not sure what actually calls this
    def _delete_doc(self, doc_id: str) -> bool:
        """
        Delete a document from the PostgresML Collection.

        Args:
            doc_id (str): ID of the document to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        return True

    def _insert(
        self,
        nodes: Sequence[BaseNode],
        **insert_kwargs: Any,
    ) -> None:
        """Insert a set of documents (each a node)."""
        documents = [
            {
                "id": node.node_id,
                "content": node.get_content(),
                "metadata": node.metadata,
            }
            for node in nodes
        ]

        async def upsert_documents():
            await self.collection.upsert_documents(documents)

        run_async_tasks([upsert_documents()])

    def add_documents(
        self,
        docs: Sequence[Document],
        allow_update: bool = True,
    ) -> None:
        nodes = [
            TextNode(text=doc.get_content(), metadata=doc.metadata) for doc in docs  # type: ignore
        ]
        self._insert(nodes)

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        raise NotImplementedError(
            "PostgresML does not support deleting a reference document"
        )

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        raise NotImplementedError(
            "PostgresML does not support updating a reference document"
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a Retriever for this managed index."""
        from llama_index.indices.managed.postgresml.retriever import (
            PostgresMLRetriever,
        )

        return PostgresMLRetriever(self, **kwargs)

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        from llama_index.indices.managed.postgresml.query import (
            PostgresMLQueryEngine,
        )

        return PostgresMLQueryEngine(self)

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        collection_name: str,
        documents: Sequence[Document],
        pipeline_name: Optional[str] = None,
        pipeline_schema: Optional[Dict[str, Any]] = None,
        pgml_database_url: Optional[str] = None,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        # deprecated
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> IndexType:
        """Build a PostgresML index from a sequence of documents."""
        nodes = [
            TextNode(text=document.get_content(), metadata=document.metadata)  # type: ignore
            for document in documents
        ]
        return cls(
            collection_name,
            nodes=nodes,
            show_progress=show_progress,
            **kwargs,
        )
