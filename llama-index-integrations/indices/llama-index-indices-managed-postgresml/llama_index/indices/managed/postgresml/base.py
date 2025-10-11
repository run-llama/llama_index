"""
Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

import logging
from llama_index.core.async_utils import run_async_tasks
from typing import Any, Dict, Optional, Sequence, Type

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever

from llama_index.core.data_structs.data_structs import IndexDict, IndexStructType
from llama_index.core.indices.managed.base import BaseManagedIndex, IndexType
from llama_index.core.schema import (
    BaseNode,
    Document,
    TextNode,
)
from pgml import Collection, Pipeline, init_logger

init_logger()

_logger = logging.getLogger(__name__)


class PostgresMLIndexStruct(IndexDict):
    """PostgresML Index Struct."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get index struct type."""
        # return IndexStructType.POSTGRESML
        return "POSTGRESML"


class PostgresMLIndex(BaseManagedIndex):
    """
    PostgresML Index.

    The PostgresML index implements a managed index that uses PostgresML as the backend.
    PostgresML performs a lot of the functions in traditional indexes in the backend:
    - breaks down a document into chunks (nodes)
    - Creates the embedding for each chunk (node)
    - Performs the search for the top k most similar nodes to a query
    - Optionally can perform text-generation or chat completion
    """

    def __init__(
        self,
        collection_name: str,
        pipeline_name: Optional[str] = None,
        pipeline_schema: Optional[Dict[str, Any]] = None,
        pgml_database_url: Optional[str] = None,
        show_progress: bool = True,
        upsert_parallel_batches: int = 1,
        nodes: Optional[Sequence[BaseNode]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PostgresML SDK."""
        self.show_progress = show_progress
        self.upsert_parallel_batches = upsert_parallel_batches

        index_struct = PostgresMLIndexStruct(
            index_id=collection_name,
            summary="PostgresML Index",
        )

        super().__init__(
            show_progress=show_progress,
            index_struct=index_struct,
            **kwargs,
        )

        # Create our Collection and Pipeline
        self.collection = Collection(collection_name, pgml_database_url)
        if pipeline_name is None:
            pipeline_name = "v1"
        if pipeline_schema is None:
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

        # We must wrap self.collection.add_pipeline() with this async function
        # This is a limitation of the pyo3 async implementation
        async def add_pipeline():
            await self.collection.add_pipeline(self.pipeline)

        run_async_tasks([add_pipeline()])

        if nodes:
            self._insert(nodes)

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

        args = {"parallel_batches": self.upsert_parallel_batches, **insert_kwargs}

        # We must wrap self.collection.upsert_documents() with this async function
        # This is a limitation of the pyo3 async implementation
        async def upsert_documents():
            await self.collection.upsert_documents(documents, args)

        run_async_tasks([upsert_documents()])

    def add_documents(
        self,
        docs: Sequence[Document],
        **insert_kwargs: Any,
    ) -> None:
        nodes = [TextNode(**doc.dict()) for doc in docs]
        self._insert(nodes, **insert_kwargs)

    def delete_ref_doc(self, ref_doc_id: str) -> None:
        # We must wrap self.collection.delete_documents() with this async function
        # This is a limitation of the pyo3 async implementation
        async def delete_documents():
            await self.collection.delete_documents({"id": {"$eq": ref_doc_id}})

        run_async_tasks([delete_documents()])

    def update_ref_doc(self, document: Document) -> None:
        node = TextNode(**document.dict())
        self._insert([node], merge=True)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a Retriever for this managed index."""
        from llama_index.indices.managed.postgresml.retriever import (
            PostgresMLRetriever,
        )

        return PostgresMLRetriever(self, **kwargs)

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        from llama_index.indices.managed.postgresml.retriever import (
            PostgresMLRetriever,
        )
        from llama_index.indices.managed.postgresml.query import (
            PostgresMLQueryEngine,
        )

        return PostgresMLQueryEngine(PostgresMLRetriever(self, **kwargs), **kwargs)

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        collection_name: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        pipeline_schema: Optional[Dict[str, Any]] = None,
        pgml_database_url: Optional[str] = None,
        show_progress: bool = False,
        upsert_parallel_batches: int = 1,
        **kwargs: Any,
    ) -> IndexType:
        """Build a PostgresML index from a sequence of documents."""
        if collection_name is None:
            raise Exception("collection_name is a required argument")
        nodes = [TextNode(**doc.dict()) for doc in documents]
        return cls(
            collection_name,
            pipeline_name=pipeline_name,
            pipeline_schema=pipeline_schema,
            pgml_database_url=pgml_database_url,
            nodes=nodes,
            show_progress=show_progress,
            upsert_parallel_batches=upsert_parallel_batches,
            **kwargs,
        )
