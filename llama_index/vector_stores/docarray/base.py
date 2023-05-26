import logging
import os
from typing import Any, List, cast, Optional, Literal, Dict
from abc import ABC, abstractmethod
import json
import numpy as np
from pydantic import Field

from llama_index.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict


logger = logging.getLogger(__name__)


class DocArrayVectorStore(VectorStore, ABC):
    """DocArray Vector Store base class."""

    stores_text: bool = True

    @abstractmethod
    def _init_index(self, **kwargs):
        pass

    @property
    def client(self) -> Any:
        """Get client."""
        return None

    def num_docs(self) -> int:
        return self._index.num_docs()

    @staticmethod
    def _get_schema(**embeddings_params):
        from docarray.typing import NdArray
        from docarray import BaseDoc

        class DocArraySchema(BaseDoc):
            id: Optional[str] = None
            text: Optional[str] = None
            metadata: Optional[dict] = None
            embedding: NdArray = Field(**embeddings_params)

        return DocArraySchema

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        """Add embedding results to vector store."""
        from docarray import DocList
        # check to see if empty document list was passed
        if len(embedding_results) == 0:
            return []

        docs = DocList[self._schema](
            self._schema(
                id=result.id,
                metadata=node_to_metadata_dict(result.node),
                text=result.node.get_text(),
                embedding=result.embedding,
            )
            for result in embedding_results
        )
        self._index.index(docs)
        logger.info(f"Successfully added {len(docs)} documents to the index")
        if self._ref_docs is not None:
            self._update_ref_docs(docs)
        return [doc.id for doc in docs]

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete doc."""
        docs_to_be_removed = self._find_docs_to_be_removed(doc_id)
        print(docs_to_be_removed, 'aaaa')
        print(self._ref_docs)
        if not docs_to_be_removed:
            logger.warning(f"Document with doc_id {doc_id} not found")
            return

        del self._index[docs_to_be_removed]
        logger.info(f"Deleted {len(docs_to_be_removed)} documents from the index")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        if query.filters:
            # only for ExactMatchFilters
            filter_query = {'metadata__' + filter.key: {'$eq': filter.value} for filter in query.filters.filters}
            query = (
                self._index.build_query()  # get empty query object
                .find(query=self._schema(embedding=np.array(query.query_embedding)), search_field='embedding', limit=query.similarity_top_k)  # add vector similarity search
                .filter(filter_query=filter_query)  # add filter search
                .build()  # build the query
            )

            # execute the combined query and return the results
            docs, scores = self._index.execute_query(query)
        else:
            docs, scores = self._index.find(
                query=self._schema(embedding=np.array(query.query_embedding)),
                search_field='embedding',
                limit=query.similarity_top_k,
            )
        nodes = [
            Node(
                doc_id=doc.id,
                text=doc.text,
                embedding=None,
                relationships={
                    DocumentRelationship.SOURCE: doc.metadata['doc_id'],
                },
            )
            for doc in docs
        ]
        ids = [node.doc_id for node in nodes]
        logger.info(f"Found {len(nodes)} results for the query")

        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)
