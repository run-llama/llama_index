import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from pydantic import Field

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores.types import (NodeWithEmbedding, VectorStore,
                                             VectorStoreQuery,
                                             VectorStoreQueryResult)
from llama_index.vector_stores.utils import (metadata_dict_to_node,
                                             node_to_metadata_dict)

logger = logging.getLogger(__name__)


class DocArrayVectorStore(VectorStore, ABC):
    """DocArray Vector Store Base Class.


    This is an abstract base class for creating a DocArray vector store.
    The subclasses should implement _init_index and _find_docs_to_be_removed methods.
    """

    from docarray import BaseDoc, DocList
    from docarray.index import HnswDocumentIndex, InMemoryExactNNIndex

    # will get initialized by child classes
    _index: Any
    _schema: Type["BaseDoc"]
    _ref_docs: Dict[str, List[str]]

    stores_text: bool = True

    def _update_ref_docs(self, docs: DocList) -> None:
        pass

    @abstractmethod
    def _init_index(
        self, **kwargs: Any
    ) -> Union[HnswDocumentIndex, InMemoryExactNNIndex]:
        """Initializes the index.

        This method should be overridden by the subclasses.
        """
        pass

    @abstractmethod
    def _find_docs_to_be_removed(self, doc_id: str) -> List[str]:
        """Finds the documents to be removed from the vector store.

        Args:
            doc_id (str): Document ID that should be removed.

        Returns:
            List[str]: List of document IDs to be removed.

        This is an abstract method and needs to be implemented in any concrete subclass.
        """
        pass

    @property
    def client(self) -> Any:
        """Get client."""
        return None

    def num_docs(self) -> int:
        """Retrieves the number of documents in the index.

        Returns:
            int: The number of documents in the index.
        """
        return self._index.num_docs()

    @staticmethod
    def _get_schema(**embeddings_params: Any) -> Type["BaseDoc"]:
        """Fetches the schema for DocArray indices.

        Args:
            **embeddings_params: Variable length argument list for the embedding.

        Returns:
            DocArraySchema: Schema for a DocArray index.
        """

        from docarray import BaseDoc
        from docarray.typing import ID, NdArray

        class DocArraySchema(BaseDoc):
            id: Optional[ID] = None
            text: Optional[str] = None
            metadata: Optional[dict] = None
            embedding: NdArray = Field(**embeddings_params)

        return DocArraySchema

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        """Adds embedding results to the vector store.

        Args:
            embedding_results (List[NodeWithEmbedding]): List of nodes
            with corresponding embeddings.

        Returns:
            List[str]: List of document IDs added to the vector store.
        """
        from docarray import DocList

        # check to see if empty document list was passed
        if len(embedding_results) == 0:
            return []

        docs = DocList[self._schema](  # type: ignore[name-defined]
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
        """Deletes a document from the vector store.

        Args:
            doc_id (str): Document ID to be deleted.
            **delete_kwargs (Any): Additional arguments to pass to the delete method.
        """
        docs_to_be_removed = self._find_docs_to_be_removed(doc_id)
        if not docs_to_be_removed:
            logger.warning(f"Document with doc_id {doc_id} not found")
            return

        del self._index[docs_to_be_removed]
        logger.info(f"Deleted {len(docs_to_be_removed)} documents from the index")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Queries the vector store and retrieves the results.

        Args:
            query (VectorStoreQuery): Query for the vector store.

        Returns:
            VectorStoreQueryResult: Result of the query from vector store.
        """
        if query.filters:
            # only for ExactMatchFilters
            filter_query = {
                "metadata__" + filter.key: {"$eq": filter.value}
                for filter in query.filters.filters
            }
            query = (
                self._index.build_query()  # get empty query object
                .find(
                    query=self._schema(embedding=np.array(query.query_embedding)),
                    search_field="embedding",
                    limit=query.similarity_top_k,
                )  # add vector similarity search
                .filter(filter_query=filter_query)  # add filter search
                .build()  # build the query
            )

            # execute the combined query and return the results
            docs, scores = self._index.execute_query(query)
        else:
            docs, scores = self._index.find(
                query=self._schema(embedding=np.array(query.query_embedding)),
                search_field="embedding",
                limit=query.similarity_top_k,
            )
        nodes, ids = [], []
        for doc in docs:
            extra_info, node_info, relationships = metadata_dict_to_node(doc.metadata)
            nodes.append(
                Node(
                    doc_id=doc.id,
                    text=doc.text,
                    extra_info=extra_info,
                    node_info=node_info,
                    relationships=relationships,
                )
            )
            ids.append(doc.id)
        logger.info(f"Found {len(nodes)} results for the query")

        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)
