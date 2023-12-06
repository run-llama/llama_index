import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np

from llama_index.bridge.pydantic import Field
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)


class DocArrayVectorStore(VectorStore, ABC):
    """DocArray Vector Store Base Class.


    This is an abstract base class for creating a DocArray vector store.
    The subclasses should implement _init_index and _find_docs_to_be_removed methods.
    """

    # for mypy. will get initialized by the subclass.
    _index: Any
    _schema: Any
    _ref_docs: Dict[str, List[str]]

    stores_text: bool = True
    flat_metadata: bool = False

    def _update_ref_docs(self, docs) -> None:  # type: ignore[no-untyped-def]
        pass

    @abstractmethod
    def _init_index(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        """Initializes the index.

        This method should be overridden by the subclasses.
        """

    @abstractmethod
    def _find_docs_to_be_removed(self, doc_id: str) -> List[str]:
        """Finds the documents to be removed from the vector store.

        Args:
            doc_id (str): Document ID that should be removed.

        Returns:
            List[str]: List of document IDs to be removed.

        This is an abstract method and needs to be implemented in any concrete subclass.
        """

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
    def _get_schema(**embeddings_params: Any) -> Type:
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
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Adds nodes to the vector store.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings.

        Returns:
            List[str]: List of document IDs added to the vector store.
        """
        from docarray import DocList

        # check to see if empty document list was passed
        if len(nodes) == 0:
            return []

        docs = DocList[self._schema](  # type: ignore[name-defined]
            self._schema(
                id=node.node_id,
                metadata=node_to_metadata_dict(node, flat_metadata=self.flat_metadata),
                text=node.get_content(metadata_mode=MetadataMode.NONE),
                embedding=node.get_embedding(),
            )
            for node in nodes
        )
        self._index.index(docs)
        logger.info(f"Successfully added {len(docs)} documents to the index")
        if self._ref_docs is not None:
            self._update_ref_docs(docs)
        return [doc.id for doc in docs]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Deletes a document from the vector store.

        Args:
            ref_doc_id (str): Document ID to be deleted.
            **delete_kwargs (Any): Additional arguments to pass to the delete method.
        """
        docs_to_be_removed = self._find_docs_to_be_removed(ref_doc_id)
        if not docs_to_be_removed:
            logger.warning(f"Document with doc_id {ref_doc_id} not found")
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
                for filter in query.filters.legacy_filters()
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
            try:
                node = metadata_dict_to_node(doc.metadata)
                node.text = doc.text
            except Exception:
                # TODO: legacy metadata support
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    doc.metadata
                )
                node = TextNode(
                    id_=doc.id,
                    text=doc.text,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            nodes.append(node)
            ids.append(doc.id)
        logger.info(f"Found {len(nodes)} results for the query")

        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)
