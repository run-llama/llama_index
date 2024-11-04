"""KDB.AI vector store index.

An index that is built within KDB.AI.

"""

import logging
from typing import Any, List, Callable, Optional

import pandas as pd

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.kdbai.utils import (
    default_sparse_encoder,
)

DEFAULT_COLUMN_NAMES = ["document_id", "text", "embeddings"]

DEFAULT_BATCH_SIZE = 100


# INITIALISE LOGGER AND SET FORMAT
logger = logging.getLogger(__name__)


class KDBAIVectorStore(BasePydanticVectorStore):
    """The KDBAI Vector Store.

    In this vector store we store the text, its embedding and
    its metadata in a KDBAI vector store table. This implementation
    allows the use of an already existing table.

    Args:
        table kdbai.Table: The KDB.AI table to use as storage.
        batch (int, optional): batch size to insert data.
            Default is 100.

    Returns:
        KDBAIVectorStore: Vectorstore that supports add and query.
    """

    stores_text: bool = True
    flat_metadata: bool = True

    hybrid_search: bool = False
    batch_size: int

    _table: Any = PrivateAttr()
    _sparse_encoder: Optional[Callable] = PrivateAttr()

    def __init__(
        self,
        table: Any = None,
        hybrid_search: bool = False,
        sparse_encoder: Optional[Callable] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        try:
            import kdbai_client as kdbai

            logger.info("KDBAI client version: " + kdbai.__version__)

        except ImportError:
            raise ValueError(
                "Could not import kdbai_client package."
                "Please add it to the dependencies."
            )

        super().__init__(batch_size=batch_size, hybrid_search=hybrid_search)

        if table is None:
            raise ValueError("Must provide an existing KDB.AI table.")
        else:
            self._table = table

        if hybrid_search:
            if sparse_encoder is None:
                self._sparse_encoder = default_sparse_encoder
            else:
                self._sparse_encoder = sparse_encoder

    @property
    def client(self) -> Any:
        """Return KDB.AI client."""
        return self._table

    @classmethod
    def class_name(cls) -> str:
        return "KDBAIVectorStore"

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to the KDBAI Vector Store.

        Args:
            nodes (List[BaseNode]): List of nodes to be added.

        Returns:
            List[str]: List of document IDs that were added.
        """
        try:
            import kdbai_client as kdbai

            logger.info("KDBAI client version: " + kdbai.__version__)

        except ImportError:
            raise ValueError(
                "Could not import kdbai_client package."
                "Please add it to the dependencies."
            )

        df = pd.DataFrame()
        docs = []

        schema = self._table.schema

        if self.hybrid_search:
            schema = [item for item in schema if item["name"] != "sparseVectors"]

        try:
            for node in nodes:
                doc = {
                    "document_id": node.node_id.encode("utf-8"),
                    "text": node.text.encode("utf-8"),
                    "embeddings": node.embedding,
                }

                if self.hybrid_search:
                    doc["sparseVectors"] = self._sparse_encoder(node.get_content())

                # handle metadata columns
                if len(schema) > len(DEFAULT_COLUMN_NAMES):
                    for column in [
                        item
                        for item in schema
                        if item["name"] not in DEFAULT_COLUMN_NAMES
                    ]:
                        try:
                            doc[column["name"]] = node.metadata[column["name"]]
                        except Exception as e:
                            logger.error(
                                f"Error writing column {column['name']} as type {column['type']}: {e}."
                            )

                docs.append(doc)

            df = pd.DataFrame(docs)
            for i in range((len(df) - 1) // self.batch_size + 1):
                batch = df.iloc[i * self.batch_size : (i + 1) * self.batch_size]
                try:
                    self._table.insert(batch)
                    logger.info(f"inserted batch {i}")
                except Exception as e:
                    logger.exception(
                        f"Failed to insert batch {i} of documents into the datastore: {e}"
                    )

            return [x.decode("utf-8") for x in df["document_id"].tolist()]

        except Exception as e:
            logger.error(f"Error preparing data for KDB.AI: {e}.")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        try:
            import kdbai_client as kdbai

            logger.info("KDBAI client version: " + kdbai.__version__)

        except ImportError:
            raise ValueError(
                "Could not import kdbai_client package."
                "Please add it to the dependencies."
            )

        if query.alpha:
            raise ValueError(
                "Could not run hybrid search. "
                "Please remove alpha and provide KDBAI weights for the two indexes though the vector_store_kwargs."
            )

        if query.filters:
            filter = query.filters
            if kwargs.get("filter"):
                filter.extend(kwargs.pop("filter"))
            kwargs["filter"] = filter

        if kwargs.get("index"):
            index = kwargs.pop("index")
            if self.hybrid_search:
                indexSparse = kwargs.pop("indexSparse", None)
                indexWeight = kwargs.pop("indexWeight", None)
                indexSparseWeight = kwargs.pop("indexSparseWeight", None)
                if not all([indexSparse, indexWeight, indexSparseWeight]):
                    raise ValueError(
                        "Could not run hybrid search. "
                        "Please provide KDBAI sparse index name and weights."
                    )
        else:
            raise ValueError(
                "Could not run the search. " "Please provide KDBAI index name."
            )

        if self.hybrid_search:
            sparse_vectors = [self._sparse_encoder(query.query_str)]

            qry = {index: [query.query_embedding], indexSparse: sparse_vectors}

            index_params = {
                index: {"weight": indexWeight},
                indexSparse: {"weight": indexSparseWeight},
            }

            results = self._table.search(
                vectors=qry,
                index_params=index_params,
                n=query.similarity_top_k,
                **kwargs,
            )[0]
        else:
            results = self._table.search(
                vectors={index: [query.query_embedding]},
                n=query.similarity_top_k,
                **kwargs,
            )[0]

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        for result in results.to_dict(orient="records"):
            metadata = {x: result[x] for x in result if x not in DEFAULT_COLUMN_NAMES}
            node = TextNode(
                text=result["text"], id_=result["document_id"], metadata=metadata
            )
            top_k_ids.append(result["document_id"])
            top_k_nodes.append(node)
            top_k_scores.append(result["__nn_distance"])

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    def delete(self, **delete_kwargs: Any) -> None:
        raise Exception("Not implemented.")
