"""LanceDB vector store."""

import logging
from typing import Any, List, Optional

import numpy as np
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pandas import DataFrame

import lancedb

_logger = logging.getLogger(__name__)


def _to_lance_filter(standard_filters: MetadataFilters) -> Any:
    """Translate standard metadata filters to Lance specific spec."""
    filters = []
    for filter in standard_filters.legacy_filters():
        if isinstance(filter.value, str):
            filters.append(filter.key + ' = "' + filter.value + '"')
        else:
            filters.append(filter.key + " = " + str(filter.value))
    return " AND ".join(filters)


def _to_llama_similarities(results: DataFrame) -> List[float]:
    keys = results.keys()
    normalized_similarities: np.ndarray
    if "score" in keys:
        normalized_similarities = np.exp(results["score"] - np.max(results["score"]))
    elif "_distance" in keys:
        normalized_similarities = np.exp(-results["_distance"])
    else:
        normalized_similarities = np.linspace(1, 0, len(results))
    return normalized_similarities.tolist()


class LanceDBVectorStore(BasePydanticVectorStore):
    """
    The LanceDB Vector Store.

    Stores text and embeddings in LanceDB. The vector store will open an existing
        LanceDB dataset or create the dataset if it does not exist.

    Args:
        uri (str, required): Location where LanceDB will store its files.
        table_name (str, optional): The table name where the embeddings will be stored.
            Defaults to "vectors".
        vector_column_name (str, optional): The vector column name in the table if different from default.
            Defaults to "vector", in keeping with lancedb convention.
        nprobes (int, optional): The number of probes used.
            A higher number makes search more accurate but also slower.
            Defaults to 20.
        refine_factor: (int, optional): Refine the results by reading extra elements
            and re-ranking them in memory.
            Defaults to None

    Raises:
        ImportError: Unable to import `lancedb`.

    Returns:
        LanceDBVectorStore: VectorStore that supports creating LanceDB datasets and
            querying it.

    Examples:
        `pip install llama-index-vector-stores-lancedb`

        ```python
        from llama_index.vector_stores.lancedb import LanceDBVectorStore

        vector_store = LanceDBVectorStore(uri="/tmp/lancedb")
        ```
    """

    stores_text = True
    flat_metadata: bool = True
    _connection: Any = PrivateAttr()
    uri: Optional[str]
    table_name: Optional[str]
    vector_column_name: Optional[str]
    nprobes: Optional[int]
    refine_factor: Optional[int]
    text_key: Optional[str]
    doc_id_key: Optional[str]

    def __init__(
        self,
        uri: Optional[str],
        table_name: str = "vectors",
        vector_column_name: str = "vector",
        nprobes: int = 20,
        refine_factor: Optional[int] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        doc_id_key: str = DEFAULT_DOC_ID_KEY,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self._connection = lancedb.connect(uri)
        super().__init__(
            uri=uri,
            table_name=table_name,
            vector_column_name=vector_column_name,
            nprobes=nprobes,
            refine_factor=refine_factor,
            text_key=text_key,
            doc_id_key=doc_id_key,
            **kwargs,
        )

    @property
    def client(self) -> None:
        """Get client."""
        return self._connection

    @classmethod
    def from_params(
        cls,
        uri: Optional[str],
        table_name: str = "vectors",
        vector_column_name: str = "vector",
        nprobes: int = 20,
        refine_factor: Optional[int] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        doc_id_key: str = DEFAULT_DOC_ID_KEY,
        **kwargs: Any,
    ) -> "LanceDBVectorStore":
        """Create instance from params."""
        _connection_ = cls._connection
        return cls(
            _connection=_connection_,
            uri=uri,
            table_name=table_name,
            vector_column_name=vector_column_name,
            nprobes=nprobes,
            refine_factor=refine_factor,
            text_key=text_key,
            doc_id_key=doc_id_key,
            **kwargs,
        )

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        if not nodes:
            _logger.debug("No nodes to add. Skipping the database operation.")
            return []
        data = []
        ids = []
        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=False, flat_metadata=self.flat_metadata
            )
            append_data = {
                "id": node.node_id,
                "doc_id": node.ref_doc_id,
                "vector": node.get_embedding(),
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                "metadata": metadata,
            }
            data.append(append_data)
            ids.append(node.node_id)

        if self.table_name in self._connection.table_names():
            tbl = self._connection.open_table(self.table_name)
            tbl.add(data)
        else:
            self._connection.create_table(self.table_name, data)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        table = self._connection.open_table(self.table_name)
        table.delete('doc_id = "' + ref_doc_id + '"')

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        if query.filters is not None:
            if "where" in kwargs:
                raise ValueError(
                    "Cannot specify filter via both query and kwargs. "
                    "Use kwargs only for lancedb specific items that are "
                    "not supported via the generic query interface."
                )
            where = _to_lance_filter(query.filters)
        else:
            where = kwargs.pop("where", None)

        table = self._connection.open_table(self.table_name)
        lance_query = (
            table.search(
                query=query.query_embedding,
                vector_column_name=self.vector_column_name,
            )
            .limit(query.similarity_top_k)
            .where(where)
            .nprobes(self.nprobes)
        )

        if self.refine_factor is not None:
            lance_query.refine_factor(self.refine_factor)

        results = lance_query.to_pandas()
        nodes = []
        for _, item in results.iterrows():
            try:
                node = metadata_dict_to_node(item.metadata)
                node.embedding = list(item[self.vector_column_name])
            except Exception:
                # deprecated legacy logic for backward compatibility
                _logger.debug(
                    "Failed to parse Node metadata, fallback to legacy logic."
                )
                if "metadata" in item:
                    metadata, node_info, _relation = legacy_metadata_dict_to_node(
                        item.metadata, text_key=self.text_key
                    )
                else:
                    metadata, node_info = {}, {}
                node = TextNode(
                    text=item[self.text_key] or "",
                    id_=item.id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(
                            node_id=item[self.doc_id_key]
                        ),
                    },
                )

            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=_to_llama_similarities(results),
            ids=results["id"].tolist(),
        )
