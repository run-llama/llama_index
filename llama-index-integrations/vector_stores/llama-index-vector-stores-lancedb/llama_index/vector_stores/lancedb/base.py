"""LanceDB vector store."""

import os
import logging
from typing import Any, List, Optional
import warnings

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
    FilterOperator,
    FilterCondition,
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

from .util import sql_operator_mapper, reranker_mapper


def _to_lance_filter(standard_filters: MetadataFilters, metadata_keys: list) -> Any:
    """Translate standard metadata filters to Lance specific spec."""
    filters = []
    for filter in standard_filters.filters:
        key = filter.key
        if filter.key in metadata_keys:
            key = f"metadata.{filter.key}"
        if (
            filter.operator == FilterOperator.TEXT_MATCH
            or filter.operator == FilterOperator.NE
        ):
            filters.append(
                key + sql_operator_mapper[filter.operator] + f"'%{filter.value}%'"
            )
        if isinstance(filter.value, list):
            val = ",".join(filter.value)
            filters.append(key + sql_operator_mapper[filter.operator] + f"({val})")
        elif isinstance(filter.value, int):
            filters.append(
                key + sql_operator_mapper[filter.operator] + f"{filter.value}"
            )
        else:
            filters.append(
                key + sql_operator_mapper[filter.operator] + f"'{filter.value!s}'"
            )

    if standard_filters.condition == FilterCondition.OR:
        return " OR ".join(filters)
    else:
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
        api_key (str, optional): The API key to use LanceDB cloud.
            Defaults to None. You can also set the `LANCE_API_KEY` environment variable.
        region (str, optional): The region to use for your LanceDB cloud db.
            Defaults to None.

    Raises:
        ImportError: Unable to import `lancedb`.

    Returns:
        LanceDBVectorStore: VectorStore that supports creating LanceDB datasets and
            querying it.

    Examples:
        `pip install llama-index-vector-stores-lancedb`

        ```python
        from llama_index.vector_stores.lancedb import LanceDBVectorStore

        vector_store = LanceDBVectorStore()  # native invocation
        ```
    """

    stores_text = True
    flat_metadata: bool = True
    uri: Optional[str]
    vector_column_name: Optional[str]
    nprobes: Optional[int]
    refine_factor: Optional[int]
    text_key: Optional[str]
    doc_id_key: Optional[str]
    api_key: Optional[str]
    region: Optional[str]

    _table_name: Optional[str] = PrivateAttr()
    _connection: Any = PrivateAttr()
    _table: Any = PrivateAttr()
    _metadata_keys: Any = PrivateAttr()

    def __init__(
        self,
        connection: Optional[Any] = None,
        uri: Optional[str] = "/tmp/lancedb",
        table: Optional[Any] = None,
        table_name: Optional[str] = "vectors",
        vector_column_name: str = "vector",
        nprobes: int = 20,
        refine_factor: Optional[int] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        doc_id_key: str = DEFAULT_DOC_ID_KEY,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self._table_name = table_name
        self._metadata_keys = None

        if isinstance(connection, lancedb.db.LanceDBConnection):
            self._connection = connection
        elif isinstance(connection, str):
            raise ValueError(
                "`connection` has to be a lancedb.db.LanceDBConnection object."
            )
        else:
            if api_key is None and os.getenv("LANCE_API_KEY") is None:
                if uri.startswith("db://"):
                    raise ValueError("API key is required for LanceDB cloud.")
                else:
                    self._connection = lancedb.connect(uri)
            else:
                if "db://" not in uri:
                    self._connection = lancedb.connect(uri)
                    warnings.warn(
                        "api key provided with local uri. The data will be stored locally"
                    )
                self._connection = lancedb.connect(
                    uri, api_key=api_key or os.getenv("LANCE_API_KEY"), region=region
                )

        if table:
            assert isinstance(table, lancedb.db.LanceTable)
            self._table = table
            self._table_name = table.name
        else:
            if self._table_exists():
                self._table = self._connection.open_table(table_name)
            else:
                self._table = None

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
    def from_table(cls, table: lancedb.db.LanceTable) -> "LanceDBVectorStore":
        """Create instance from table."""
        if not isinstance(table, lancedb.db.LanceTable):
            raise Exception("argument is not lancdb table instance")
        return cls(table=table)

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

    def _table_exists(self, tbl_name: Optional[str] = None) -> bool:
        return (tbl_name or self._table_name) in self._connection.table_names()

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
            if not self._metadata_keys:
                self._metadata_keys = list(metadata.keys())
            append_data = {
                "id": node.node_id,
                self.doc_id_key: node.ref_doc_id,
                self.vector_column_name: node.get_embedding(),
                self.text_key: node.get_content(metadata_mode=MetadataMode.NONE),
                "metadata": metadata,
            }
            data.append(append_data)
            ids.append(node.node_id)

            if self._table is None:
                self._table = self._connection.create_table(
                    self._table_name, data, mode=add_kwargs.pop("mode", "overwrite")
                )

            self._table.add(data, mode=add_kwargs.pop("mode", "overwrite"))

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        table = self._connection.open_table(self._table_name)
        table.delete('document_id = "' + ref_doc_id + '"')

    def _init_reranker(self, **kwargs: Any) -> None:
        """Returns lancedb supported reranker object."""
        reranker = kwargs.get("reranker")
        if reranker:
            reranker_cls = reranker_mapper.get(reranker)
            mod = __import__("lancedb.rerankers", fromlist=["rerankers"])
            _reranker = getattr(mod, reranker_cls)

            if "reranker_kwargs" in kwargs:
                return _reranker(**kwargs["reranker_kwargs"])
            else:
                return _reranker()
        else:
            raise NotImplementedError(f"Reranker {reranker} not implemented.")

    def query(
        self,
        query: VectorStoreQuery,
        reranker: Optional[lancedb.rerankers.Reranker] = None,
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
            where = _to_lance_filter(query.filters, self._metadata_keys)
        else:
            where = kwargs.pop("where", None)

        query_type = kwargs.pop("query_type", "vector")

        if query_type == "vector":
            _query = query.query_embedding
        else:
            self._table.create_fts_index(self.text_key, replace=True)
            if query_type == "hybrid":
                _query = (query.query_embedding, query.query_str)
            else:
                _query = query.query_str

        lance_query = (
            self._table.search(
                query=_query,
                vector_column_name=self.vector_column_name,
                query_type=query_type,
            )
            .limit(query.similarity_top_k)
            .where(where)
            .nprobes(self.nprobes)
        )

        if "reranker" in kwargs:
            _reranker = self._init_reranker(**kwargs)
            lance_query.rerank(reranker=_reranker)

        if self.refine_factor is not None:
            lance_query.refine_factor(self.refine_factor)

        results = lance_query.to_pandas()

        if len(results) == 0:
            raise Warning("query results are empty..")

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
                if item.metadata:
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
