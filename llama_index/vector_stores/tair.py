"""Tair Vector store index.

An index that is built on top of Alibaba Cloud's Tair database.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llama_index.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import node_to_metadata_dict

_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from tair import Tair


def _to_filter_expr(filters: MetadataFilters) -> str:
    conditions = []
    for f in filters.legacy_filters():
        value = str(f.value)
        if isinstance(f.value, str):
            value = '"' + value + '"'
        conditions.append(f"{f.key}=={value}")
    return "&&".join(conditions)


class TairVectorStore(VectorStore):
    stores_text = True
    stores_node = True
    flat_metadata = False

    def __init__(
        self,
        tair_url: str,
        index_name: str,
        index_type: str = "HNSW",
        index_args: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize TairVectorStore.

        Two index types are available: FLAT & HNSW.

        index args for HNSW:
            - ef_construct
            - M
            - ef_search

        Detailed info for these arguments can be found here:
        https://www.alibabacloud.com/help/en/tair/latest/tairvector#section-c76-ull-5mk

        Args:
            index_name (str): Name of the index.
            index_type (str): Type of the index. Defaults to 'HNSW'.
            index_args (Dict[str, Any]): Arguments for the index. Defaults to None.
            tair_url (str): URL for the Tair instance.
            overwrite (bool): Whether to overwrite the index if it already exists.
                Defaults to False.
            kwargs (Any): Additional arguments to pass to the Tair client.

        Raises:
            ValueError: If tair-py is not installed
            ValueError: If failed to connect to Tair instance

        Examples:
            >>> from llama_index.vector_stores.tair import TairVectorStore
            >>> # Create a TairVectorStore
            >>> vector_store = TairVectorStore(
            >>>     tair_url="redis://{username}:{password}@r-bp****************.\
                redis.rds.aliyuncs.com:{port}",
            >>>     index_name="my_index",
            >>>     index_type="HNSW",
            >>>     index_args={"M": 16, "ef_construct": 200},
            >>>     overwrite=True)

        """
        try:
            from tair import Tair, tairvector  # noqa
        except ImportError:
            raise ValueError(
                "Could not import tair-py python package. "
                "Please install it with `pip install tair`."
            )
        try:
            self._tair_client = Tair.from_url(tair_url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Tair failed to connect: {e}")

        # index identifiers
        self._index_name = index_name
        self._index_type = index_type
        self._metric_type = "L2"
        self._overwrite = overwrite
        self._index_args = {}
        self._query_args = {}
        if index_type == "HNSW":
            if index_args is not None:
                ef_construct = index_args.get("ef_construct", 500)
                M = index_args.get("M", 24)
                ef_search = index_args.get("ef_search", 400)
            else:
                ef_construct = 500
                M = 24
                ef_search = 400

            self._index_args = {"ef_construct": ef_construct, "M": M}
            self._query_args = {"ef_search": ef_search}

    @property
    def client(self) -> "Tair":
        """Return the Tair client instance."""
        return self._tair_client

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to the index.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings

        Returns:
            List[str]: List of ids of the documents added to the index.
        """
        # check to see if empty document list was passed
        if len(nodes) == 0:
            return []

        # set vector dim for creation if index doesn't exist
        self.dim = len(nodes[0].get_embedding())

        if self._index_exists():
            if self._overwrite:
                self.delete_index()
                self._create_index()
            else:
                logging.info(f"Adding document to existing index {self._index_name}")
        else:
            self._create_index()

        ids = []
        for node in nodes:
            attributes = {
                "id": node.node_id,
                "doc_id": node.ref_doc_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
            }
            metadata_dict = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            attributes.update(metadata_dict)

            ids.append(node.node_id)
            self._tair_client.tvs_hset(
                self._index_name,
                f"{node.ref_doc_id}#{node.node_id}",
                vector=node.get_embedding(),
                is_binary=False,
                **attributes,
            )

        _logger.info(f"Added {len(ids)} documents to index {self._index_name}")
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id

        """
        iter = self._tair_client.tvs_scan(self._index_name, "%s#*" % ref_doc_id)
        for k in iter:
            self._tair_client.tvs_del(self._index_name, k)

    def delete_index(self) -> None:
        """Delete the index and all documents."""
        _logger.info(f"Deleting index {self._index_name}")
        self._tair_client.tvs_del_index(self._index_name)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the index.

        Args:
            query (VectorStoreQuery): query object

        Returns:
            VectorStoreQueryResult: query result

        Raises:
            ValueError: If query.query_embedding is None.
        """
        filter_expr = None
        if query.filters is not None:
            filter_expr = _to_filter_expr(query.filters)

        if not query.query_embedding:
            raise ValueError("Query embedding is required for querying.")

        _logger.info(f"Querying index {self._index_name}")

        query_args = self._query_args
        if self._index_type == "HNSW" and "ef_search" in kwargs:
            query_args["ef_search"] = kwargs["ef_search"]

        results = self._tair_client.tvs_knnsearch(
            self._index_name,
            query.similarity_top_k,
            query.query_embedding,
            False,
            filter_str=filter_expr,
            **query_args,
        )
        results = [(k.decode(), float(s)) for k, s in results]

        ids = []
        nodes = []
        scores = []
        pipe = self._tair_client.pipeline(transaction=False)
        for key, score in results:
            scores.append(score)
            pipe.tvs_hmget(self._index_name, key, "id", "doc_id", "text")
        metadatas = pipe.execute()
        for i, m in enumerate(metadatas):
            # TODO: properly get the _node_conent
            doc_id = m[0].decode()
            node = TextNode(
                text=m[2].decode(),
                id_=doc_id,
                embedding=None,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=m[1].decode())
                },
            )
            ids.append(doc_id)
            nodes.append(node)
        _logger.info(f"Found {len(nodes)} results for query with id {ids}")

        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)

    def _create_index(self) -> None:
        try:
            from tair import tairvector
        except ImportError:
            raise ValueError(
                "Could not import tair-py python package. "
                "Please install it with `pip install tair`."
            )
        _logger.info(f"Creating index {self._index_name}")
        self._tair_client.tvs_create_index(
            self._index_name,
            self.dim,
            distance_type=self._metric_type,
            index_type=self._index_type,
            data_type=tairvector.DataType.Float32,
            **self._index_args,
        )

    def _index_exists(self) -> bool:
        index = self._tair_client.tvs_get_index(self._index_name)
        return index is not None
