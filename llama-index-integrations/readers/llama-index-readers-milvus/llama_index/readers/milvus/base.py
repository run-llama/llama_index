"""Milvus reader."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MilvusReader(BaseReader):
    """Milvus reader."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        use_secure: bool = False,
    ):
        """Initialize with parameters."""
        import_err_msg = (
            "`pymilvus` package not found, please run `pip install pymilvus`"
        )
        try:
            import pymilvus  # noqa
        except ImportError:
            raise ImportError(import_err_msg)

        from pymilvus import MilvusException

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.use_secure = use_secure
        self.collection = None

        self.default_search_params = {
            "IVF_FLAT": {"metric_type": "IP", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "IP", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "IP", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "IP", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "IP", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "IP", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "IP", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "IP", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "IP", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "IP", "params": {}},
        }
        try:
            self._create_connection_alias()
        except MilvusException:
            raise

    def load_data(
        self,
        query_vector: List[float],
        collection_name: str,
        expr: Any = None,
        search_params: Optional[dict] = None,
        limit: int = 10,
    ) -> List[Document]:
        """
        Load data from Milvus.

        Args:
            collection_name (str): Name of the Milvus collection.
            query_vector (List[float]): Query vector.
            limit (int): Number of results to return.

        Returns:
            List[Document]: A list of documents.

        """
        from pymilvus import Collection, MilvusException

        try:
            self.collection = Collection(collection_name, using=self.alias)
        except MilvusException:
            raise

        assert self.collection is not None
        try:
            self.collection.load()
        except MilvusException:
            raise
        if search_params is None:
            search_params = self._create_search_params()

        res = self.collection.search(
            [query_vector],
            "embedding",
            param=search_params,
            expr=expr,
            output_fields=["doc_id", "text"],
            limit=limit,
        )

        documents = []
        # TODO: In future append embedding when more efficient
        for hit in res[0]:
            document = Document(
                id_=hit.entity.get("doc_id"),
                text=hit.entity.get("text"),
            )

            documents.append(document)

        return documents

    def _create_connection_alias(self) -> None:
        from pymilvus import connections

        self.alias = None
        # Attempt to reuse an open connection
        for x in connections.list_connections():
            addr = connections.get_connection_addr(x[0])
            if (
                x[1]
                and ("address" in addr)
                and (addr["address"] == f"{self.host}:{self.port}")
            ):
                self.alias = x[0]
                break

        # Connect to the Milvus instance using the passed in Environment variables
        if self.alias is None:
            self.alias = uuid4().hex
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
                user=self.user,  # type: ignore
                password=self.password,  # type: ignore
                secure=self.use_secure,
            )

    def _create_search_params(self) -> Dict[str, Any]:
        assert self.collection is not None
        index = self.collection.indexes[0]._index_params
        search_params = self.default_search_params[index["index_type"]]
        search_params["metric_type"] = index["metric_type"]
        return search_params
