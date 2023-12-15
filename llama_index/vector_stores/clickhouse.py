import logging

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional
from llama_index.bridge.pydantic import PrivateAttr
from llama_index.schema import BaseNode, MetadataMode, TextNode, NodeRelationship
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import node_to_metadata_dict


IMPORT_ERROR_MSG = "`clickhouse-connect` package not found, please run `pip install clickhouse-connect`"
DEFAULT_DATABASE = "llama_index"
DEFAULT_VECTOR_STORE_TABLE = "vector_store"
DEFAULT_VECTOR_INDEX_PARAMS = {
    "distance_fun": "cosineDistance",
}
_logger = logging.getLogger(__name__)

@dataclass
class Record:
    id: str
    hash: str
    type: str
    chunk_type: str
    text: str

    doc_id: Optional[str]
    doc_path: Optional[str]
    doc_version: Optional[str]
    doc_author: Optional[str]
    doc_category: Optional[str]
    doc_owner: Optional[str]

    deleted: bool

    abstract: str
    keywords: List[str]

    previous: Optional[str]
    next: Optional[str]
    parent: Optional[str]

    metadata: Dict[str, Any]
    embedding: List[float]
    embedding_optional_1: List[float]
    embedding_optional_2: List[float]
    embedding_optional_3: List[float]

    @staticmethod
    def columns() -> List[str]:
        return [field.name for field in fields(Record)]

    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(Record)}

    def values(self) -> List[Any]:
        record_dict = self.to_dict()
        return list(record_dict.values())

    @staticmethod
    def from_node(node: BaseNode) -> "Record":
        _previous = ""
        _next = ""
        _parent = ""
        if node.relationships.get(NodeRelationship.PREVIOUS) is not None:
            _previous = (node.relationships.get(NodeRelationship.PREVIOUS).node_id)
        if node.relationships.get(NodeRelationship.NEXT) is not None:
            _next = (node.relationships.get(NodeRelationship.NEXT).node_id)
        if node.relationships.get(NodeRelationship.PARENT) is not None:
            _parent = (node.relationships.get(NodeRelationship.PARENT).node_id)

        return Record(
            id=node.node_id,
            hash=node.hash,
            type=node.get_type(),
            chunk_type=node.metadata.get("chunk_type"),
            text=node.get_content(metadata_mode=MetadataMode.NONE),

            doc_id=node.metadata.get("doc_id"),
            doc_path=node.metadata.get("doc_path"),
            doc_version=node.metadata.get("doc_version"),
            doc_author=node.metadata.get("doc_author"),
            doc_category=node.metadata.get("doc_category"),
            doc_owner=node.metadata.get("doc_owner"),

            deleted=node.metadata.get("deleted"),

            abstract=node.metadata.get("abstract"),
            keywords=node.metadata.get("keywords"),

            previous=_previous,
            next=_next,
            parent=_parent,

            metadata=node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=True,
            ),
            embedding=node.get_embedding(),
            embedding_optional_1=[0],
            embedding_optional_2=[0],
            embedding_optional_3=[0],
        )
class IndexOptions:
    def __init__(self, index_type: str, index_params: dict[str, Any]):
        self.index_type = index_type
        self.index_params = index_params if index_params is not None else {}

    @property
    def distance_fun(self) -> str:
        return self.index_params.get("distance_fun", "cosineDistance")


class ClickhouseVectorStore(BasePydanticVectorStore):
    stores_text = True
    host: str
    port: int
    username: str
    password: str
    database: str
    table_name: str

    ##embedding_dimension: int
    index_type: str
    index_params: dict[str, Any]
    perform_setup: bool
    debug: bool
    _client: Any = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
            self,
            host: str,
            port: int,
            database: str,
            username: str = "",
            password: str = "",
            table_name: str = DEFAULT_VECTOR_STORE_TABLE,
            embedding_dimension: int = 786,
            index_type: str = "annoy",
            index_params: dict[str, Any] = DEFAULT_VECTOR_INDEX_PARAMS,
            perform_setup: bool = True,
            debug: bool = False,
    ) -> None:
        table_name = table_name.lower()

        super().__init__(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            table_name=table_name,
            ##embedding_dimension=embedding_dimension,  embedding槽位扩宽，不再固定
            index_type=index_type,
            index_params=index_params,
            perform_setup=perform_setup,
            debug=debug,
        )
    @property
    def index_options(self) -> IndexOptions:
        return IndexOptions(self.index_type, self.index_params)

    @classmethod
    def class_name(cls) -> str:
        return "ClickhouseVectorStore"

    @property
    def client(self) -> Any:
        if not self._is_initialized:
            return None
        return self._client

    def _connect(self) -> Any:
        try:
            import clickhouse_connect
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            database=self.database,
            username=self.username,
            password=self.password,
        )
        self._client = client

    def _create_database_if_not_exists(self) -> None:
        self._client.command(f"CREATE DATABASE IF NOT EXISTS {self.database}")

    def _create_tables_if_not_exists(self) -> None:
        self._client.command("SET allow_experimental_object_type = 1;")
        self._client.command("SET allow_experimental_annoy_index = 1;")
        self._client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.table_name}
            (
                  id                  FixedString(256),
                  hash                FixedString(256),
                  type                FixedString(32),
                  chunk_type          FixedString(32),
                  text                String,
                  doc_id              Nullable(FixedString(256)),
                  doc_path            Nullable(String),
                  doc_version         Nullable(FixedString(64)),
                  doc_author          Nullable(FixedString(64)),
                  doc_category        Nullable(FixedString(64)),
                  doc_owner           Nullable(FixedString(64)),
                  deleted             Bool,
                  abstract            Nullable(String),
                  keywords            Nullable(String),
                  previous            Nullable(FixedString(256)),
                  next                Nullable(FixedString(256)),
                  parent              Nullable(FixedString(256)),
                  metadata            JSON,
                  embedding           Array(Float32),
                  embedding_optional_1 Array(Float32),
                  embedding_optional_2 Array(Float32),
                  embedding_optional_3 Array(Float32),
                  INDEX idx_doc_id doc_id TYPE set(0) GRANULARITY 1,
                  INDEX idx_doc_version doc_version TYPE set(0) GRANULARITY 1,
                  INDEX deleted deleted TYPE set(0) GRANULARITY 1,
                  INDEX idx_previous previous TYPE set(0) GRANULARITY 1,
                  INDEX idx_next next TYPE set(0) GRANULARITY 1,
                  INDEX idx_parent parent TYPE set(0) GRANULARITY 1,
                  INDEX idx_embedding embedding TYPE annoy('cosineDistance') GRANULARITY 8192,
                  INDEX idx_embedding_optional_1 embedding_optional_1 TYPE annoy('cosineDistance') GRANULARITY 8192,
                  INDEX idx_embedding_optional_2 embedding_optional_2 TYPE annoy('cosineDistance') GRANULARITY 8192,
                  INDEX idx_embedding_optional_3 embedding_optional_3 TYPE annoy('cosineDistance') GRANULARITY 8192
            )
            ENGINE = MergeTree()
            PRIMARY KEY (id);"""
        )

    def _initialize(self) -> None:
        if not self._is_initialized:
            self._connect()
            if self.perform_setup:
                self._create_database_if_not_exists()
                self._create_tables_if_not_exists()
            self._is_initialized = True

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        self._initialize()
        node_ids = []
        node_data = []
        for node in nodes:
            node_ids.append(node.node_id)
            node_data.append(Record.from_node(node).values())
        self._client.insert(self.table_name, node_data, Record.columns())

        return node_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._initialize()
        self._client.command(
            f"DELETE FROM {self.database}.{self.table_name} WHERE ref_doc_id = '{ref_doc_id}'"
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
        """
        self._initialize()

        parameters = {'query_embedding': query.query_embedding,
                      'similarity_top_k': query.similarity_top_k,
                      'table': self.table_name}
        query_sql = """
            SELECT id, text, L2Distance(embedding, {query_embedding:Array(Float32)}) as distance
            FROM {table:Identifier}
            order by L2Distance(embedding, {query_embedding:Array(Float32)}) LIMIT {similarity_top_k:UInt8}
            """
        res = self._client.query(query_sql, parameters=parameters)

        _logger.debug(
            f"Successfully searched embedding in collection: {self.table_name}"
            f" Num Results: {len(res.result_rows)}"
        )

        nodes = []
        similarities = []
        ids = []

        for row in res.result_rows:
            ids.append(row[0])
            node = TextNode(text=row[1])
            nodes.append(node)
            similarities.append(row[2])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
