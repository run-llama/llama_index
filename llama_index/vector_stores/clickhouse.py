import logging

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional
from llama_index.bridge.pydantic import PrivateAttr
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import node_to_metadata_dict


IMPORT_ERROR_MSG = "`clickhouse_connect` package not found, please run `pip install clickhouse_connect`"
DEFAULT_DATABASE = "llama_index"
DEFAULT_VECTOR_STORE_TABLE = "vector_store"
DEFAULT_VECTOR_INDEX_PARAMS = {
    "distance_fun": "cosineDistance",
}
_logger = logging.getLogger(__name__)

@dataclasses
class Record:
    id: str
    hash: str
    type: int
    chunk_type: str
    text: str

    doc_id: Optional[str]
    doc_path: Optional[str]
    doc_version: Optional[str]
    doc_mark_deleted: bool

    abstract: str
    keyword: List[str]

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
        return Record(
            id=node.node_id,
            type=int(node.get_type()),
            hash=node.hash,
            ref_doc_id=node.ref_doc_id,
            embedding=node.get_embedding(),
            metadata=node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=True,
            ),
            excluded_embed_metadata_keys=[],
            excluded_llm_metadata_keys=[],
            relationships=node.dict().get("relationships", {}),
            text=node.get_content(metadata_mode=MetadataMode.NONE),
            start_char_idx=0,
            end_char_idx=0,
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

    embedding_dimension: int
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
            embedding_dimension=embedding_dimension,
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
              doc_id              FixedString(256),
              doc_path            String,
              doc_version         FixedString(64),
              doc_mark_deleted    Bool,
              abstract            String,
              keywords            Array(String),
              previous            FixedString(256),
              next                FixedString(256),
              parent              FixedString(256),
              metadata            JSON,
              embedding           Array(Float32),
              embedding_optional_1 Array(Float32),
              embedding_optional_2 Array(Float32),
              embedding_optional_3 Array(Float32),
              INDEX idx_doc_id doc_id TYPE set(0) GRANULARITY 1,
              INDEX idx_doc_version doc_version TYPE set(0) GRANULARITY 1,
              INDEX idx_doc_mark_deleted doc_mark_deleted TYPE set(0) GRANULARITY 1,
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
        self._client.command(
            f"""
            CREATE TABLE IF NOT EXISTS collection_information
            (
                collection_id       FixedString(36),
                name FixedString(256),
                description String,
                desc_embedding Array(Float32),
                desc_embedding_meta JSON,
                keywords Array(String),
                state FixedString(32),
                embedding_strategy JSON,
                metadata JSON,
                creation_date DateTime('Asia/Shanghai'),
                last_modified_date DateTime('Asia/Shanghai'),
                index_date DateTime('Asia/Shanghai'),
                INDEX idx_collection_id collection_id TYPE set(0) GRANULARITY 1,
                INDEX idx_state state TYPE set(0) GRANULARITY 1,
                INDEX idx_desc_embedding desc_embedding TYPE annoy('cosineDistance') GRANULARITY 8192
            )
            ENGINE = MergeTree()
            PRIMARY KEY (collection_id);"""
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
