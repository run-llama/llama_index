"""MyScale vector store.

An index that is built on top of an existing MyScale cluster.

"""
import json
import logging
from typing import Any, Dict, List, Optional, cast

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.indices.service_context import ServiceContext
from llama_index.readers.myscale import (
    MyScaleSettings,
    escape_str,
    format_list_to_string,
)
from llama_index.utils import iter_batch
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = logging.getLogger(__name__)


class MyScaleVectorStore(VectorStore):
    """MyScale Vector Store.

    In this vector store, embeddings and docs are stored within an existing
    MyScale cluster.

    During query time, the index uses MyScale to query for the top
    k most similar nodes.

    Args:
        myscale_client (httpclient): clickhouse-connect httpclient of
            an existing MyScale cluster.
        table (str, optional): The name of the MyScale table
            where data will be stored. Defaults to "llama_index".
        database (str, optional): The name of the MyScale database
            where data will be stored. Defaults to "default".
        index_type (str, optional): The type of the MyScale vector index.
            Defaults to "IVFFLAT".
        metric (str, optional): The metric type of the MyScale vector index.
            Defaults to "cosine".
        batch_size (int, optional): the size of documents to insert. Defaults to 32.
        index_params (dict, optional): The index parameters for MyScale.
            Defaults to None.
        search_params (dict, optional): The search parameters for a MyScale query.
            Defaults to None.
        service_context (ServiceContext, optional): Vector store service context.
            Defaults to None

    """

    stores_text: bool = True
    _index_existed: bool = False

    def __init__(
        self,
        myscale_client: Optional[Any] = None,
        table: str = "llama_index",
        database: str = "default",
        index_type: str = "IVFFLAT",
        metric: str = "cosine",
        batch_size: int = 32,
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = """
            `clickhouse_connect` package not found, 
            please run `pip install clickhouse-connect`
        """
        try:
            from clickhouse_connect.driver.httpclient import HttpClient  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        if myscale_client is None:
            raise ValueError("Missing MyScale client!")

        self._client = cast(HttpClient, myscale_client)
        self.config = MyScaleSettings(
            table=table,
            database=database,
            index_type=index_type,
            metric=metric,
            batch_size=batch_size,
            index_params=index_params,
            search_params=search_params,
            **kwargs,
        )

        # schema column name, type, and construct format method
        self.column_config: Dict = {
            "id": {"type": "String", "extract_func": lambda x: x.id},
            "doc_id": {"type": "String", "extract_func": lambda x: x.ref_doc_id},
            "text": {
                "type": "String",
                "extract_func": lambda x: escape_str(x.node.text or ""),
            },
            "vector": {
                "type": "Array(Float32)",
                "extract_func": lambda x: format_list_to_string(x.embedding),
            },
            "node_info": {
                "type": "JSON",
                "extract_func": lambda x: json.dumps(x.node.node_info),
            },
            "extra_info": {
                "type": "JSON",
                "extract_func": lambda x: json.dumps(x.node.extra_info),
            },
        }

        if service_context is not None:
            service_context = cast(ServiceContext, service_context)
            dimension = len(
                service_context.embed_model.get_query_embedding("try this out")
            )
            self._create_index(dimension)

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def _create_index(self, dimension: int) -> None:
        index_params = (
            ", " + ",".join([f"'{k}={v}'" for k, v in self.config.index_params.items()])
            if self.config.index_params
            else ""
        )
        schema_ = f"""
            CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(
                {",".join([f'{k} {v["type"]}' for k, v in self.column_config.items()])},
                CONSTRAINT vector_length CHECK length(vector) = {dimension},
                VECTOR INDEX {self.config.table}_index vector TYPE 
                {self.config.index_type}('metric_type={self.config.metric}'{index_params})
            ) ENGINE = MergeTree ORDER BY id
            """
        self.dim = dimension
        self._client.command("SET allow_experimental_object_type=1")
        self._client.command(schema_)
        self._index_existed = True

    def _build_insert_statement(
        self,
        values: List[NodeWithEmbedding],
    ) -> str:
        _data = []
        for item in values:
            item_value_str = ",".join(
                [
                    f"'{column['extract_func'](item)}'"
                    for column in self.column_config.values()
                ]
            )
            _data.append(f"({item_value_str})")

        insert_statement = f"""
                INSERT INTO TABLE 
                    {self.config.database}.{self.config.table}({",".join(self.column_config.keys())})
                VALUES
                    {','.join(_data)}
                """
        return insert_statement

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """

        if not embedding_results:
            return []

        if not self._index_existed:
            self._create_index(len(embedding_results[0].embedding))

        for result_batch in iter_batch(embedding_results, self.config.batch_size):
            insert_statement = self._build_insert_statement(values=result_batch)
            self._client.command(insert_statement)

        return [result.id for result in embedding_results]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        raise NotImplementedError("Delete not yet implemented for MyScale index.")

    def drop(self) -> None:
        """Drop MyScale Index and table"""
        self._client.command(
            f"DROP TABLE IF EXISTS {self.config.database}.{self.config.table}"
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query

        """
        if query.filters is not None:
            raise ValueError(
                "Metadata filters not implemented for SimpleVectorStore yet."
            )

        query_embedding = cast(List[float], query.query_embedding)
        where_str = (
            f"doc_id in {format_list_to_string(query.doc_ids)}"
            if query.doc_ids
            else None
        )
        query_statement = self.config.build_query_statement(
            query_embed=query_embedding,
            where_str=where_str,
            limit=query.similarity_top_k,
        )

        nodes = []
        ids = []
        similarities = []
        for r in self._client.query(query_statement).named_results():
            node = Node(
                doc_id=r["doc_id"],
                text=r["text"],
                extra_info=r["extra_info"],
                node_info=r["node_info"],
                relationships={DocumentRelationship.SOURCE: r["doc_id"]},
            )

            nodes.append(node)
            similarities.append(r["dist"])
            ids.append(r["id"])
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
