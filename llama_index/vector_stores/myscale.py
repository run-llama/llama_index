"""MyScale vector store.

An index that is built on top of an existing MyScale cluster.

"""
import json
import logging
from typing import Any, Dict, List, Optional, cast

from llama_index.readers.myscale import (
    MyScaleSettings,
    escape_str,
    format_list_to_string,
)
from llama_index.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.service_context import ServiceContext
from llama_index.utils import iter_batch
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
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
    metadata_column: str = "metadata"
    AMPLIFY_RATIO_LE5 = 100
    AMPLIFY_RATIO_GT5 = 20
    AMPLIFY_RATIO_GT50 = 10

    def __init__(
        self,
        myscale_client: Optional[Any] = None,
        table: str = "llama_index",
        database: str = "default",
        index_type: str = "MSTG",
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
            from clickhouse_connect.driver.httpclient import HttpClient
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
            "id": {"type": "String", "extract_func": lambda x: x.node_id},
            "doc_id": {"type": "String", "extract_func": lambda x: x.ref_doc_id},
            "text": {
                "type": "String",
                "extract_func": lambda x: escape_str(
                    x.get_content(metadata_mode=MetadataMode.NONE) or ""
                ),
            },
            "vector": {
                "type": "Array(Float32)",
                "extract_func": lambda x: format_list_to_string(x.get_embedding()),
            },
            "node_info": {
                "type": "JSON",
                "extract_func": lambda x: json.dumps(x.node_info),
            },
            "metadata": {
                "type": "JSON",
                "extract_func": lambda x: json.dumps(x.metadata),
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
        values: List[BaseNode],
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

        return f"""
                INSERT INTO TABLE
                    {self.config.database}.{self.config.table}({",".join(self.column_config.keys())})
                VALUES
                    {','.join(_data)}
                """

    def _build_hybrid_search_statement(
        self, stage_one_sql: str, query_str: str, similarity_top_k: int
    ) -> str:
        terms_pattern = [f"(?i){x}" for x in query_str.split(" ")]
        column_keys = self.column_config.keys()
        return (
            f"SELECT {','.join(filter(lambda k: k != 'vector', column_keys))}, "
            f"dist FROM ({stage_one_sql}) tempt "
            f"ORDER BY length(multiMatchAllIndices(text, {terms_pattern})) "
            f"AS distance1 DESC, "
            f"log(1 + countMatches(text, '(?i)({query_str.replace(' ', '|')})')) "
            f"AS distance2 DESC limit {similarity_top_k}"
        )

    def _append_meta_filter_condition(
        self, where_str: Optional[str], exact_match_filter: list
    ) -> str:
        filter_str = " AND ".join(
            f"JSONExtractString(toJSONString("
            f"{self.metadata_column}), '{filter_item.key}') "
            f"= '{filter_item.value}'"
            for filter_item in exact_match_filter
        )
        if where_str is None:
            where_str = filter_str
        else:
            where_str = " AND " + filter_str
        return where_str

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        if not nodes:
            return []

        if not self._index_existed:
            self._create_index(len(nodes[0].get_embedding()))

        for result_batch in iter_batch(nodes, self.config.batch_size):
            insert_statement = self._build_insert_statement(values=result_batch)
            self._client.command(insert_statement)

        return [result.node_id for result in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        self._client.command(
            f"DELETE FROM {self.config.database}.{self.config.table} "
            f"where doc_id='{ref_doc_id}'"
        )

    def drop(self) -> None:
        """Drop MyScale Index and table."""
        self._client.command(
            f"DROP TABLE IF EXISTS {self.config.database}.{self.config.table}"
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query

        """
        query_embedding = cast(List[float], query.query_embedding)
        where_str = (
            f"doc_id in {format_list_to_string(query.doc_ids)}"
            if query.doc_ids
            else None
        )
        if query.filters is not None:
            where_str = self._append_meta_filter_condition(
                where_str, query.filters.filters
            )

        # build query sql
        query_statement = self.config.build_query_statement(
            query_embed=query_embedding,
            where_str=where_str,
            limit=query.similarity_top_k,
        )
        if query.mode == VectorStoreQueryMode.HYBRID and query.query_str is not None:
            amplify_ratio = self.AMPLIFY_RATIO_LE5
            if 5 < query.similarity_top_k < 50:
                amplify_ratio = self.AMPLIFY_RATIO_GT5
            if query.similarity_top_k > 50:
                amplify_ratio = self.AMPLIFY_RATIO_GT50
            query_statement = self._build_hybrid_search_statement(
                self.config.build_query_statement(
                    query_embed=query_embedding,
                    where_str=where_str,
                    limit=query.similarity_top_k * amplify_ratio,
                ),
                query.query_str,
                query.similarity_top_k,
            )
            logger.debug(f"hybrid query_statement={query_statement}")
        nodes = []
        ids = []
        similarities = []
        for r in self._client.query(query_statement).named_results():
            start_char_idx = None
            end_char_idx = None

            if isinstance(r["node_info"], dict):
                start_char_idx = r["node_info"].get("start", None)
                end_char_idx = r["node_info"].get("end", None)
            node = TextNode(
                id_=r["doc_id"],
                text=r["text"],
                metadata=r["metadata"],
                start_char_idx=start_char_idx,
                end_char_idx=end_char_idx,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=r["doc_id"])
                },
            )

            nodes.append(node)
            similarities.append(r["dist"])
            ids.append(r["id"])
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
