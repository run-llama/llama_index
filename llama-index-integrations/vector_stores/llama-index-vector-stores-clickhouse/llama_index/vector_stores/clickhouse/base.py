"""
ClickHouse vector store.

An index that is built on top of an existing ClickHouse cluster.
For current documentation, refer to : https://clickhouse.com/docs/engines/table-engines/mergetree-family/annindexes
"""

import importlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, cast

from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

logger = logging.getLogger(__name__)


def _default_tokenizer(text: str) -> List[str]:
    """Default tokenizer."""
    tokens = re.split(r"[ \n]", text)  # split by space or newline
    result = []
    for token in tokens:
        if token.strip() == "":
            continue
        result.append(token.strip())
    return result


def escape_str(value: str) -> str:
    BS = "\\"
    must_escape = (BS, "'")
    return (
        "".join(f"{BS}{c}" if c in must_escape else c for c in value) if value else ""
    )


def format_list_to_string(lst: List) -> str:
    return "[" + ",".join(escape_str(str(item)) for item in lst) + "]"


DISTANCE_MAPPING = {
    "l2": "L2Distance",
    "cosine": "cosineDistance",
}


class ClickHouseSettings:
    """
    ClickHouse Client Configuration.

    Args:
        table (str): Table name to operate on.
        database (str): Database name to find the table.
        engine (str): Engine. Options are "MergeTree" and "Memory". Default is "MergeTree".
        index_type (str): Index type string.
        metric (str): Metric type to compute distance e.g., cosine or l2
        batch_size (int): The size of documents to insert.
        index_params (dict, optional): Index build parameter.
        search_params (dict, optional): Index search parameters for ClickHouse query.

    """

    def __init__(
        self,
        table: str,
        database: str,
        engine: str,
        index_type: str,
        metric: str,
        batch_size: int,
        dimension: Optional[int] = None,
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        self.table = table
        self.database = database
        self.engine = engine
        self.index_type = index_type
        self.metric = metric
        self.batch_size = batch_size
        self.dimension = dimension
        self.index_params = index_params
        self.search_params = search_params

    def build_query_statement(
        self,
        query_embed: List[float],
        where_str: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        query_embed_str = format_list_to_string(query_embed)
        where_str = f"WHERE {where_str}" if where_str else ""
        distance = DISTANCE_MAPPING[self.metric]
        return f"""
            SELECT id, doc_id, text, node_info, metadata,
            {distance}(vector, {query_embed_str}) AS score
            FROM {self.database}.{self.table} {where_str}
            ORDER BY score ASC
            LIMIT {limit}
            """


class ClickHouseVectorStore(BasePydanticVectorStore):
    """
    ClickHouse Vector Store.
    In this vector store, embeddings and docs are stored within an existing
    ClickHouse cluster.
    During query time, the index uses ClickHouse to query for the top
    k most similar nodes.

    Args:
        clickhouse_client (httpclient): clickhouse-connect httpclient of
            an existing ClickHouse cluster.
        table (str, optional): The name of the ClickHouse table
            where data will be stored. Defaults to "llama_index".
        database (str, optional): The name of the ClickHouse database
            where data will be stored. Defaults to "default".
        index_type (str, optional): The type of the ClickHouse vector index.
            Defaults to "HNSW", supported are ("NONE", "HNSW"). Use NONE for brute-force KNN search.
        metric (str, optional): The metric type of the ClickHouse vector index.
            Defaults to "cosine". Alternate metric type is "l2".
        batch_size (int, optional): the size of documents to insert. Defaults to 1000.
        index_params (dict, optional): The index parameters for ClickHouse.
            Defaults to None. For HNSW, following parameters are supported :-
            "quantization" : One of 'f64','f32', 'f16', 'bf16', 'i8', 'b1'. Default is 'bf16'.
            "hnsw_max_connections_per_layer": Default is 32.
            "hnsw_candidate_list_size_for_construction" : Default is 128.
        search_params (dict, optional): The search parameters for a ClickHouse query.
            Defaults to None.

    Examples:
        `pip install llama-index-vector-stores-clickhouse`

        ```python
        from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
        import clickhouse_connect

        # initialize client
        client = clickhouse_connect.get_client(
            host="localhost",
            port=8123,
            username="default",
            password="",
        )

        vector_store = ClickHouseVectorStore(clickhouse_client=client)
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = False
    _table_existed: bool = PrivateAttr(default=False)
    _client: Any = PrivateAttr()
    _config: Any = PrivateAttr()
    _dim: Any = PrivateAttr()
    _column_config: Any = PrivateAttr()
    _column_names: List[str] = PrivateAttr()
    _column_type_names: List[str] = PrivateAttr()
    metadata_column: str = "metadata"
    AMPLIFY_RATIO_LE5: int = 100
    AMPLIFY_RATIO_GT5: int = 20
    AMPLIFY_RATIO_GT50: int = 10

    def __init__(
        self,
        clickhouse_client: Optional[Any] = None,
        table: str = "llama_index",
        database: str = "default",
        engine: str = "MergeTree",
        index_type: str = "HNSW",
        metric: str = "cosine",
        batch_size: int = 1000,
        dimension: Optional[int] = None,
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = """
            `clickhouse_connect` package not found,
            please run `pip install clickhouse-connect`
        """
        clickhouse_connect_spec = importlib.util.find_spec(
            "clickhouse_connect.driver.httpclient"
        )
        if clickhouse_connect_spec is None:
            raise ImportError(import_err_msg)

        if clickhouse_client is None:
            raise ValueError("Missing ClickHouse client!")
        client = clickhouse_client
        config = ClickHouseSettings(
            table=table,
            database=database,
            engine=engine,
            index_type=index_type,
            metric=metric,
            batch_size=batch_size,
            dimension=dimension,
            index_params=index_params,
            search_params=search_params,
            **kwargs,
        )

        # schema column name, type, and construct format method
        column_config: Dict = {
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
                "extract_func": lambda x: x.get_embedding(),
            },
            "node_info": {
                "type": "Tuple(start Nullable(UInt64), end Nullable(UInt64))",
                "extract_func": lambda x: x.get_node_info(),
            },
            "metadata": {
                "type": "String",
                "extract_func": lambda x: json.dumps(x.metadata),
            },
        }
        column_names = list(column_config.keys())
        column_type_names = [
            column_config[column_name]["type"] for column_name in column_names
        ]

        super().__init__(
            clickhouse_client=clickhouse_client,
            table=table,
            database=database,
            engine=engine,
            index_type=index_type,
            metric=metric,
            batch_size=batch_size,
            dimension=dimension,
            index_params=index_params,
            search_params=search_params,
        )
        self._client = client
        self._config = config
        self._column_config = column_config
        self._column_names = column_names
        self._column_type_names = column_type_names
        if dimension is None:
            dimension = len(Settings.embed_model.get_query_embedding("try this out"))
        self.create_table(dimension)

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def create_table(self, dimension: int) -> None:
        index = ""
        settings = {"allow_experimental_vector_similarity_index": "1"}
        quantization = "bf16"
        M = 32
        ef_c = 128

        if self._config.index_type.lower() == "hnsw":
            if (
                self._config.index_params
                and "quantization" in self._config.index_params
            ):
                quantization = self._config.index_params["quantization"]
            if (
                self._config.index_params
                and "hnsw_max_connections_per_layer" in self._config.index_params
            ):
                M = self._config.index_params["hnsw_max_connections_per_layer"]
            if (
                self._config.index_params
                and "hnsw_candidate_list_size_for_construction"
                in self._config.index_params
            ):
                ef_c = self._config.index_params[
                    "hnsw_candidate_list_size_for_construction"
                ]
            index = f"INDEX vector_index vector TYPE vector_similarity('hnsw', '{DISTANCE_MAPPING[self._config.metric]}', {dimension}, '{quantization}', {M}, {ef_c})"
        schema_ = f"""
            CREATE TABLE IF NOT EXISTS {self._config.database}.{self._config.table}(
                {",".join([f"{k} {v['type']}" for k, v in self._column_config.items()])},
                CONSTRAINT vector_length CHECK length(vector) = {dimension},
                {index}
            ) ENGINE = MergeTree ORDER BY id
            """
        self._dim = dimension
        self.drop()
        self._client.command(schema_, settings=settings)
        self._table_existed = True

    def _upload_batch(
        self,
        batch: List[BaseNode],
    ) -> None:
        _data = []
        # we assume all rows have all columns
        for idx, item in enumerate(batch):
            _row = []
            for column_name in self._column_names:
                _row.append(self._column_config[column_name]["extract_func"](item))
            _data.append(_row)

        self._client.insert(
            f"{self._config.database}.{self._config.table}",
            data=_data,
            column_names=self._column_names,
            column_type_names=self._column_type_names,
        )

    def _build_text_search_statement(
        self, query_str: str, similarity_top_k: int
    ) -> str:
        safe_tokens = []
        for token in _default_tokenizer(query_str):
            # First escape regex special characters
            regex_escaped = re.escape(token)
            # Then escape for SQL string
            sql_escaped = escape_str(regex_escaped)
            safe_tokens.append(sql_escaped)

        terms_pattern = [f"\\b(?i){token}\\b" for token in safe_tokens]
        joined_tokens_pattern = escape_str("|".join(safe_tokens))
        column_keys = [k for k in self._column_config if k != "vector"]
        column_list = ",".join(column_keys)
        return (
            f"SELECT {column_list}, score "
            f"FROM {self._config.database}.{self._config.table} WHERE score > 0 "
            f"ORDER BY length(multiMatchAllIndices(text, {terms_pattern})) "
            f"AS score DESC, "
            f"log(1 + countMatches(text, '\\b(?i)({joined_tokens_pattern})\\b')) "
            f"AS d2 DESC limit {similarity_top_k}"
        )

    def _build_hybrid_search_statement(
        self, stage_one_sql: str, query_str: str, similarity_top_k: int
    ) -> str:
        safe_tokens = []
        for token in _default_tokenizer(query_str):
            # First escape regex special characters
            regex_escaped = re.escape(token)
            # Then escape for SQL string
            sql_escaped = escape_str(regex_escaped)
            safe_tokens.append(sql_escaped)

        terms_pattern = [f"\\b(?i){token}\\b" for token in safe_tokens]
        joined_tokens_pattern = escape_str("|".join(safe_tokens))
        column_keys = [k for k in self._column_config if k != "vector"]
        column_list = ",".join(column_keys)
        return (
            f"SELECT {column_list}, score "
            f"FROM ({stage_one_sql}) tempt "
            f"ORDER BY length(multiMatchAllIndices(text, {terms_pattern})) "
            f"AS d1 DESC, "
            f"log(1 + countMatches(text, '\\b(?i)({joined_tokens_pattern})\\b')) "
            f"AS d2 DESC limit {similarity_top_k}"
        )

    def _append_meta_filter_condition(
        self, where_str: Optional[str], exact_match_filter: list
    ) -> str:
        if not exact_match_filter:
            return where_str or ""

        filter_conditions = []
        for filter_item in exact_match_filter:
            # Use JSONExtractString function with properly escaped keys and values
            key = escape_str(filter_item.key)
            value = escape_str(filter_item.value)
            filter_conditions.append(
                f"JSONExtractString({self.metadata_column}, '{key}') = '{value}'"
            )

        filter_str = " AND ".join(filter_conditions)

        if not where_str:
            return filter_str
        return f"{where_str} AND {filter_str}"

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        if not nodes:
            return []

        if not self._table_existed:
            self.create_table(len(nodes[0].get_embedding()))

        for batch in iter_batch(nodes, self._config.batch_size):
            self._upload_batch(batch=batch)

        return [result.node_id for result in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        query = f"DELETE FROM {self._config.database}.{self._config.table} WHERE doc_id = %(ref_doc_id)s"
        self._client.command(query, parameters={"ref_doc_id": ref_doc_id})

    def drop(self) -> None:
        """Drop ClickHouse table."""
        self._client.command(
            f"DROP TABLE IF EXISTS {self._config.database}.{self._config.table}"
        )

    def query(
        self, query: VectorStoreQuery, where: Optional[str] = None, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query
            where (str): additional where filter

        """
        query_embedding = cast(List[float], query.query_embedding)
        where_str = where
        if query.doc_ids:
            if where_str is not None:
                where_str = f"{where_str} AND {f'doc_id IN {format_list_to_string(query.doc_ids)}'}"
            else:
                where_str = f"doc_id IN {format_list_to_string(query.doc_ids)}"

        # TODO: Support other filter types
        if query.filters is not None and len(query.filters.legacy_filters()) > 0:
            where_str = self._append_meta_filter_condition(
                where_str, query.filters.legacy_filters()
            )

        # build query sql
        if query.mode == VectorStoreQueryMode.DEFAULT:
            query_statement = self._config.build_query_statement(
                query_embed=query_embedding,
                where_str=where_str,
                limit=query.similarity_top_k,
            )
        elif query.mode == VectorStoreQueryMode.HYBRID:
            if query.query_str is not None:
                amplify_ratio = self.AMPLIFY_RATIO_LE5
                if 5 < query.similarity_top_k < 50:
                    amplify_ratio = self.AMPLIFY_RATIO_GT5
                if query.similarity_top_k > 50:
                    amplify_ratio = self.AMPLIFY_RATIO_GT50
                query_statement = self._build_hybrid_search_statement(
                    self._config.build_query_statement(
                        query_embed=query_embedding,
                        where_str=where_str,
                        limit=query.similarity_top_k * amplify_ratio,
                    ),
                    query.query_str,
                    query.similarity_top_k,
                )
                logger.debug(f"hybrid query_statement={query_statement}")
            else:
                raise ValueError("query_str must be specified for a hybrid query.")
        elif query.mode == VectorStoreQueryMode.TEXT_SEARCH:
            if query.query_str is not None:
                query_statement = self._build_text_search_statement(
                    query.query_str,
                    query.similarity_top_k,
                )
                logger.debug(f"text query_statement={query_statement}")
            else:
                raise ValueError("query_str must be specified for a text query.")
        else:
            raise ValueError(f"query mode {query.mode!s} not supported")
        nodes = []
        ids = []
        similarities = []
        response = self._client.query(query_statement)
        column_names = response.column_names
        id_idx = column_names.index("id")
        text_idx = column_names.index("text")
        metadata_idx = column_names.index("metadata")
        node_info_idx = column_names.index("node_info")
        score_idx = column_names.index("score")
        for r in response.result_rows:
            start_char_idx = None
            end_char_idx = None

            if isinstance(r[node_info_idx], dict):
                start_char_idx = r[node_info_idx].get("start", None)
                end_char_idx = r[node_info_idx].get("end", None)
            node = TextNode(
                id_=r[id_idx],
                text=r[text_idx],
                metadata=json.loads(r[metadata_idx]),
                start_char_idx=start_char_idx,
                end_char_idx=end_char_idx,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=r[id_idx])
                },
            )

            nodes.append(node)
            similarities.append(r[score_idx])
            ids.append(r[id_idx])
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
