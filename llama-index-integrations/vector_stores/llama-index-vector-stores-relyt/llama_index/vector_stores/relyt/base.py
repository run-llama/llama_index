import logging
from typing import Any, List, Sequence

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult, FilterCondition, MetadataFilters, FilterOperator,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pgvecto_rs.sdk import PGVectoRs, Record
from pydantic import StrictStr
from sqlalchemy import text

logger = logging.getLogger(__name__)
import_err_msg = (
    '`pgvecto_rs.sdk` package not found, please run `pip install "pgvecto_rs[sdk]"`'
)


class RelytVectorStore(BasePydanticVectorStore):
    """Relyt Vector Store.

    Examples:
        `pip install llama-index-vector-stores-relyt`

        ```python
        from llama_index.vector_stores.relyt import RelytVectorStore

        # Setup relyt client
        from pgvecto_rs.sdk import PGVectoRs
        import os

        URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
            port=os.getenv("RELYT_PORT", "5432"),
            host=os.getenv("RELYT_HOST", "localhost"),
            username=os.getenv("RELYT_USER", "postgres"),
            password=os.getenv("RELYT_PASS", "mysecretpassword"),
            db_name=os.getenv("RELYT_NAME", "postgres"),
        )

        client = PGVectoRs(
            db_url=URL,
            collection_name="example",
            dimension=1536,  # Using OpenAI’s text-embedding-ada-002
        )

        # Initialize RelytVectorStore
        vector_store = RelytVectorStore(client=client)
        ```
    """

    stores_text = True

    _client: "PGVectoRs" = PrivateAttr()
    _collection_name: str = PrivateAttr()

    def __init__(self, client: "PGVectoRs", collection_name: str, enable_vector_index: bool) -> None:
        self._client: PGVectoRs = client
        self._collection_name = collection_name
        self._enable_vector_index = enable_vector_index
        self.init_index()
        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        return "RelytStore"

    def init_index(self):
        index_name = f"idx_{self._collection_name}_embedding"
        with self._client._engine.connect() as conn:
            with conn.begin():
                index_query = text(
                    f"""
                        SELECT 1
                        FROM pg_indexes
                        WHERE indexname = '{index_name}';
                    """)
                result = conn.execute(index_query).scalar()
                if not result and self._enable_vector_index:
                    index_statement = text(
                        f"""
                            CREATE INDEX {index_name}
                            ON collection_{self._collection_name}
                            USING vectors (embedding vector_l2_ops)
                            WITH (options = $$
                            optimizing.optimizing_threads = 10
                            segment.max_growing_segment_size = 2000
                            segment.max_sealed_segment_size = 30000000
                            [indexing.hnsw]
                            m=30
                            ef_construction=500
                            $$);
                        """)
                    conn.execute(index_statement)
                index_name = f"meta_{self._collection_name}_embedding"
                index_query = text(
                    f"""
                        SELECT 1
                        FROM pg_indexes
                        WHERE indexname = '{index_name}';
                    """
                )
                result = conn.execute(index_query).scalar()
                if not result:
                    index_statement = text(
                        f""" CREATE INDEX {index_name} ON collection_{self._collection_name} USING gin (meta); """)
                    conn.execute(index_statement)

    @property
    def client(self) -> Any:
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        records = [
            Record(
                id=node.id_,
                text=node.get_content(metadata_mode=MetadataMode.NONE),
                meta=node_to_metadata_dict(node, remove_text=True),
                embedding=node.get_embedding(),
            )
            for node in nodes
        ]

        self._client.insert(records)
        return [node.id_ for node in nodes]

    def delete(self, filters: str, **delete_kwargs: Any) -> None:
        if filters is None:
            raise ValueError("filters cannot be None")

        filter_condition = f"WHERE {filters}"

        with self._client._engine.connect() as conn:
            with conn.begin():
                sql_query = f""" DELETE FROM collection_{self._collection_name} {filter_condition}"""
                conn.execute(text(sql_query))

    def drop(self) -> None:
        self._client.drop()

    def to_postgres_operator(self, operator: FilterOperator) -> str:
        if operator == FilterOperator.EQ:
            return " = "
        elif operator == FilterOperator.GT:
            return " > "
        elif operator == FilterOperator.LT:
            return " < "
        elif operator == FilterOperator.NE:
            return " != "
        elif operator == FilterOperator.GTE:
            return " >= "
        elif operator == FilterOperator.LTE:
            return " <= "
        elif operator == FilterOperator.IN:
            return " in "

    def to_postgres_conditions(self, operator: FilterOperator) -> str:
        if operator == FilterCondition.AND:
            return "AND"
        elif operator == FilterCondition.OR:
            return "OR"

    def transformer_filter(self, filters) -> str:
        filter_statement = ""
        for filter in filters:
            if isinstance(filter, MetadataFilters):
                f_stmt = self.transformer_filter(filter)
                if filter_statement == "":
                    filter_statement = f_stmt
                else:
                    filter_statement += filter.condition + f_stmt
            else:
                key = filter.key
                value = filter.value
                op = filter.operator
                if isinstance(value, StrictStr):
                    value = "'{}'".format(value)
                if op == FilterOperator.IN:
                    new_val = []
                    for v in value:
                        if isinstance(v, StrictStr):
                            new_val.append("'{}'".format(v))
                        else:
                            new_val.append(str(v))
                    value = "(" + ",".join(new_val) + ")"
                filter_cond = key + self.to_postgres_operator(op) + value
                if filter_statement == "":
                    filter_statement = filter_cond
                else:
                    filter_statement += filters.condition + filter_cond
        return filter_statement

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        # Add the filter if provided
        try:
            from sqlalchemy.engine import Row
        except ImportError:
            raise ImportError(
                "Could not import Row from sqlalchemy.engine. "
                "Please 'pip install sqlalchemy>=1.4'."
            )

        embedding = VectorStoreQuery.query_embedding
        k = VectorStoreQuery.similarity_top_k
        filter_condition = ""
        filters = VectorStoreQuery.filters

        if filters is not None:
            filter_condition += f"WHERE {self.transformer_filter(filters)}"

        sql_query = f"""
                        SELECT id, text, meta, embedding <-> :embedding as distance
                        FROM {self._collection_name}
                        {filter_condition}
                        ORDER BY embedding <-> :embedding
                        LIMIT :k
                    """

        # Set up the query parameters
        embedding_str = ", ".join(format(x) for x in embedding)
        embedding_str = "[" + embedding_str + "]"
        params = {"embedding": embedding_str, "k": k}

        # Execute the query and fetch the results
        with self.engine.connect() as conn:
            results: Sequence[Row] = conn.execute(text(sql_query), params).fetchall()

        nodes = [
            metadata_dict_to_node(reocrd.meta, text=reocrd.text)
            for reocrd in results
        ]

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=[r.distance for r in results],
            ids=[str(r.id) for r in results],
        )
