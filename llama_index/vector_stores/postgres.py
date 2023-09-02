from typing import List, Any, Type, Optional
from collections import namedtuple

from llama_index.schema import MetadataMode, TextNode
from llama_index.vector_stores.types import (
    VectorStore,
    NodeWithEmbedding,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    MetadataFilters,
)
from llama_index.vector_stores.utils import node_to_metadata_dict, metadata_dict_to_node

DBEmbeddingRow = namedtuple(
    "DBEmbeddingRow", ["node_id", "text", "metadata", "similarity"]
)


def get_data_model(
    base: Type,
    index_name: str,
    hybrid_search: bool,
    text_search_config: str,
    embed_dim: int = 1536,
) -> Any:
    """
    This part create a dynamic sqlalchemy model with a new table
    """
    from pgvector.sqlalchemy import Vector
    from sqlalchemy import Column, Computed
    from sqlalchemy.dialects.postgresql import BIGINT, VARCHAR, JSON
    from sqlalchemy.schema import Index
    from sqlalchemy.sql import func

    from sqlalchemy.dialects.postgresql import TSVECTOR
    from sqlalchemy.types import TypeDecorator

    class TSVector(TypeDecorator):
        impl = TSVECTOR

    tablename = "data_%s" % index_name  # dynamic table name
    class_name = "Data%s" % index_name  # dynamic class name

    if hybrid_search:

        class HybridAbstractData(base):  # type: ignore
            __abstract__ = True  # this line is necessary
            id = Column(BIGINT, primary_key=True, autoincrement=True)
            text = Column(VARCHAR, nullable=False)
            metadata_ = Column(JSON)
            node_id = Column(VARCHAR)
            embedding = Column(Vector(embed_dim))  # type: ignore
            text_search_tsv = Column(  # type: ignore
                TSVector(),
                Computed(
                    "to_tsvector('%s', text)" % text_search_config, persisted=True
                ),
            )

        model = type(class_name, (HybridAbstractData,), {"__tablename__": tablename})

        Index("text_search_tsv_idx", model.text_search_tsv, postgresql_using="gin")
    else:

        class AbstractData(base):  # type: ignore
            __abstract__ = True  # this line is necessary
            id = Column(BIGINT, primary_key=True, autoincrement=True)
            text = Column(VARCHAR, nullable=False)
            metadata_ = Column(JSON)
            node_id = Column(VARCHAR)
            embedding = Column(Vector(embed_dim))  # type: ignore

        model = type(class_name, (AbstractData,), {"__tablename__": tablename})

    return model


class PGVectorStore(VectorStore):
    from sqlalchemy.sql.selectable import Select

    stores_text = True
    flat_metadata = False

    def __init__(
        self,
        connection_string: str,
        async_connection_string: str,
        table_name: str,
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
    ) -> None:
        try:
            import sqlalchemy  # noqa: F401
            import pgvector  # noqa: F401
            import psycopg2  # noqa: F401
            import asyncpg  # noqa: F401
            import sqlalchemy.ext.asyncio  # noqa: F401
        except ImportError:
            raise ImportError(
                "`sqlalchemy[asyncio]`, `pgvector`, `psycopg2-binary` and `asyncpg` "
                "packages should be pre installed"
            )

        self.connection_string = connection_string
        self.async_connection_string = async_connection_string
        self.table_name: str = table_name.lower()
        self._hybrid_search = hybrid_search
        self._text_search_config = text_search_config

        if self._hybrid_search and text_search_config is None:
            raise ValueError(
                "Sparse vector index creation requires "
                "a text search configuration specification."
            )

        # def __enter__(self):
        from sqlalchemy.orm import declarative_base

        self._base = declarative_base()
        # sqlalchemy model
        self.table_class = get_data_model(
            self._base,
            self.table_name,
            self._hybrid_search,
            self._text_search_config,
            embed_dim=embed_dim,
        )
        self._connect()
        self._create_extension()
        self._create_tables_if_not_exists()

    async def close(self) -> None:
        self._session.close_all()
        self._engine.dispose()

        await self._async_engine.dispose()

    @classmethod
    def from_params(
        cls,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        table_name: str,
        hybrid_search: bool = False,
        embed_dim: int = 1536,
    ) -> "PGVectorStore":
        """Return connection string from database parameters."""
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        async_conn_str = (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        )
        return cls(
            connection_string=conn_str,
            async_connection_string=async_conn_str,
            table_name=table_name,
            hybrid_search=hybrid_search,
            embed_dim=embed_dim,
        )

    def _connect(self) -> Any:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.ext.asyncio import async_sessionmaker

        self._engine = create_engine(self.connection_string)
        self._session = sessionmaker(self._engine)

        self._async_engine = create_async_engine(self.async_connection_string)
        self._async_session = async_sessionmaker(self._async_engine)

    def _create_tables_if_not_exists(self) -> None:
        with self._session() as session:
            with session.begin():
                self._base.metadata.create_all(session.connection())

    def _create_extension(self) -> None:
        import sqlalchemy

        with self._session() as session:
            with session.begin():
                statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
                session.execute(statement)
                session.commit()

    def _node_to_table_row(self, node: NodeWithEmbedding) -> Any:
        return self.table_class(
            node_id=node.id,
            embedding=node.embedding,
            text=node.node.get_content(metadata_mode=MetadataMode.NONE),
            metadata_=node_to_metadata_dict(
                node.node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            ),
        )

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        ids = []
        with self._session() as session:
            with session.begin():
                for result in embedding_results:
                    ids.append(result.id)
                    item = self._node_to_table_row(result)
                    session.add(item)
                session.commit()
        return ids

    async def async_add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        ids = []
        async with self._async_session() as session:
            async with session.begin():
                for result in embedding_results:
                    ids.append(result.id)
                    item = self._node_to_table_row(result)
                    session.add(item)
                await session.commit()
        return ids

    def _apply_filters_and_limit(
        self,
        stmt: Select,
        limit: int,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> Any:
        import sqlalchemy

        if metadata_filters:
            for filter_ in metadata_filters.filters:
                bind_parameter = f"value_{filter_.key}"
                stmt = stmt.where(  # type: ignore
                    sqlalchemy.text(f"metadata_->>'{filter_.key}' = :{bind_parameter}")
                )
                stmt = stmt.params(  # type: ignore
                    **{bind_parameter: str(filter_.value)}
                )
        return stmt.limit(limit)  # type: ignore

    def _build_query(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> Any:
        import sqlalchemy
        from sqlalchemy import select

        stmt = select(  # type: ignore
            self.table_class, self.table_class.embedding.cosine_distance(embedding)
        ).order_by(self.table_class.embedding.cosine_distance(embedding))

        return self._apply_filters_and_limit(stmt, limit, metadata_filters)

    def _query_with_score(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_query(embedding, limit, metadata_filters)
        with self._session() as session:
            with session.begin():
                res = session.execute(stmt)
                return [
                    DBEmbeddingRow(
                        node_id=item.node_id,
                        text=item.text,
                        metadata=item.metadata_,
                        similarity=(1 - distance) if distance is not None else 0,
                    )
                    for item, distance in res.all()
                ]

    async def _aquery_with_score(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_query(embedding, limit, metadata_filters)
        async with self._async_session() as async_session:
            async with async_session.begin():
                res = await async_session.execute(stmt)
                return [
                    DBEmbeddingRow(
                        node_id=item.node_id,
                        text=item.text,
                        metadata=item.metadata_,
                        similarity=(1 - distance) if distance is not None else 0,
                    )
                    for item, distance in res.all()
                ]

    def _build_sparse_query(
        self,
        query_str: Optional[str],
        limit: int,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> Any:
        from sqlalchemy import select
        from sqlalchemy.sql import func, text

        if query_str is None:
            raise ValueError("query_str must be specified for a sparse vector query.")

        stmt = (
            select(  # type: ignore
                self.table_class,
                func.ts_rank(
                    self.table_class.text_search_tsv, func.plainto_tsquery(query_str)
                ).label("rank"),
            )
            .where(self.table_class.text_search_tsv.match(query_str))
            .order_by(text("rank desc"))
        )

        return self._apply_filters_and_limit(stmt, limit, metadata_filters)  # type: ignore

    async def _async_sparse_query_with_rank(
        self,
        query: VectorStoreQuery,
        limit: int,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_sparse_query(query.query_str, limit, metadata_filters)
        async with self._async_session() as async_session:
            async with async_session.begin():
                res = await async_session.execute(stmt)
                return [
                    DBEmbeddingRow(
                        node_id=item.node_id,
                        text=item.text,
                        metadata=item.metadata_,
                        similarity=rank,
                    )
                    for item, rank in res.all()
                ]

    def _sparse_query_with_rank(
        self,
        query: VectorStoreQuery,
        limit: int,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_sparse_query(query.query_str, limit, metadata_filters)
        with self._session() as session:
            with session.begin():
                res = session.execute(stmt)
                return [
                    DBEmbeddingRow(
                        node_id=item.node_id,
                        text=item.text,
                        metadata=item.metadata_,
                        similarity=rank,
                    )
                    for item, rank in res.all()
                ]

    async def _async_hybrid_query(
        self, query: VectorStoreQuery
    ) -> List[VectorStoreQueryResult]:
        import asyncio

        results = await asyncio.gather(
            self._aquery_with_score(
                query.query_embedding, query.similarity_top_k, query.filters
            ),
            self._async_sparse_query_with_rank(
                query, query.similarity_top_k, query.filters
            ),
        )

        alpha = query.alpha if query.alpha is not None else 1
        weights = [alpha, 1 - alpha]
        results = [
            [(res, weights[idx]) for res in result]
            for idx, result in enumerate(results)
        ]

        return results[: query.similarity_top_k]

    def _hybrid_query(self, query: VectorStoreQuery) -> List[DBEmbeddingRow]:
        alpha = query.alpha if query.alpha is not None else 1

        dense_results = self._query_with_score(
            query.query_embedding, query.similarity_top_k, query.filters
        )
        dense_results = [(res, alpha) for res in dense_results]

        sparse_results = self._sparse_query_with_rank(
            query, query.similarity_top_k, query.filters
        )
        sparse_results = [(res, 1 - alpha) for res in sparse_results]

        combined_results = [dense_results, sparse_results]

        return combined_results[: query.similarity_top_k]

    def _db_rows_to_query_result(
        self, rows: List[DBEmbeddingRow]
    ) -> VectorStoreQueryResult:
        nodes = []
        similarities = []
        ids = []
        for db_embedding_row in rows:
            try:
                node = metadata_dict_to_node(db_embedding_row.metadata)
                node.set_content(str(db_embedding_row.text))
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                node = TextNode(
                    id_=db_embedding_row.node_id,
                    text=db_embedding_row.text,
                    metadata=db_embedding_row.metadata,
                )
            similarities.append(db_embedding_row.similarity)
            ids.append(db_embedding_row.node_id)
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        if query.mode is VectorStoreQueryMode.HYBRID:
            results = await self._async_hybrid_query(query)
        else:
            results = await self._aquery_with_score(
                query.query_embedding, query.similarity_top_k, query.filters
            )

        return self._db_rows_to_query_result(results)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.mode is VectorStoreQueryMode.HYBRID:
            results = self._hybrid_query(query)
        else:
            results = self._query_with_score(
                query.query_embedding, query.similarity_top_k, query.filters
            )

        return self._db_rows_to_query_result(results)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        import sqlalchemy

        with self._session() as session:
            with session.begin():
                stmt = sqlalchemy.text(
                    f"DELETE FROM public.data_{self.table_name} where "
                    f"(metadata_->>'doc_id')::text = '{ref_doc_id}' "
                )

                session.execute(stmt)
                session.commit()
