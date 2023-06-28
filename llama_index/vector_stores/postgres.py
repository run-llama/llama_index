from typing import List, Any, Type, Optional

from llama_index.schema import MetadataMode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.types import (
    VectorStore,
    NodeWithEmbedding,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import node_to_metadata_dict, metadata_dict_to_node


def get_data_model(base: Type, index_name: str) -> Any:
    """
    This part create a dynamic sqlalchemy model with a new table
    """
    from pgvector.sqlalchemy import Vector
    from sqlalchemy import Column
    from sqlalchemy.dialects.postgresql import BIGINT, VARCHAR, JSON

    class AbstractData(base):  # type: ignore
        __abstract__ = True  # tShis line is necessary
        id = Column(BIGINT, primary_key=True, autoincrement=True)
        text = Column(VARCHAR, nullable=False)
        metadata_ = Column(JSON)
        doc_id = Column(VARCHAR)  # TODO: change to node_id
        embedding = Column(Vector(1536))  # type: ignore

    tablename = "data_%s" % index_name  # dynamic table name
    class_name = "Data%s" % index_name  # dynamic class name
    model = type(class_name, (AbstractData,), {"__tablename__": tablename})
    return model


class PGVectorStore(VectorStore):
    stores_text = True
    flat_metadata = False

    def __init__(self, connection_string: str, table_name: str) -> None:
        try:
            import sqlalchemy  # noqa: F401
            import pgvector  # noqa: F401
            import psycopg2  # noqa: F401
        except ImportError:
            raise ImportError(
                """`sqlalchemy`, `pgvector`, and 
            `psycopg2-binary` packages should be pre installed"""
            )

        self.connection_string = connection_string
        self.table_name: str = table_name.lower()
        self._conn: Any

        # def __enter__(self):
        from sqlalchemy.orm import declarative_base

        self._conn = self._connect()
        self._base = declarative_base()
        # sqlalchemy model
        self.table_class = get_data_model(self._base, self.table_name)
        self._create_extension()
        self._create_tables_if_not_exists()

    def __del__(self) -> None:
        self._conn.close()
        self.engine.dispose()

    @classmethod
    def from_params(
        cls,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        table_name: str,
    ) -> "PGVectorStore":
        """Return connection string from database parameters."""
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        return cls(connection_string=conn_str, table_name=table_name)

    def _connect(self) -> Any:
        import sqlalchemy

        self.engine = sqlalchemy.create_engine(self.connection_string)
        conn: sqlalchemy.engine.Connection = self.engine.connect()

        return conn

    def _create_tables_if_not_exists(self) -> None:
        with self._conn.begin():
            self._base.metadata.create_all(self._conn)

    def _create_extension(self) -> None:
        from sqlalchemy.orm import Session
        import sqlalchemy

        with Session(self._conn) as session:
            statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
            session.execute(statement)
            session.commit()

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        from sqlalchemy.orm import Session

        ids = []
        with Session(self._conn) as session:
            for result in embedding_results:
                ids.append(result.id)

                item = self.table_class(
                    id_=result.id,
                    embedding=result.embedding,
                    text=result.node.get_content(metadata_mode=MetadataMode.NONE),
                    metadata_=node_to_metadata_dict(
                        result.node, remove_text=True, flat_metadata=self.flat_metadata
                    ),
                )
                session.add(item)
            session.commit()
        return ids

    def _query_with_score(
        self, embedding: Optional[List[float]], limit: int = 10
    ) -> List[Any]:
        from sqlalchemy.orm import Session

        with Session(self._conn) as session:
            res = (
                session.query(
                    self.table_class,
                    self.table_class.embedding.l2_distance(embedding),  # type: ignore
                )
                .order_by(self.table_class.embedding.l2_distance(embedding))
                .limit(limit)
            )  # type: ignore
        return res.all()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        results = self._query_with_score(query.query_embedding, query.similarity_top_k)
        nodes = []
        similarities = []
        ids = []
        for item, sim in results:
            try:
                node = metadata_dict_to_node(item.metadata_)
                node.set_content(str(item.text))
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                node = TextNode(
                    id_=item.doc_id,
                    text=item.text,
                    metadata=item.metadata_,
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id=item.doc_id),
                    },
                )
            similarities.append(sim)
            ids.append(item.doc_id)
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        from sqlalchemy.orm import Session
        import sqlalchemy

        with Session(self._conn) as session:
            stmt = sqlalchemy.delete(self.table_class).where(
                self.table_class.doc_id == ref_doc_id
            )
            session.execute(stmt)
            session.commit()
