import contextlib
import logging
import uuid
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import sqlalchemy
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy.types import Float, UserDefinedType

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

_logger = logging.getLogger(__name__)


DEFAULT_COLLECTION_NAME = "llama_index_tidb_vector"
Base = declarative_base()  # type: Any


class VectorType(UserDefinedType):
    cache_ok = True

    def __init__(self, dim=None) -> None:
        super(UserDefinedType, self).__init__()
        self.dim = None

    def get_col_spec(self, **kw):
        if self.dim is None:
            return "VECTOR<FLOAT>"
        return "VECTOR(%d)" % self.dim

    def bind_processor(self, dialect):
        def process(value, dim=None):
            if value is None:
                return value

            if isinstance(value, np.ndarray):
                if value.ndim != 1:
                    raise ValueError("expected ndim to be 1")

                if not np.issubdtype(value.dtype, np.integer) and not np.issubdtype(
                    value.dtype, np.floating
                ):
                    raise ValueError("dtype must be numeric")

                value = value.tolist()

            if dim is not None and len(value) != dim:
                raise ValueError("expected %d dimensions, not %d" % (dim, len(value)))

            return "[" + ",".join([str(float(v)) for v in value]) + "]"

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None or isinstance(value, np.ndarray):
                return value

            return np.array(value[1:-1].split(","), dtype=np.float32)

        return process

    class comparator_factory(UserDefinedType.Comparator):
        def l2_distance(self, other):
            return self.op("<-->", return_type=Float)(other)

        def cosine_distance(self, other):
            return self.op("<==>", return_type=Float)(other)


_classes: Any = None


def _create_vector_model(table_name: str):
    global _classes
    if _classes is not None:
        return _classes

    class SQLVectorModel(Base):
        __tablename__ = table_name
        id = sqlalchemy.Column(
            sqlalchemy.String(36), primary_key=True, default=lambda: str(uuid.uuid4())
        )
        embedding = sqlalchemy.Column(VectorType())
        document = sqlalchemy.Column(sqlalchemy.Text, nullable=True)
        meta = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)

    _classes = SQLVectorModel
    return _classes


class TiDBVector(BasePydanticVectorStore):
    stores_text = True
    flat_metadata = False

    _connection_string: str = PrivateAttr()
    _engine_args: Dict[str, Any] = PrivateAttr()
    _bind: sqlalchemy.engine.Engine = PrivateAttr()
    _table_model: Any = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)
    _pre_delete_collection: bool = PrivateAttr(default=False)

    def __init__(
        self,
        connection_string: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._connection_string = connection_string
        self._engine_args = engine_args or {}
        self._pre_delete_collection = pre_delete_collection
        self._table_model = _create_vector_model(collection_name)

    def __del__(self) -> None:
        if isinstance(self._bind, sqlalchemy.engine.Connection):
            self._bind.close()

    def create_table_if_not_exists(self) -> None:
        if self._pre_delete_collection:
            self.drop_table()
        with Session(self._bind) as session, session.begin():
            Base.metadata.create_all(session.get_bind())
            # wait for tidb support vector index

    def drop_table(self) -> None:
        with Session(self._bind) as session, session.begin():
            Base.metadata.drop_all(session.get_bind())

    def _create_engine(self) -> sqlalchemy.engine.Engine:
        return sqlalchemy.create_engine(
            url=self._connection_string, **self._engine_args
        )

    def _initialize(self) -> None:
        if not self._is_initialized:
            self._bind = self._create_engine()
            self.create_table_if_not_exists()
            self._is_initialized = True

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session, bind to _conn string."""
        yield Session(self._bind)

    @classmethod
    def class_name(cls) -> str:
        return "TiDBVector"

    @property
    def client(self) -> Any:
        if not self._is_initialized:
            return None
        return self._bind

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        self._initialize()
        ids = []
        with Session(self._bind) as session:
            for node in nodes:
                embeded_doc = self._table_model(
                    id=node.node_id,
                    embedding=node.get_embedding(),
                    document=node.get_content(metadata_mode=MetadataMode.NONE),
                    meta=node_to_metadata_dict(
                        node,
                        remove_text=True,
                        flat_metadata=self.flat_metadata,
                    ),
                )
                ids.append(node.node_id)
                session.add(embeded_doc)
            session.commit()

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._initialize()
        with Session(self._bind) as session:
            condition = self._table_model.meta["doc_id"] == ref_doc_id

            stmt = sqlalchemy.delete(self._table_model).where(condition)
            session.execute(stmt)
            session.commit()

    def _similarity_search_with_score(
        self,
        embedding: List[float],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        with Session(self._bind) as session:
            results: List[Any] = (
                session.query(
                    self._table_model.id,
                    self._table_model.meta,
                    self._table_model.document,
                    self._table_model.embedding.cosine_distance(embedding).label(
                        "distance"
                    ),
                )
                .order_by(sqlalchemy.asc("distance"))
                .limit(limit)
                .all()
            )

        nodes = []
        similarities = []
        ids = []
        for row in results:
            try:
                node = metadata_dict_to_node(row.meta)
                node.set_content(str(row.document))
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                _logger.warning(
                    "Failed to parse metadata dict, falling back to legacy logic."
                )
                node = TextNode(
                    id_=row.id,
                    text=row.document,
                    metadata=row.meta,
                )
            similarities.append((1 - row.distance) if row.distance is not None else 0)
            ids.append(row.id)
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        self._initialize()
        return self._similarity_search_with_score(
            query.query_embedding,
            query.similarity_top_k,
            query.filters,
            **kwargs,
        )
