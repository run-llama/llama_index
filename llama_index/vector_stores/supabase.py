from __future__ import annotations

import math
import uuid
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

from flupy import flu
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    and_,
    cast,
    delete,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.dialects import postgresql

from llama_index.vector_stores.exc import (
    ArgError,
    CollectionAlreadyExists,
    CollectionNotFound,
    FilterError,
    Unreachable,
)

if TYPE_CHECKING:
    from llama_index.readers.supabase import Client
    from llama_index.vector_stores.types import VectorStore


MetadataValues = Union[str, int, float, bool, List[str]]
Metadata = Dict[str, MetadataValues]
Numeric = Union[int, float, complex]
Record = Tuple[str, Iterable[Numeric], Metadata]


class IndexMethod(str, Enum):
    ivfflat = "ivfflat"


class IndexMeasure(str, Enum):
    cosine_distance = "cosine_distance"
    l2_distance = "l2_distance"
    max_inner_product = "max_inner_product"


INDEX_MEASURE_TO_OPS = {
    IndexMeasure.cosine_distance: "vector_cosine_ops",
    IndexMeasure.l2_distance: "vector_l2_ops",
    IndexMeasure.max_inner_product: "vector_ip_ops",
}

INDEX_MEASURE_TO_SQLA_ACC = {
    IndexMeasure.cosine_distance: lambda x: x.cosine_distance,
    IndexMeasure.l2_distance: lambda x: x.l2_distance,
    IndexMeasure.max_inner_product: lambda x: x.max_inner_product,
}


class SupabaseStore(VectorStore):
    def __init__(self, name: str, dimension: int, client: Client):
        self.client = client
        self.name = name
        self.dimension = dimension
        self.table = build_table(name, client.meta, dimension)
        self._index: Optional[str] = None

    def __repr__(self):
        return f'vecs.Collection(name="{self.name}", dimension={self.dimension})'

    def __len__(self) -> int:
        with self.client.Session() as sess:
            with sess.begin():
                stmt = select(func.count()).select_from(self.table)
                return sess.execute(stmt).scalar() or 0

    def _create(self):
        existing_collections = self.__class__._list_collections(self.client)
        existing_collection_names = [x.name for x in existing_collections]
        if self.name in existing_collection_names:
            raise CollectionAlreadyExists(
                "Collection with requested name already exists"
            )
        self.table.create(self.client.engine)
        return self

    def _drop(self):
        existing_collections = self.__class__._list_collections(self.client)
        existing_collection_names = [x.name for x in existing_collections]
        if self.name not in existing_collection_names:
            raise CollectionNotFound("Collection with requested name not found")
        self.table.drop(self.client.engine)
        return self

    def upsert(self, vectors: Iterable[Tuple[str, Iterable[Numeric], Metadata]]):
        chunk_size = 500

        with self.client.Session() as sess:
            with sess.begin():
                for chunk in flu(vectors).chunk(chunk_size):
                    stmt = postgresql.insert(self.table).values(chunk)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[self.table.c.id],
                        set_=dict(
                            vec=stmt.excluded.vec, metadata=stmt.excluded.metadata
                        ),
                    )
                    sess.execute(stmt)
        return

    def fetch(self, ids: Iterable[str]) -> List[Record]:
        if isinstance(ids, str):
            raise ArgError("ids must be a list of strings")

        chunk_size = 12
        records = []
        with self.client.Session() as sess:
            with sess.begin():
                for id_chunk in flu(ids).chunk(chunk_size):
                    stmt = select(self.table).where(self.table.c.id.in_(id_chunk))
                    chunk_records = sess.execute(stmt)
                    records.extend(chunk_records)
        return records

    def delete(self, ids: Iterable[str]) -> List[str]:
        if isinstance(ids, str):
            raise ArgError("ids must be a list of strings")

        chunk_size = 12

        del_ids = list(ids)
        ids = []
        with self.client.Session() as sess:
            with sess.begin():
                for id_chunk in flu(del_ids).chunk(chunk_size):
                    stmt = (
                        delete(self.table)
                        .where(self.table.c.id.in_(id_chunk))
                        .returning(self.table.c.id)
                    )
                    ids.extend(sess.execute(stmt).scalars() or [])
        return ids

    def __getitem__(self, items):
        if not isinstance(items, str):
            raise ArgError("items must be a string id")

        row = self.fetch([items])

        if row == []:
            raise KeyError("no item found with requested id")
        return row[0]

    def query(
        self,
        query_vector: Iterable[Numeric],
        limit: int = 10,
        filters: Optional[Dict] = None,
        measure: Union[IndexMeasure, str] = IndexMeasure.cosine_distance,
        include_value: bool = False,
        include_metadata: bool = False,
    ) -> Union[List[Record], List[str]]:
        if limit > 1000:
            raise ArgError("limit must be <= 1000")

        # ValueError on bad input
        try:
            imeasure = IndexMeasure(measure)
        except ValueError:
            raise ArgError("Invalid index measure")

        if not self.is_indexed_for_measure(imeasure):
            warnings.warn(
                f"Query does not have a covering index for {imeasure}. See Collection.create_index"
            )

        distance_lambda = INDEX_MEASURE_TO_SQLA_ACC.get(imeasure)
        if distance_lambda is None:
            # unreachable
            raise ArgError("invalid distance_measure")  # pragma: no cover

        distance_clause = distance_lambda(self.table.c.vec)(query_vector)

        cols = [self.table.c.id]

        if include_value:
            cols.append(distance_clause)

        if include_metadata:
            cols.append(self.table.c.metadata)

        stmt = select(*cols)
        if filters:
            stmt = stmt.filter(build_filters(self.table.c.metadata, filters))  # type: ignore

        stmt = stmt.order_by(distance_clause)
        stmt = stmt.limit(limit)

        with self.client.Session() as sess:
            with sess.begin():
                # index ignored if greater than n_lists
                sess.execute(text("set local ivfflat.probes = 10"))
                if len(cols) == 1:
                    return [str(x) for x in sess.scalars(stmt).fetchall()]
                return sess.execute(stmt).fetchall() or []

    @classmethod
    def _list_collections(cls, client: "Client") -> List["Collection"]:
        query = text(
            """
        select
            relname as table_name,
            atttypmod as embedding_dim
        from
            pg_class pc
            join pg_attribute pa
                on pc.oid = pa.attrelid
        where
            pc.relnamespace = 'vecs'::regnamespace
            and pc.relkind = 'r'
            and pa.attname = 'vec'
            and not pc.relname ^@ '_'
        """
        )
        xc = []
        with client.Session() as sess:
            for name, dimension in sess.execute(query):
                existing_collection = cls(name, dimension, client)
                xc.append(existing_collection)
        return xc

    @property
    def index(self) -> Optional[str]:
        if self._index is None:
            query = text(
                """
            select
                relname as table_name
            from
                pg_class pc
            where
                pc.relnamespace = 'vecs'::regnamespace
                and relname ilike 'ix_vector%'
                and pc.relkind = 'i'
            """
            )
            with self.client.Session() as sess:
                ix_name = sess.execute(query).scalar()
            self._index = ix_name
        return self._index

    def is_indexed_for_measure(self, measure: IndexMeasure):
        index_name = self.index
        if index_name is None:
            return False

        ops = INDEX_MEASURE_TO_OPS.get(measure)
        if ops is None:
            return False

        if ops in index_name:
            return True

        return False

    def create_index(
        self,
        measure: IndexMeasure = IndexMeasure.cosine_distance,
        method: IndexMethod = IndexMethod.ivfflat,
        replace=True,
    ):
        if not method == IndexMethod.ivfflat:
            # at time of writing, no other methods are supported by pgvector
            raise ArgError("invalid index method")

        if replace:
            self._index = None
        else:
            if self.index is not None:
                raise ArgError("replace is set to False but an index exists")

        ops = INDEX_MEASURE_TO_OPS.get(measure)
        if ops is None:
            raise ArgError("Unknown index measure")

        # Clone the table
        clone_table = build_table(f"_{self.name}", self.client.meta, self.dimension)

        # hacky
        try:
            clone_table.drop(self.client.engine)
        except Exception:
            pass

        with self.client.Session() as sess:
            n_records: int = sess.execute(func.count(self.table.c.id)).scalar()  # type: ignore

        with self.client.Session() as sess:
            with sess.begin():
                n_index_seed = min(5000, n_records)
                clone_table.create(sess.connection())
                stmt_seed_table = clone_table.insert().from_select(
                    self.table.c,
                    select(self.table).order_by(func.random()).limit(n_index_seed),
                )
                sess.execute(stmt_seed_table)

                n_lists = (
                    int(max(n_records / 1000, 30))
                    if n_records < 1_000_000
                    else int(math.sqrt(n_records))
                )

                unique_string = str(uuid.uuid4()).replace("-", "_")[0:7]

                sess.execute(
                    text(
                        f"""
                        create index ix_{ops}_{n_lists}_{unique_string}
                          on vecs."{clone_table.name}"
                          using ivfflat (vec {ops}) with (lists={n_lists})
                        """
                    )
                )

                sess.execute(
                    text(
                        f"""
                        create index ix_meta_{unique_string}
                          on vecs."{clone_table.name}"
                          using gin ( metadata )
                        """
                    )
                )

                # Fully populate the table
                stmt = postgresql.insert(clone_table).from_select(
                    self.table.c, select(self.table)
                )
                stmt = stmt.on_conflict_do_nothing()
                sess.execute(stmt)

                # Replace the table
                sess.execute(text(f"drop table vecs.{self.table.name};"))
                sess.execute(
                    text(
                        f"alter table vecs._{self.table.name} rename to {self.table.name};"
                    )
                )


def build_filters(json_col: Column, filters: Dict):
    if not isinstance(filters, dict):
        raise FilterError("filters must be a dict")

    if len(filters) > 1:
        raise FilterError("max 1 entry per filter")

    for key, value in filters.items():
        if not isinstance(key, str):
            raise FilterError("*filters* keys must be strings")

        if key in ("$and", "$or"):
            if not isinstance(value, list):
                raise FilterError(
                    "$and/$or filters must have associated list of conditions"
                )

            if key == "$and":
                return and_(*[build_filters(json_col, subcond) for subcond in value])

            if key == "$or":
                return or_(*[build_filters(json_col, subcond) for subcond in value])

            raise Unreachable()

        if isinstance(value, dict):
            if len(value) > 1:
                raise FilterError("only operator permitted")
            for operator, clause in value.items():
                if operator not in ("$eq", "$ne", "$lt", "$lte", "$gt", "$gte"):
                    raise FilterError("unknown operator")

                matches_value = cast(clause, postgresql.JSONB)

                if operator == "$eq":
                    return json_col.op("->")(key) == matches_value

                if operator == "$ne":
                    return json_col.op("->")(key) != matches_value

                if operator == "$lt":
                    return json_col.op("->")(key) < matches_value

                if operator == "$lte":
                    return json_col.op("->")(key) <= matches_value

                if operator == "$gt":
                    return json_col.op("->")(key) > matches_value

                if operator == "$gte":
                    return json_col.op("->")(key) >= matches_value

                else:
                    raise Unreachable()


def build_table(name: str, meta: MetaData, dimension) -> Table:
    return Table(
        name,
        meta,
        Column("id", String, primary_key=True),
        Column("vec", Vector(dimension), nullable=False),
        Column(
            "metadata",
            postgresql.JSONB,
            server_default=text("'{}'::jsonb"),
            nullable=False,
        ),
        extend_existing=True,
    )
