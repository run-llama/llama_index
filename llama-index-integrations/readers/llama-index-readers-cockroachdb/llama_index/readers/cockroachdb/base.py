"""CockroachDB reader for pulling rows into LlamaIndex Documents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy
from llama_index.core.bridge.pydantic import Field
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class CockroachDBReader(BaseReader):
    """Read rows from CockroachDB into LlamaIndex Documents.

    Two modes:

    1. ``query=...``: provide a raw SELECT and the reader maps each row to a
       Document using ``text_column`` and ``metadata_columns``.
    2. ``table=...`` + ``text_column=...``: the reader builds
       ``SELECT * FROM table`` and applies the same mapping.

    Examples:
        ```python
        reader = CockroachDBReader.from_params(
            host="localhost", port=26257, database="defaultdb", user="root",
        )
        docs = reader.load_data(
            table="articles",
            text_column="body",
            metadata_columns=["id", "author", "published_at"],
        )
        ```
    """

    uri: str | None = Field(
        default=None,
        description="SQLAlchemy URL, e.g. cockroachdb+psycopg2://root@host:26257/db",
    )
    engine: Any | None = Field(default=None, exclude=True)

    def __init__(
        self,
        uri: str | sqlalchemy.engine.URL | None = None,
        engine: sqlalchemy.engine.Engine | None = None,
    ) -> None:
        super().__init__()
        if uri is None and engine is None:
            raise ValueError("Provide either uri or engine.")
        self.uri = str(uri) if uri is not None else None
        self.engine = engine

    @classmethod
    def from_params(
        cls,
        host: str,
        port: int | str = 26257,
        database: str = "defaultdb",
        user: str = "root",
        password: str | None = None,
        sslmode: str = "verify-full",
        sslrootcert: str | None = None,
    ) -> CockroachDBReader:
        params: list[str] = []
        if sslmode:
            params.append(f"sslmode={sslmode}")
        if sslrootcert:
            params.append(f"sslrootcert={sslrootcert}")
        qs = ("?" + "&".join(params)) if params else ""
        cred = f"{user}:{password}" if password else user
        uri = f"cockroachdb+psycopg2://{cred}@{host}:{port}/{database}{qs}"
        return cls(uri=uri)

    def _get_engine(self) -> sqlalchemy.engine.Engine:
        if self.engine is not None:
            return self.engine
        from sqlalchemy import create_engine

        self.engine = create_engine(self.uri)
        return self.engine

    def load_data(
        self,
        query: str | None = None,
        table: str | None = None,
        schema: str = "public",
        text_column: str = "text",
        metadata_columns: Sequence[str] | None = None,
        id_column: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[Document]:
        if query is None and table is None:
            raise ValueError("Provide either query or table.")
        if query is None:
            cols = "*"
            query = f"SELECT {cols} FROM {schema}.{table}"

        engine = self._get_engine()
        from sqlalchemy import text as sql_text

        docs: list[Document] = []
        with engine.connect() as conn:
            result = conn.execute(sql_text(query), params or {})
            rows = result.mappings().all()
            for row in rows:
                row_dict = dict(row)
                if text_column not in row_dict:
                    raise KeyError(f"text_column {text_column!r} not in row keys {list(row_dict)}")
                content = row_dict.pop(text_column)
                if content is None:
                    continue
                doc_id = (
                    str(row_dict.get(id_column)) if id_column and id_column in row_dict else None
                )
                if metadata_columns is not None:
                    metadata = {k: row_dict.get(k) for k in metadata_columns if k in row_dict}
                else:
                    metadata = {k: v for k, v in row_dict.items() if not _looks_like_blob(v)}
                docs.append(Document(text=str(content), metadata=metadata, id_=doc_id))
        return docs


def _looks_like_blob(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))
