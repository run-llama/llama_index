"""Structured store indices."""

import importlib
from typing import Any

from llama_index.core.indices.struct_store.json_query import JSONQueryEngine
from llama_index.core.indices.struct_store.pandas import (
    GPTPandasIndex,
    PandasIndex,
)

__all__ = [
    "SQLStructStoreIndex",
    "SQLContextContainerBuilder",
    "PandasIndex",
    "NLStructStoreQueryEngine",
    "SQLStructStoreQueryEngine",
    "JSONQueryEngine",
    # legacy
    "GPTSQLStructStoreIndex",
    "GPTPandasIndex",
    "GPTNLStructStoreQueryEngine",
    "GPTSQLStructStoreQueryEngine",
    "SQLTableRetrieverQueryEngine",
    "NLSQLTableQueryEngine",
]


# SQL support imports SQLAlchemy, which is slow to load, so these names are
# resolved on first access (PEP 562) instead of at package import time.
_LAZY_IMPORTS = {
    "GPTSQLStructStoreIndex": "llama_index.core.indices.struct_store.sql",
    "SQLContextContainerBuilder": "llama_index.core.indices.struct_store.sql",
    "SQLStructStoreIndex": "llama_index.core.indices.struct_store.sql",
    "GPTNLStructStoreQueryEngine": "llama_index.core.indices.struct_store.sql_query",
    "GPTSQLStructStoreQueryEngine": "llama_index.core.indices.struct_store.sql_query",
    "NLSQLTableQueryEngine": "llama_index.core.indices.struct_store.sql_query",
    "NLStructStoreQueryEngine": "llama_index.core.indices.struct_store.sql_query",
    "SQLStructStoreQueryEngine": "llama_index.core.indices.struct_store.sql_query",
    "SQLTableRetrieverQueryEngine": "llama_index.core.indices.struct_store.sql_query",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        value = getattr(importlib.import_module(_LAZY_IMPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
