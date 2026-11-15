"""SQLAlchemy should only be imported when SQL features are used."""

import subprocess
import sys

import pytest


def _modules_after(code: str) -> str:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"{code}\nimport sys\nprint('sqlalchemy' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.mark.parametrize(
    "code",
    [
        "import llama_index.core",
        "from llama_index.core import VectorStoreIndex",
        "from llama_index.core.indices import VectorStoreIndex",
        "from llama_index.core.query_engine import RetrieverQueryEngine",
        "from llama_index.core.memory import Memory",
    ],
)
def test_core_import_does_not_load_sqlalchemy(code: str) -> None:
    assert _modules_after(code) == "False"


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("llama_index.core", "SQLDatabase"),
        ("llama_index.core", "SQLDocumentContextBuilder"),
        ("llama_index.core", "SQLContextBuilder"),
        ("llama_index.core.indices", "SQLStructStoreIndex"),
        ("llama_index.core.indices", "GPTSQLStructStoreIndex"),
        ("llama_index.core.indices.struct_store", "SQLContextContainerBuilder"),
        ("llama_index.core.indices.struct_store", "NLSQLTableQueryEngine"),
        ("llama_index.core.query_engine", "NLSQLTableQueryEngine"),
        ("llama_index.core.query_engine", "PGVectorSQLQueryEngine"),
        ("llama_index.core.query_engine", "SQLTableRetrieverQueryEngine"),
        ("llama_index.core.query_engine", "SQLJoinQueryEngine"),
        ("llama_index.core.query_engine", "SQLAutoVectorQueryEngine"),
    ],
)
def test_sql_names_still_importable(module: str, name: str) -> None:
    assert _modules_after(f"from {module} import {name}") == "True"


def test_sql_context_builder_alias() -> None:
    from llama_index.core import SQLContextBuilder, SQLDocumentContextBuilder

    assert SQLContextBuilder is SQLDocumentContextBuilder


def test_unknown_attribute_still_raises() -> None:
    import llama_index.core

    with pytest.raises(AttributeError):
        llama_index.core.DefinitelyNotARealName  # noqa: B018
