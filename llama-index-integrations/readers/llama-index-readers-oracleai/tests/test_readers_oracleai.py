import os

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.readers.oracleai import OracleReader, OracleTextSplitter

try:
    import oracledb  # type: ignore
except Exception:
    oracledb = None  # allow collection even if client not installed


def test_class():
    names_of_base_classes = [b.__name__ for b in OracleReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def _env_or_default(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val else default


def _connect_or_skip():
    if oracledb is None:
        pytest.skip("oracledb client not installed")
    # Reuse the same VECDB_* names as the rest of the Oracle integration tests.
    username = _env_or_default("VECDB_USER", "")
    password = _env_or_default("VECDB_PASS", "")
    dsn = _env_or_default("VECDB_HOST", "")
    if not username or not password or not dsn:
        pytest.skip("Oracle test credentials are not set")
    try:
        return oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception as exc:
        pytest.skip(f"Could not connect to Oracle: {exc}")


### Test OracleReader #####
def test_loader_test() -> None:
    connection = _connect_or_skip()
    try:
        cursor = connection.cursor()
        cursor.execute("drop table if exists llama_demo")
        cursor.execute("create table llama_demo(id number, text varchar2(25))")

        rows = [
            (1, "First"),
            (2, "Second"),
            (3, "Third"),
            (4, "Fourth"),
            (5, "Fifth"),
            (6, "Sixth"),
            (7, "Seventh"),
        ]

        cursor.executemany("insert into llama_demo(id, text) values (:1, :2)", rows)
        connection.commit()
        cursor.close()

        # load from database column
        loader_params = {
            "owner": _env_or_default("VECDB_USER", ""),
            "tablename": "llama_demo",
            "colname": "text",
        }
        loader = OracleReader(conn=connection, params=loader_params)
        docs = loader.load()

        # verify
        assert len(docs) != 0
        # print(f"Document#1: {docs[0].text}")
    finally:
        connection.close()


### Test OracleTextSplitter ####
def test_splitter_test() -> None:
    connection = _connect_or_skip()
    try:
        doc = """Llamaindex is a wonderful framework to load, split, chunk
                and embed your data!!"""

        # by words , max = 1000
        splitter_params = {
            "by": "words",
            "max": "1000",
            "overlap": "200",
            "split": "custom",
            "custom_list": [","],
            "extended": "true",
            "normalize": "all",
        }
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)
        chunks = splitter.split_text(doc)

        # verify
        assert len(chunks) != 0
        # print(f"1. Number of chunks: {len(chunks)}")

        # by chars , max = 4000
        splitter_params = {
            "by": "chars",
            "max": "4000",
            "overlap": "800",
            "split": "NEWLINE",
            "normalize": "all",
        }
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)
        chunks = splitter.split_text(doc)

        # verify
        assert len(chunks) != 0
        # print(f"2. Number of chunks: {len(chunks)}")

        # by words , max = 10
        splitter_params = {
            "by": "words",
            "max": "10",
            "overlap": "2",
            "split": "SENTENCE",
        }
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)
        chunks = splitter.split_text(doc)

        # verify
        assert len(chunks) != 0
        # print(f"3. Number of chunks: {len(chunks)}")

        # by chars , max = 50
        splitter_params = {
            "by": "chars",
            "max": "50",
            "overlap": "10",
            "split": "SPACE",
            "normalize": "all",
        }
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)
        chunks = splitter.split_text(doc)

        # verify
        assert len(chunks) != 0
        # print(f"4. Number of chunks: {len(chunks)}")
    finally:
        connection.close()


# test loader and splitter
# test_loader_test()
# test_splitter_test()
