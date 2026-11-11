"""
Unit tests for llama_index.utils.oracleai.

These use a duck-typed connection/cursor double, so no Oracle database and no
live connection are required.
"""

import logging

import pytest
from llama_index.core.schema import Document
from llama_index.utils.oracleai import OracleSummary

SUMMARY_PARAMS = {
    "provider": "database",
    "glevel": "S",
    "numParagraphs": 1,
    "language": "english",
}


class FakeVar:
    """Stands in for the CLOB bind variable returned by cursor.var()."""

    def __init__(self, value: str) -> None:
        self._value = value

    def getvalue(self) -> str:
        return self._value


class FakeCursor:
    def __init__(self, calls: list, var_value: str = "a summary") -> None:
        self._calls = calls
        self._var_value = var_value
        self.closed = False

    def var(self, _type):
        return FakeVar(self._var_value)

    def execute(self, statement, **binds):
        self._calls.append((statement, binds))

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    """Records every statement executed so tests can assert on SQL and binds."""

    def __init__(self, var_value: str = "a summary") -> None:
        self.calls: list = []
        self.cursor_obj: FakeCursor | None = None
        self._var_value = var_value

    def cursor(self) -> FakeCursor:
        self.cursor_obj = FakeCursor(self.calls, self._var_value)
        return self.cursor_obj


class DeadConnection:
    """A connection whose cursor() itself fails, e.g. dropped or pool exhausted."""

    def cursor(self):
        raise RuntimeError("DPY-1001: not connected to database")


def _summary(conn, **kwargs) -> OracleSummary:
    return OracleSummary(conn=conn, params=SUMMARY_PARAMS, **kwargs)


def test_none_input_returns_none() -> None:
    conn = FakeConnection()
    assert _summary(conn).get_summary(None) is None
    assert conn.calls == []


def test_str_input() -> None:
    conn = FakeConnection()
    assert _summary(conn).get_summary("some text") == ["a summary"]
    assert len(conn.calls) == 1
    statement, binds = conn.calls[0]
    assert "dbms_vector_chain.utl_to_summary" in statement
    assert binds["data"] == "some text"
    assert conn.cursor_obj.closed is True


def test_document_input() -> None:
    conn = FakeConnection()
    assert _summary(conn).get_summary(Document(text="doc text")) == ["a summary"]
    assert conn.calls[0][1]["data"] == "doc text"


def test_list_of_str_input() -> None:
    conn = FakeConnection()
    assert _summary(conn).get_summary(["a", "b"]) == ["a summary", "a summary"]
    assert [binds["data"] for _, binds in conn.calls] == ["a", "b"]


def test_list_of_document_input() -> None:
    conn = FakeConnection()
    docs = [Document(text="x"), Document(text="y")]
    assert _summary(conn).get_summary(docs) == ["a summary", "a summary"]
    assert [binds["data"] for _, binds in conn.calls] == ["x", "y"]


def test_all_branches_share_one_statement_text() -> None:
    """All input shapes must issue the identical PL/SQL text so Oracle caches one."""
    statements = set()
    for docs in ("s", Document(text="d"), ["a"], [Document(text="b")]):
        conn = FakeConnection()
        _summary(conn).get_summary(docs)
        statements.update(statement for statement, _ in conn.calls)
    assert len(statements) == 1


def test_proxy_is_set_before_the_summary_call() -> None:
    conn = FakeConnection()
    _summary(conn, proxy="http://proxy:80").get_summary("some text")
    assert "utl_http.set_proxy" in conn.calls[0][0]
    assert conn.calls[0][1]["proxy"] == "http://proxy:80"
    assert len(conn.calls) == 2


def test_invalid_top_level_input_type() -> None:
    conn = FakeConnection()
    with pytest.raises(Exception, match="Invalid input type"):
        _summary(conn).get_summary(42)
    assert conn.cursor_obj.closed is True


def test_invalid_item_inside_list() -> None:
    conn = FakeConnection()
    with pytest.raises(Exception, match="Invalid input type"):
        _summary(conn).get_summary(["ok", 42])
    assert conn.cursor_obj.closed is True


def test_cursor_creation_failure_propagates_the_real_error() -> None:
    """
    Regression: the real Oracle error must not be masked by UnboundLocalError.

    cursor.close() used to live in the except block, so when self.conn.cursor()
    itself raised, `cursor` was unbound and close() raised UnboundLocalError,
    replacing the exception the caller needed to see.
    """
    with pytest.raises(RuntimeError, match="DPY-1001"):
        _summary(DeadConnection()).get_summary("some text")


def test_execute_failure_closes_the_cursor_and_reraises() -> None:
    conn = FakeConnection()

    def boom(statement, **binds):
        raise RuntimeError("ORA-00942: table or view does not exist")

    original_cursor = conn.cursor

    def patched_cursor():
        cursor = original_cursor()
        cursor.execute = boom
        return cursor

    conn.cursor = patched_cursor
    with pytest.raises(RuntimeError, match="ORA-00942"):
        _summary(conn).get_summary("some text")
    assert conn.cursor_obj.closed is True


def test_failure_is_reported_through_logging_not_stdout(caplog, capsys) -> None:
    """The error must be reachable by the application's logging config."""
    with caplog.at_level(logging.ERROR, logger="llama_index.utils.oracleai.base"):
        with pytest.raises(RuntimeError):
            _summary(DeadConnection()).get_summary("some text")
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert "DPY-1001" in caplog.text
    assert capsys.readouterr().out == ""
