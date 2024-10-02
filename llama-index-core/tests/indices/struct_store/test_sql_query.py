from typing import Any, Dict, Tuple

import pytest
from llama_index.core.async_utils import asyncio_run
from llama_index.core.indices.struct_store.base import default_output_parser
from llama_index.core.indices.struct_store.sql import SQLStructStoreIndex
from llama_index.core.indices.struct_store.sql_query import (
    NLSQLTableQueryEngine,
    NLStructStoreQueryEngine,
    SQLStructStoreQueryEngine,
)
from llama_index.core.schema import Document
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
from sqlalchemy.exc import OperationalError


def test_sql_index_query(
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test SQLStructStoreIndex."""
    index_kwargs, query_kwargs = struct_kwargs
    docs = [Document(text="user_id:2,foo:bar"), Document(text="user_id:8,foo:hello")]
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData()
    table_name = "test_table"
    # NOTE: table is created by tying to metadata_obj
    Table(
        table_name,
        metadata_obj,
        Column("user_id", Integer, primary_key=True),
        Column("foo", String(16), nullable=False),
    )
    metadata_obj.create_all(engine)
    sql_database = SQLDatabase(engine)
    # NOTE: we can use the default output parser for this
    index = SQLStructStoreIndex.from_documents(
        docs,
        sql_database=sql_database,
        table_name=table_name,
        **index_kwargs,
    )

    # query the index with SQL
    sql_to_test = "SELECT user_id, foo FROM test_table"
    sql_query_engine = SQLStructStoreQueryEngine(index, **query_kwargs)
    response = sql_query_engine.query(sql_to_test)
    assert str(response) == "[(2, 'bar'), (8, 'hello')]"

    # query the index with natural language
    nl_query_engine = NLStructStoreQueryEngine(index, **query_kwargs)
    response = nl_query_engine.query("test_table:user_id,foo")
    assert str(response) == "[(2, 'bar'), (8, 'hello')]"

    nl_table_engine = NLSQLTableQueryEngine(index.sql_database)
    response = nl_table_engine.query("test_table:user_id,foo")
    assert str(response) == "[(2, 'bar'), (8, 'hello')]"

    with pytest.raises(NotImplementedError, match="invalid SQL") as exc_info:
        sql_query_engine.query("LLM didn't provide SQL at all")
    assert isinstance(exc_info.value.__cause__, OperationalError)

    ## sql_only=True tests
    # query the index with SQL
    sql_query_engine = SQLStructStoreQueryEngine(index, sql_only=True, **query_kwargs)
    response = sql_query_engine.query(sql_to_test)
    assert str(response) == sql_to_test

    # query the index with natural language
    nl_query_engine = NLStructStoreQueryEngine(index, sql_only=True, **query_kwargs)
    response = nl_query_engine.query("test_table:user_id,foo")
    assert str(response) == sql_to_test

    nl_table_engine = NLSQLTableQueryEngine(index.sql_database, sql_only=True)
    response = nl_table_engine.query("test_table:user_id,foo")
    assert str(response) == sql_to_test

    # query with markdown return
    nl_table_engine = NLSQLTableQueryEngine(
        index.sql_database, synthesize_response=False, markdown_response=True
    )
    response = nl_table_engine.query("test_table:user_id,foo")
    assert (
        str(response)
        == """| user_id | foo |
|---|---|
| 2 | bar |
| 8 | hello |"""
    )


def test_sql_index_async_query(
    allow_networking: Any,
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test SQLStructStoreIndex."""
    index_kwargs, query_kwargs = struct_kwargs
    docs = [Document(text="user_id:2,foo:bar"), Document(text="user_id:8,foo:hello")]
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData()
    table_name = "test_table"
    # NOTE: table is created by tying to metadata_obj
    Table(
        table_name,
        metadata_obj,
        Column("user_id", Integer, primary_key=True),
        Column("foo", String(16), nullable=False),
    )
    metadata_obj.create_all(engine)
    sql_database = SQLDatabase(engine)
    # NOTE: we can use the default output parser for this
    index = SQLStructStoreIndex.from_documents(
        docs,
        sql_database=sql_database,
        table_name=table_name,
        **index_kwargs,
    )

    sql_to_test = "SELECT user_id, foo FROM test_table"
    # query the index with SQL
    sql_query_engine = SQLStructStoreQueryEngine(index, **query_kwargs)
    task = sql_query_engine.aquery(sql_to_test)
    response = asyncio_run(task)
    assert str(response) == "[(2, 'bar'), (8, 'hello')]"

    # query the index with natural language
    nl_query_engine = NLStructStoreQueryEngine(index, **query_kwargs)
    task = nl_query_engine.aquery("test_table:user_id,foo")
    response = asyncio_run(task)
    assert str(response) == "[(2, 'bar'), (8, 'hello')]"

    nl_table_engine = NLSQLTableQueryEngine(index.sql_database)
    task = nl_table_engine.aquery("test_table:user_id,foo")
    response = asyncio_run(task)
    assert str(response) == "[(2, 'bar'), (8, 'hello')]"

    ## sql_only = True  ###
    # query the index with SQL
    sql_query_engine = SQLStructStoreQueryEngine(index, sql_only=True, **query_kwargs)
    task = sql_query_engine.aquery(sql_to_test)
    response = asyncio_run(task)
    assert str(response) == sql_to_test

    # query the index with natural language
    nl_query_engine = NLStructStoreQueryEngine(index, sql_only=True, **query_kwargs)
    task = nl_query_engine.aquery("test_table:user_id,foo")
    response = asyncio_run(task)
    assert str(response) == sql_to_test

    nl_table_engine = NLSQLTableQueryEngine(index.sql_database, sql_only=True)
    task = nl_table_engine.aquery("test_table:user_id,foo")
    response = asyncio_run(task)
    assert str(response) == sql_to_test


def test_default_output_parser() -> None:
    """Test default output parser."""
    test_str = "user_id:2\n" "foo:bar\n" ",,testing:testing2..\n" "number:123,456,789\n"
    fields = default_output_parser(test_str)
    assert fields == {
        "user_id": "2",
        "foo": "bar",
        "testing": "testing2",
        "number": "123456789",
    }


def test_nl_query_engine_parser(
    patch_llm_predictor,
    patch_token_text_splitter,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test the sql response parser."""
    index_kwargs, _ = struct_kwargs
    docs = [Document(text="user_id:2,foo:bar"), Document(text="user_id:8,foo:hello")]
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData()
    table_name = "test_table"
    # NOTE: table is created by tying to metadata_obj
    Table(
        table_name,
        metadata_obj,
        Column("user_id", Integer, primary_key=True),
        Column("foo", String(16), nullable=False),
    )
    metadata_obj.create_all(engine)
    sql_database = SQLDatabase(engine)
    # NOTE: we can use the default output parser for this
    index = SQLStructStoreIndex.from_documents(
        docs,
        sql_database=sql_database,
        table_name=table_name,
        **index_kwargs,
    )
    nl_query_engine = NLStructStoreQueryEngine(index)

    # Response with SQLResult
    response = "SELECT * FROM table; SQLResult: [(1, 'value')]"
    assert nl_query_engine._parse_response_to_sql(response) == "SELECT * FROM table;"

    # Response with SQLQuery
    response = "SQLQuery: SELECT * FROM table;"
    assert nl_query_engine._parse_response_to_sql(response) == "SELECT * FROM table;"

    # Response with ```sql markdown
    response = "```sql\nSELECT * FROM table;\n```"
    assert nl_query_engine._parse_response_to_sql(response) == "SELECT * FROM table;"

    # Response with extra text after semi-colon
    response = "SELECT * FROM table; This is extra text."
    assert nl_query_engine._parse_response_to_sql(response) == "SELECT * FROM table;"

    # Response with escaped single quotes
    response = "SELECT * FROM table WHERE name = \\'O\\'Reilly\\';"
    assert (
        nl_query_engine._parse_response_to_sql(response)
        == "SELECT * FROM table WHERE name = ''O''Reilly'';"
    )

    # Response with escaped single quotes
    response = "SQLQuery: ```sql\nSELECT * FROM table WHERE name = \\'O\\'Reilly\\';\n``` Extra test SQLResult: [(1, 'value')]"
    assert (
        nl_query_engine._parse_response_to_sql(response)
        == "SELECT * FROM table WHERE name = ''O''Reilly'';"
    )
