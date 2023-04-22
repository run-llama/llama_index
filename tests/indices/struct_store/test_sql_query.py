import asyncio
from typing import Any, Dict, Tuple
from unittest.mock import patch

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
from gpt_index.indices.struct_store.base import default_output_parser
from gpt_index.indices.struct_store.sql import GPTSQLStructStoreIndex
from gpt_index.indices.struct_store.sql_query import (
    GPTNLStructStoreQueryEngine,
    GPTSQLStructStoreQueryEngine,
)
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase
from gpt_index.llm_predictor.base import LLMPredictor
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_predict import mock_llmpredictor_apredict


@patch_common
def test_sql_index_query(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test GPTSQLStructStoreIndex."""
    index_kwargs, query_kwargs = struct_kwargs
    docs = [Document(text="user_id:2,foo:bar"), Document(text="user_id:8,foo:hello")]
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData(bind=engine)
    table_name = "test_table"
    # NOTE: table is created by tying to metadata_obj
    Table(
        table_name,
        metadata_obj,
        Column("user_id", Integer, primary_key=True),
        Column("foo", String(16), nullable=False),
    )
    metadata_obj.create_all()
    sql_database = SQLDatabase(engine)
    # NOTE: we can use the default output parser for this
    index = GPTSQLStructStoreIndex.from_documents(
        docs, sql_database=sql_database, table_name=table_name, **index_kwargs
    )

    # query the index with SQL
    sql_query_engine = GPTSQLStructStoreQueryEngine(index, **query_kwargs)
    response = sql_query_engine.query("SELECT user_id, foo FROM test_table")
    assert str(response) == "[(2, 'bar'), (8, 'hello')]"

    # query the index with natural language
    nl_query_engine = GPTNLStructStoreQueryEngine(index, **query_kwargs)
    response = nl_query_engine.query("test_table:user_id,foo")
    assert str(response) == "[(2, 'bar'), (8, 'hello')]"


@patch_common
@patch.object(LLMPredictor, "apredict", side_effect=mock_llmpredictor_apredict)
def test_sql_index_async_query(
    _mock_async_predict: Any,
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test GPTSQLStructStoreIndex."""
    index_kwargs, query_kwargs = struct_kwargs
    docs = [Document(text="user_id:2,foo:bar"), Document(text="user_id:8,foo:hello")]
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData(bind=engine)
    table_name = "test_table"
    # NOTE: table is created by tying to metadata_obj
    Table(
        table_name,
        metadata_obj,
        Column("user_id", Integer, primary_key=True),
        Column("foo", String(16), nullable=False),
    )
    metadata_obj.create_all()
    sql_database = SQLDatabase(engine)
    # NOTE: we can use the default output parser for this
    index = GPTSQLStructStoreIndex.from_documents(
        docs, sql_database=sql_database, table_name=table_name, **index_kwargs
    )

    # query the index with SQL
    sql_query_engine = GPTSQLStructStoreQueryEngine(index, **query_kwargs)
    task = sql_query_engine.aquery("SELECT user_id, foo FROM test_table")
    response = asyncio.run(task)
    assert str(response) == "[(2, 'bar'), (8, 'hello')]"

    # query the index with natural language
    nl_query_engine = GPTNLStructStoreQueryEngine(index, **query_kwargs)
    task = nl_query_engine.aquery("test_table:user_id,foo")
    response = asyncio.run(task)
    assert str(response) == "[(2, 'bar'), (8, 'hello')]"


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
