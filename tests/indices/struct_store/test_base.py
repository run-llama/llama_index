"""Test struct store indices."""

from typing import Any, Dict, Tuple

import pytest
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    column,
    create_engine,
    select,
)

from gpt_index.indices.struct_store.sql import GPTSQLStructStoreIndex
from gpt_index.readers.schema.base import Document
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_prompts import (
    MOCK_REFINE_PROMPT,
    MOCK_SCHEMA_EXTRACT_PROMPT,
    MOCK_TEXT_QA_PROMPT,
)


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    # NOTE: QuestionAnswer and Refine templates aren't technically used
    index_kwargs = {
        "schema_extract_prompt": MOCK_SCHEMA_EXTRACT_PROMPT,
    }
    query_kwargs = {
        "text_qa_template": MOCK_TEXT_QA_PROMPT,
        "refine_template": MOCK_REFINE_PROMPT,
    }
    return index_kwargs, query_kwargs


@patch_common
def test_sql_index(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_splitter: Any,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test GPTSQLStructStoreIndex."""
    docs = [Document(text="user_id:2,foo:bar"), Document(text="user_id:8,foo:hello")]
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData(bind=engine)
    table_name = "test_table"
    test_table = Table(
        table_name,
        metadata_obj,
        Column("user_id", Integer, primary_key=True),
        Column("foo", String(16), nullable=False),
    )
    metadata_obj.create_all()
    # NOTE: we can use the default output parser for this
    index_kwargs, _ = struct_kwargs
    index = GPTSQLStructStoreIndex(
        docs, sql_engine=engine, table_name=table_name, **index_kwargs
    )

    # test that the document is inserted
    stmt = select([column("user_id"), column("foo")]).select_from(test_table)
    engine = index.sql_database.engine
    with engine.connect() as connection:
        results = connection.execute(stmt).fetchall()
        assert results == [(2, "bar"), (8, "hello")]


@patch_common
def test_sql_index_query(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_splitter: Any,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test GPTSQLStructStoreIndex."""
    index_kwargs, query_kwargs = struct_kwargs
    docs = [Document(text="user_id:2,foo:bar"), Document(text="user_id:8,foo:hello")]
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData(bind=engine)
    table_name = "test_table"
    test_table = Table(
        table_name,
        metadata_obj,
        Column("user_id", Integer, primary_key=True),
        Column("foo", String(16), nullable=False),
    )
    metadata_obj.create_all()
    # NOTE: we can use the default output parser for this
    index = GPTSQLStructStoreIndex(
        docs, sql_engine=engine, table_name=table_name, **index_kwargs
    )

    # query the index with SQL
    response = index.query(
        "SELECT user_id, foo FROM test_table", mode="sql", **query_kwargs
    )
    assert response.response == "[(2, 'bar'), (8, 'hello')]"

    # query the index with natural language
    response = index.query("test_table:user_id,foo", mode="default", **query_kwargs)
    assert response.response == "[(2, 'bar'), (8, 'hello')]"
