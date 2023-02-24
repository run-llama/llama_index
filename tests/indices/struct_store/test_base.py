"""Test struct store indices."""

import re
from typing import Any, Dict, List, Optional, Tuple

import pytest
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    column,
    create_engine,
    delete,
    select,
)

from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.struct_store.base import default_output_parser
from gpt_index.indices.struct_store.sql import (
    GPTSQLStructStoreIndex,
    SQLContextContainerBuilder,
)
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase
from gpt_index.readers.schema.base import Document
from gpt_index.schema import BaseDocument
from tests.mock_utils.mock_decorator import patch_common
from tests.mock_utils.mock_prompts import (
    MOCK_REFINE_PROMPT,
    MOCK_SCHEMA_EXTRACT_PROMPT,
    MOCK_TABLE_CONTEXT_PROMPT,
    MOCK_TEXT_QA_PROMPT,
)


def _mock_output_parser(output: str) -> Optional[Dict[str, Any]]:
    """Mock output parser.

    Split via commas instead of newlines, in order to fit
    the format of the mock test document (newlines create
    separate text chunks in the testing code).

    """
    tups = output.split(",")

    fields = {}
    for tup in tups:
        if ":" in tup:
            tokens = tup.split(":")
            field = re.sub(r"\W+", "", tokens[0])
            value = re.sub(r"\W+", "", tokens[1])
            fields[field] = value
    return fields


@pytest.fixture
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    # NOTE: QuestionAnswer and Refine templates aren't technically used
    index_kwargs = {
        "schema_extract_prompt": MOCK_SCHEMA_EXTRACT_PROMPT,
        "output_parser": _mock_output_parser,
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
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test GPTSQLStructStoreIndex."""
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
    docs = [Document(text="user_id:2,foo:bar"), Document(text="user_id:8,foo:hello")]
    sql_database = SQLDatabase(engine)
    index = GPTSQLStructStoreIndex(
        docs, sql_database=sql_database, table_name=table_name, **index_kwargs
    )

    # test that the document is inserted
    stmt = select([column("user_id"), column("foo")]).select_from(test_table)
    engine = index.sql_database.engine
    with engine.connect() as connection:
        results = connection.execute(stmt).fetchall()
        assert results == [(2, "bar"), (8, "hello")]

    # try with documents with more text chunks
    delete_stmt = delete(test_table)
    with engine.connect() as connection:
        connection.execute(delete_stmt)
    docs = [Document(text="user_id:2\nfoo:bar"), Document(text="user_id:8\nfoo:hello")]
    index = GPTSQLStructStoreIndex(
        docs, sql_database=sql_database, table_name=table_name, **index_kwargs
    )
    # test that the document is inserted
    stmt = select([column("user_id"), column("foo")]).select_from(test_table)
    engine = index.sql_database.engine
    with engine.connect() as connection:
        results = connection.execute(stmt).fetchall()
        assert results == [(2, "bar"), (8, "hello")]


def _delete_table_items(engine: Any, table: Table) -> None:
    """Delete items from a table."""
    delete_stmt = delete(table)
    with engine.connect() as connection:
        connection.execute(delete_stmt)


@patch_common
def test_sql_index_with_context(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test GPTSQLStructStoreIndex."""
    # test setting table_context_dict
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
    docs = [Document(text="user_id:2,foo:bar"), Document(text="user_id:8,foo:hello")]
    sql_database = SQLDatabase(engine)
    table_context_dict = {"test_table": "test_table_context"}

    # test with ignore_db_schema=True
    sql_context_container = SQLContextContainerBuilder(
        sql_database, context_dict=table_context_dict
    ).build_context_container(ignore_db_schema=True)

    index = GPTSQLStructStoreIndex(
        docs,
        sql_database=sql_database,
        table_name=table_name,
        sql_context_container=sql_context_container,
        **index_kwargs
    )
    assert index.sql_context_container.context_dict == table_context_dict
    _delete_table_items(engine, test_table)

    # test with ignore_db_schema=False (default)
    sql_database = SQLDatabase(engine)
    sql_context_container = SQLContextContainerBuilder(
        sql_database, context_dict=table_context_dict
    ).build_context_container()

    index = GPTSQLStructStoreIndex(
        docs,
        sql_database=sql_database,
        table_name=table_name,
        sql_context_container=sql_context_container,
        **index_kwargs
    )
    for k, v in table_context_dict.items():
        context_dict = index.sql_context_container.context_dict
        assert context_dict is not None
        assert len(context_dict[k]) > len(v)
        assert v in context_dict[k]
    _delete_table_items(engine, test_table)

    # test setting sql_context_builder
    sql_database = SQLDatabase(engine)
    # this should cause the mock QuestionAnswer prompt to run
    context_documents_dict: Dict[str, List[BaseDocument]] = {
        "test_table": [Document("test_table_context")]
    }
    sql_context_builder = SQLContextContainerBuilder.from_documents(
        context_documents_dict,
        sql_database=sql_database,
        table_context_prompt=MOCK_TABLE_CONTEXT_PROMPT,
        table_context_task="extract_test",
    )
    sql_context_container = sql_context_builder.build_context_container(
        ignore_db_schema=True
    )
    index = GPTSQLStructStoreIndex(
        docs,
        sql_database=sql_database,
        table_name=table_name,
        sql_context_container=sql_context_container,
        **index_kwargs
    )
    assert index.sql_context_container.context_dict == {
        "test_table": "extract_test:test_table_context"
    }

    # test error if both are set
    # TODO:


@patch_common
def test_sql_index_with_derive_index(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test derive index."""
    # test setting table_context_dict
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData(bind=engine)
    table_name = "test_table"
    Table(
        table_name,
        metadata_obj,
        Column("user_id", Integer, primary_key=True),
        Column("foo", String(16), nullable=False),
    )
    metadata_obj.create_all()
    # NOTE: we can use the default output parser for this
    sql_database = SQLDatabase(engine)
    table_context_dict = {"test_table": "test_table_context"}

    context_builder = SQLContextContainerBuilder(
        sql_database, context_dict=table_context_dict
    )
    context_index_no_ignore = context_builder.derive_index_from_context(
        GPTListIndex,
    )
    context_index_with_ignore = context_builder.derive_index_from_context(
        GPTListIndex, ignore_db_schema=True
    )
    assert len(context_index_with_ignore.index_struct.nodes) == 1
    assert len(context_index_no_ignore.index_struct.nodes) > 1


@patch_common
def test_sql_index_with_index_context(
    _mock_init: Any,
    _mock_predict: Any,
    _mock_total_tokens_used: Any,
    _mock_split_text_overlap: Any,
    _mock_split_text: Any,
    struct_kwargs: Tuple[Dict, Dict],
) -> None:
    """Test GPTSQLStructStoreIndex."""
    # test setting table_context_dict
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData(bind=engine)
    table_name = "test_table"
    Table(
        table_name,
        metadata_obj,
        Column("user_id", Integer, primary_key=True),
        Column("foo", String(16), nullable=False),
    )
    metadata_obj.create_all()
    # NOTE: we can use the default output parser for this
    index_kwargs, _ = struct_kwargs
    docs = [Document(text="user_id:2,foo:bar"), Document(text="user_id:8,foo:hello")]
    sql_database = SQLDatabase(engine)
    table_context_dict = {"test_table": "test_table_context"}

    context_builder = SQLContextContainerBuilder(
        sql_database, context_dict=table_context_dict
    )
    context_index = context_builder.derive_index_from_context(
        GPTListIndex, ignore_db_schema=True
    )
    # NOTE: the response only contains the first line (metadata), since
    # with the mock patch, newlines are treated as separate calls.
    context_response = context_builder.query_index_for_context(
        context_index,
        "Context query?",
        query_tmpl="{orig_query_str}",
        store_context_str=True,
    )
    sql_context_container = context_builder.build_context_container(
        ignore_db_schema=True
    )
    assert context_response == "Context query?:table_name: test_table"
    assert sql_context_container.context_str == context_response

    index = GPTSQLStructStoreIndex(
        docs,
        sql_database=sql_database,
        table_name=table_name,
        sql_context_container=sql_context_container,
        **index_kwargs
    )
    # just assert this runs
    index.query("test_table:foo", mode="default")


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
    index = GPTSQLStructStoreIndex(
        docs, sql_database=sql_database, table_name=table_name, **index_kwargs
    )

    # query the index with SQL
    response = index.query(
        "SELECT user_id, foo FROM test_table", mode="sql", **query_kwargs
    )
    assert response.response == "[(2, 'bar'), (8, 'hello')]"

    # query the index with natural language
    response = index.query("test_table:user_id,foo", mode="default", **query_kwargs)
    assert response.response == "[(2, 'bar'), (8, 'hello')]"


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
