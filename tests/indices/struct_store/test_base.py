"""Test struct store indices."""

from typing import Dict, Tuple

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select

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


def test_sql_index(struct_kwargs: Tuple[Dict, Dict]) -> None:
    """Test GPTSQLStructStoreIndex."""
    doc = Document(text="user_id:2,foo:bar")
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
        [doc], sql_engine=engine, table_name=table_name, **index_kwargs
    )

    # test that the document is inserted
    stmt = select(["user_id", "foo"])
    engine = index.sql_database.engine
    with engine.connect() as connection:
        results = connection.execute(stmt).fetchall()
        print(results)
        raise Exception
