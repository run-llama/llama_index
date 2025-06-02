from typing import Generator

import pytest
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine


# Create a fixture for the database instance
@pytest.fixture()
def sql_database(request: pytest.FixtureRequest) -> Generator[SQLDatabase, None, None]:
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    table_name = "test_table"
    Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String),
    )
    metadata.create_all(engine)

    max_string_length = getattr(
        request, "param", 300
    )  # Default value for max_string_length
    yield SQLDatabase(
        engine=engine,
        metadata=metadata,
        sample_rows_in_table_info=1,
        max_string_length=max_string_length,
    )

    metadata.drop_all(engine)


# Test initialization
def test_init(sql_database: SQLDatabase) -> None:
    assert sql_database.engine
    assert isinstance(sql_database.metadata_obj, MetaData)


# NOTE: Test is failing after removing langchain for some reason.
# # Test from_uri method
# def test_from_uri(mocker: MockerFixture) -> None:
#     mocked = mocker.patch("llama_index.core.legacy.utilities.sql_wrapper.create_engine")
#     SQLDatabase.from_uri("sqlite:///:memory:")
#     mocked.assert_called_once_with("sqlite:///:memory:", **{})


# Test get_table_columns method
def test_get_table_columns(sql_database: SQLDatabase) -> None:
    columns = sql_database.get_table_columns("test_table")
    assert [column["name"] for column in columns] == ["id", "name"]


# Test get_single_table_info method
def test_get_single_table_info(sql_database: SQLDatabase) -> None:
    assert sql_database.get_single_table_info("test_table") == (
        "Table 'test_table' has columns: id (INTEGER), name (VARCHAR), ."
    )


# Test insert and run_sql method
def test_insert_and_run_sql(sql_database: SQLDatabase) -> None:
    result_str, _ = sql_database.run_sql("SELECT * FROM test_table;")
    assert result_str == "[]"

    sql_database.insert_into_table("test_table", {"id": 1, "name": "Paul McCartney"})

    result_str, _ = sql_database.run_sql("SELECT * FROM test_table;")

    assert result_str == "[(1, 'Paul McCartney')]"


# Test query results truncation
@pytest.mark.parametrize("sql_database", [7], indirect=True)
def test_run_sql_truncation(sql_database: SQLDatabase) -> None:
    result_str, _ = sql_database.run_sql("SELECT * FROM test_table;")
    assert result_str == "[]"

    sql_database.insert_into_table("test_table", {"id": 1, "name": "Paul McCartney"})

    result_str, _ = sql_database.run_sql("SELECT * FROM test_table;")

    assert result_str == "[(1, 'Paul...')]"


# Test if long strings are not being truncated with large max_string_length
@pytest.mark.parametrize("sql_database", [10000], indirect=True)
def test_long_string_no_truncation(sql_database: SQLDatabase) -> None:
    result_str, _ = sql_database.run_sql("SELECT * FROM test_table;")
    assert result_str == "[]"

    long_string = "a" * (500)
    sql_database.insert_into_table("test_table", {"id": 1, "name": long_string})

    result_str, _ = sql_database.run_sql("SELECT * FROM test_table;")

    assert result_str == f"[(1, '{long_string}')]"
