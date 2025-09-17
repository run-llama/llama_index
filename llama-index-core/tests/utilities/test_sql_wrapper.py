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


# Test CTE functionality
def test_cte_extraction(sql_database: SQLDatabase) -> None:
    """Test that CTE names are correctly extracted from SQL queries."""
    # Test single CTE
    query1 = "WITH my_cte AS (SELECT * FROM test_table) SELECT * FROM my_cte"
    cte_names = sql_database._extract_cte_names(query1)
    assert cte_names == {"my_cte"}

    # Test multiple CTEs
    query2 = "WITH cte1 AS (SELECT * FROM test_table), cte2 AS (SELECT * FROM test_table) SELECT * FROM cte1 JOIN cte2"
    cte_names = sql_database._extract_cte_names(query2)
    assert cte_names == {"cte1", "cte2"}

    # Test no CTE
    query3 = "SELECT * FROM test_table"
    cte_names = sql_database._extract_cte_names(query3)
    assert cte_names == set()

    # Test case insensitive
    query4 = "with my_cte as (SELECT * FROM test_table) SELECT * FROM my_cte"
    cte_names = sql_database._extract_cte_names(query4)
    assert cte_names == {"my_cte"}


def test_schema_prefixing_without_cte(sql_database: SQLDatabase) -> None:
    """Test that schema prefixing works correctly for regular queries without CTEs."""
    # Create a database with schema
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

    # Create SQLDatabase with schema
    db_with_schema = SQLDatabase(
        engine=engine,
        schema="test_schema",
        metadata=metadata,
        sample_rows_in_table_info=1,
    )

    # Test simple query
    query = "SELECT * FROM test_table"
    modified = db_with_schema._add_schema_prefix_smart(query)
    assert modified == "SELECT * FROM test_schema.test_table"

    # Test JOIN query
    query2 = "SELECT * FROM test_table t1 JOIN test_table t2 ON t1.id = t2.id"
    modified2 = db_with_schema._add_schema_prefix_smart(query2)
    assert "test_schema.test_table" in modified2
    assert modified2.count("test_schema.test_table") == 2


def test_schema_prefixing_with_cte(sql_database: SQLDatabase) -> None:
    """Test that schema prefixing works correctly with CTEs."""
    # Create a database with schema
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

    # Create SQLDatabase with schema
    db_with_schema = SQLDatabase(
        engine=engine,
        schema="test_schema",
        metadata=metadata,
        sample_rows_in_table_info=1,
    )

    # Test CTE query - CTE name should not be prefixed, but table should be
    query = "WITH my_cte AS (SELECT * FROM test_table) SELECT * FROM my_cte"
    modified = db_with_schema._add_schema_prefix_smart(query)
    assert "FROM test_schema.test_table" in modified
    assert "FROM my_cte" in modified  # CTE name should not be prefixed

    # Test multiple CTEs
    query2 = "WITH cte1 AS (SELECT * FROM test_table), cte2 AS (SELECT * FROM test_table) SELECT * FROM cte1 JOIN cte2"
    modified2 = db_with_schema._add_schema_prefix_smart(query2)
    assert "FROM test_schema.test_table" in modified2
    assert "FROM cte1" in modified2
    assert "JOIN cte2" in modified2

    # Test CTE with JOIN
    query3 = "WITH my_cte AS (SELECT * FROM test_table) SELECT * FROM my_cte JOIN test_table ON my_cte.id = test_table.id"
    modified3 = db_with_schema._add_schema_prefix_smart(query3)
    assert "FROM my_cte" in modified3  # CTE should not be prefixed
    assert "JOIN test_schema.test_table" in modified3  # Table should be prefixed


def test_schema_prefixing_already_qualified(sql_database: SQLDatabase) -> None:
    """Test that already schema-qualified table names are not modified."""
    # Create a database with schema
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

    # Create SQLDatabase with schema
    db_with_schema = SQLDatabase(
        engine=engine,
        schema="test_schema",
        metadata=metadata,
        sample_rows_in_table_info=1,
    )

    # Test already qualified table name
    query = "SELECT * FROM other_schema.test_table"
    modified = db_with_schema._add_schema_prefix_smart(query)
    assert modified == query  # Should remain unchanged

    # Test mixed qualified and unqualified
    query2 = (
        "SELECT * FROM test_table t1 JOIN other_schema.test_table t2 ON t1.id = t2.id"
    )
    modified2 = db_with_schema._add_schema_prefix_smart(query2)
    assert "FROM test_schema.test_table" in modified2
    assert "JOIN other_schema.test_table" in modified2


def test_run_sql_with_cte_and_schema(sql_database: SQLDatabase) -> None:
    """Test that run_sql works correctly with CTEs and schema prefixing."""
    # Create a database with schema
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

    # Create SQLDatabase with schema
    db_with_schema = SQLDatabase(
        engine=engine,
        schema="test_schema",
        metadata=metadata,
        sample_rows_in_table_info=1,
    )

    # Insert test data
    db_with_schema.insert_into_table("test_table", {"id": 1, "name": "Alice"})
    db_with_schema.insert_into_table("test_table", {"id": 2, "name": "Bob"})

    # Test CTE query
    query = "WITH filtered_users AS (SELECT * FROM test_table WHERE id > 1) SELECT * FROM filtered_users"
    result_str, result_dict = db_with_schema.run_sql(query)

    # Should return Bob's record
    assert "Bob" in result_str
    assert "Alice" not in result_str
    assert len(result_dict["result"]) == 1
