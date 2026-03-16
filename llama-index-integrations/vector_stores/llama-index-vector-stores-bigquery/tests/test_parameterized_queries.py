import pytest
from google.cloud import bigquery
from google.cloud.bigquery import ArrayQueryParameter, ScalarQueryParameter
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from llama_index.vector_stores.bigquery.utils import build_where_clause_and_params
from sql_assertions import assert_equivalent_sql_statements


def test_build_where_clause_and_params():
    """It should build a query from MetadataFilters and node IDs"""
    # Given a list of `filters` and a list of `node_ids`
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="author", value="ceo@company.com"),
            MetadataFilter(key="author", value="cfo@company.com"),
        ],
        condition="or",
    )
    node_ids = ["node1", "node2"]

    # When the WHERE clause and query parameters are built
    where_clause, query_params = build_where_clause_and_params(node_ids, filters)

    # Then the SQL query WHERE clause should reflect both `filters` and `node_ids`,
    query = f"""
    SELECT  *
    FROM    table
    WHERE   {where_clause}
    """

    expected_query = """
    SELECT  *
    FROM    table
    WHERE
        (SAFE.JSON_VALUE(metadata, '$."author"') = ? OR SAFE.JSON_VALUE(metadata, '$."author"') = ?)
        AND node_id IN UNNEST(@node_ids)
    """
    assert_equivalent_sql_statements(query, expected_query)

    # And the parameters should match the expected values
    expected_query_params = [
        ScalarQueryParameter(None, "STRING", "ceo@company.com"),
        ScalarQueryParameter(None, "STRING", "cfo@company.com"),
        ArrayQueryParameter("node_ids", "STRING", ["node1", "node2"]),
    ]
    assert query_params == expected_query_params


def test_build_where_clause_and_params_with_nested_filters():
    """It should build a query from nested MetadataFilters and node IDs"""
    # Given a nested list of `filters` and a list of `node_ids`
    filters = MetadataFilters(
        filters=[
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="commit_date", value="2023-08-01", operator=">="
                    ),
                    MetadataFilter(
                        key="commit_date", value="2023-08-15", operator="<="
                    ),
                ],
                condition="and",
            ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="author", value="mats@timescale.com"),
                    MetadataFilter(key="author", value="sven@timescale.com"),
                ],
                condition="or",
            ),
        ],
        condition="and",
    )

    node_ids = ["node1", "node2"]

    # When the WHERE clause and query parameters are built
    where_clause, query_params = build_where_clause_and_params(node_ids, filters)

    # Then the SQL query WHERE clause should reflect both `filters` and `node_ids`,
    query = f"""
    SELECT  *
    FROM    table
    WHERE   {where_clause}
    """

    expected_query = """
    SELECT  *
    FROM    table
    WHERE ((
      SAFE.JSON_VALUE(metadata, '$."commit_date"') >= ? AND
      SAFE.JSON_VALUE(metadata, '$."commit_date"') <= ?
    ) AND (
      SAFE.JSON_VALUE(metadata, '$."author"') = ? OR
      SAFE.JSON_VALUE(metadata, '$."author"') = ?
    )) AND node_id IN UNNEST(@node_ids)
    """
    assert_equivalent_sql_statements(query, expected_query)

    # And the parameters should match the expected values
    expected_query_params = [
        ScalarQueryParameter(None, "STRING", "2023-08-01"),
        ScalarQueryParameter(None, "STRING", "2023-08-15"),
        ScalarQueryParameter(None, "STRING", "mats@timescale.com"),
        ScalarQueryParameter(None, "STRING", "sven@timescale.com"),
        ArrayQueryParameter("node_ids", "STRING", ["node1", "node2"]),
    ]
    assert query_params == expected_query_params


@pytest.mark.parametrize(
    (
        "key",
        "value",
        "operator",
        "expected_where_clause",
        "expected_query_parameter",
    ),
    [
        (
            "magna_carta",
            "1215-12-15",
            "==",
            "(SAFE.JSON_VALUE(metadata, '$.\"magna_carta\"') = ?)",
            [bigquery.ScalarQueryParameter(None, "STRING", "1215-12-15")],
        ),
        (
            "ramanujan",
            1729,
            "!=",
            "(SAFE.JSON_VALUE(metadata, '$.\"ramanujan\"') != ?)",
            [bigquery.ScalarQueryParameter(None, "STRING", 1729)],
        ),
        (
            "salary",
            50_000,
            ">",
            "(SAFE.JSON_VALUE(metadata, '$.\"salary\"') > ?)",
            [bigquery.ScalarQueryParameter(None, "STRING", 50_000)],
        ),
        (
            "height",
            6.5,
            ">=",
            "(SAFE.JSON_VALUE(metadata, '$.\"height\"') >= ?)",
            [bigquery.ScalarQueryParameter(None, "STRING", 6.5)],
        ),
        (
            "speed",
            100,
            "<",
            "(SAFE.JSON_VALUE(metadata, '$.\"speed\"') < ?)",
            [bigquery.ScalarQueryParameter(None, "STRING", 100)],
        ),
        (
            "weight",
            120,
            "<=",
            "(SAFE.JSON_VALUE(metadata, '$.\"weight\"') <= ?)",
            [bigquery.ScalarQueryParameter(None, "STRING", 120)],
        ),
        (
            "name",
            ["Alan Turing", "Grace Hopper"],
            "in",
            "(SAFE.JSON_VALUE(metadata, '$.\"name\"') IN UNNEST(?))",
            [
                bigquery.ArrayQueryParameter(
                    None, "STRING", ["Alan Turing", "Grace Hopper"]
                )
            ],
        ),
        (
            "numbers",
            [10, 20, 30],
            "nin",
            "(SAFE.JSON_VALUE(metadata, '$.\"numbers\"') NOT IN UNNEST(?))",
            [bigquery.ArrayQueryParameter(None, "STRING", [10, 20, 30])],
        ),
        (
            "foo",
            None,
            "is_empty",
            "(JSON_TYPE(JSON_QUERY(metadata, '$.\"foo\"')) = 'null')",
            [],
        ),
    ],
)
def test_build_where_clause_and_params_with_single_filter(
    key, value, operator, expected_where_clause, expected_query_parameter
):
    """It should construct a parameterized SQL WHERE clause and corresponding query parameters."""
    # Given a MetadataFilters instance
    filters = MetadataFilters(
        filters=[MetadataFilter(key=key, value=value, operator=operator)]
    )

    # When the WHERE clause and query parameters are built
    where_clause, query_params = build_where_clause_and_params(filters=filters)

    # Then the WHERE clause should reflect the MetadataFilters
    assert where_clause == expected_where_clause

    # And the parameters should match the expected values
    assert query_params == expected_query_parameter


def test_build_where_clause_and_params_without_args():
    """It should return empty where clause and parameters if no arguments are provided."""
    # When there are no parameters
    where_clause, query_params = build_where_clause_and_params()

    # Then the results should be empty
    assert query_params == []
    assert where_clause == ""
