import pytest
from google.cloud import bigquery
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from llama_index.vector_stores.bigquery import BigQueryVectorStore
from sql_assertions import assert_equivalent_sql_statements


def test_delete_nodes_with_filters_generates_correct_sql_and_params(
    vector_store: BigQueryVectorStore,
):
    """It should execute a parameterized DELETE query with correct filtering criteria."""
    # Given filter criteria
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="author", value="ceo@company.com"),
            MetadataFilter(key="author", value="cfo@company.com"),
        ],
        condition="or",
    )
    node_ids = ["node1", "node2"]

    # When `delete` is called with the filter criteria
    vector_store.delete_nodes(node_ids, filters)

    # Then it should call BigQuery with the correct query parameters
    vector_store.client.query_and_wait.assert_called_once()
    args, kwargs = vector_store.client.query_and_wait.call_args
    actual_query = args[0]
    job_config = kwargs["job_config"]

    expected_query_params = [
        bigquery.ScalarQueryParameter(
            name=None, type_="STRING", value="ceo@company.com"
        ),
        bigquery.ScalarQueryParameter(
            name=None, type_="STRING", value="cfo@company.com"
        ),
        bigquery.ArrayQueryParameter(
            name="node_ids", array_type="STRING", values=["node1", "node2"]
        ),
    ]
    assert isinstance(job_config, bigquery.QueryJobConfig)
    assert job_config.query_parameters == expected_query_params

    # And the actual SQL query should match the expected SQL query
    expected_query = """
    DELETE FROM `mock-project.mock_dataset.mock_table`
    WHERE (SAFE.JSON_VALUE(metadata, '$."author"') = ? OR SAFE.JSON_VALUE(metadata, '$."author"') = ?)
    AND node_id IN UNNEST(@node_ids);
    """
    assert_equivalent_sql_statements(actual_query, expected_query)


def test_delete_nodes_without_arguments_raises_value_error(
    vector_store: BigQueryVectorStore,
):
    """It should raise a ValueError when neither `node_ids` nor `filters` is provided."""
    # When `delete_nodes` is called without arguments, it should raise a ValueError
    with pytest.raises(ValueError):
        vector_store.delete_nodes()
