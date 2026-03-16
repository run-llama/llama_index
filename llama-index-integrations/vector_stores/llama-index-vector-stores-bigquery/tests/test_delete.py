from google.cloud import bigquery

from llama_index.vector_stores.bigquery import BigQueryVectorStore
from sql_assertions import assert_equivalent_sql_statements


def test_delete_generates_correct_sql_and_params(vector_store: BigQueryVectorStore):
    """It should execute a parameterized DELETE query to remove nodes with the specified `ref_doc_id`."""
    # Given a `ref_doc_id` to delete
    ref_doc_id = "doc-1"

    # When `delete` is called with the `ref_doc_id`
    vector_store.delete(ref_doc_id)

    # Then it should call BigQuery with the correct query parameters
    vector_store.client.query_and_wait.assert_called_once()
    args, kwargs = vector_store.client.query_and_wait.call_args
    actual_query = args[0]
    job_config = kwargs["job_config"]

    expected_query_params = [
        bigquery.ScalarQueryParameter(name="to_delete", type_="STRING", value="doc-1")
    ]
    assert isinstance(job_config, bigquery.QueryJobConfig)
    assert job_config.query_parameters == expected_query_params

    # And the actual SQL query should match the expected SQL query
    expected_query = """
    DELETE FROM `mock-project.mock_dataset.mock_table`
    WHERE  SAFE.JSON_VALUE(metadata, '$."doc_id"') = @to_delete;
    """
    assert_equivalent_sql_statements(actual_query, expected_query)
